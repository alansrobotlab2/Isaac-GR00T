#!/usr/bin/env python3
"""
Comprehensive backbone pipeline benchmark:
  A: PyTorch BF16 flash_attention_2 (reference)
  B: PyTorch FP32 eager attention
  C: ONNX Runtime
  D: TRT FP16 engine
  E: TRT INT8 engine

Tests each stage against the flash reference to pinpoint where quality degrades.
Reports accuracy (MSE, MAE, cosine similarity, max abs error) and latency per stage.

Must run inside Docker for TRT stages. PyTorch and ONNX stages work on host.

Usage:
    python scripts/deployment/benchmark_backbone_pipeline.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --onnx_path ./groot_n1d6_onnx/backbone_model.onnx \
        --trt_fp16_path ./groot_n1d6_onnx/backbone_fp16_agx.trt \
        --trt_int8_path ./groot_n1d6_onnx/backbone_int8_agx.trt
"""

import argparse
import gc
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Conditional TRT import — stages D/E only available inside Docker
TRT_AVAILABLE = False
try:
    import tensorrt as trt  # noqa: F401

    TRT_AVAILABLE = True
except ImportError:
    pass

# Conditional ORT import — stage C needs onnxruntime
ORT_AVAILABLE = False
try:
    import onnxruntime as ort  # noqa: F401

    ORT_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers (self-contained to avoid fragile cross-script imports)
# ---------------------------------------------------------------------------


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def swap_flash_to_eager(model):
    count = 0
    for _name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation == "flash_attention_2":
                module.config._attn_implementation = "eager"
                count += 1
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation == "flash_attention_2":
                module._attn_implementation = "eager"
                count += 1
    logger.info(f"Swapped {count} flash_attention_2 -> eager")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(a.shape[0], -1).float()
    b_flat = b.reshape(b.shape[0], -1).float()
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    ref_f = ref.float()
    test_f = test.float()
    diff = ref_f - test_f
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cos_sim": cosine_sim(ref, test),
    }


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        if key == "max_abs":
            agg[key] = max(vals)
        elif key == "cos_sim":
            agg[key] = float(np.mean(vals))
            agg["cos_sim_min"] = min(vals)
        else:
            agg[key] = float(np.mean(vals))
    agg["n_samples"] = len(all_metrics)
    return agg


def benchmark_latency(run_fn, warmup=5, num_iters=20) -> dict:
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def prepare_observation(policy, dataset, traj_idx, step_idx):
    """Prepare a single observation for inference."""
    traj = dataset[traj_idx]
    modality_config = policy.get_modality_config()
    modality_configs = {k: v for k, v in modality_config.items() if k != "action"}

    data_point = extract_step_data(traj, step_idx, modality_configs, policy.embodiment_tag)

    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for key in modality_config["language"].modality_keys:
        obs[key] = data_point.text

    # Parse to batched format
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_config[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def collect_samples(policy, dataset, num_trajs=5, steps_per_traj=2):
    """Collect backbone input/output samples via hooks on the flash reference model."""
    logger.info(f"Collecting {num_trajs * steps_per_traj} samples "
                f"({num_trajs} trajs x {steps_per_traj} steps)...")

    samples = []
    max_traj = min(num_trajs, len(dataset))

    for traj_idx in range(max_traj):
        traj = dataset[traj_idx]
        traj_len = len(traj)
        step_indices = list(range(0, min(traj_len, steps_per_traj * 16), 16))[:steps_per_traj]

        for step_idx in step_indices:
            obs = prepare_observation(policy, dataset, traj_idx, step_idx)

            # Run through VLA processor + collation (same path as policy.get_action)
            unbatched_obs = []
            batch_size = obs["video"][list(obs["video"].keys())[0]].shape[0]
            for i in range(batch_size):
                unbatched_obs.append({
                    "video": {k: v[i] for k, v in obs["video"].items()},
                    "state": {k: v[i] for k, v in obs["state"].items()},
                    "language": {k: v[i] for k, v in obs["language"].items()},
                })

            processed = []
            for o in unbatched_obs:
                vla = VLAStepData(
                    images=o["video"],
                    states=o["state"],
                    actions={},
                    text=o["language"][policy.language_key][0],
                    embodiment=policy.embodiment_tag,
                )
                processed.append(policy.processor(
                    [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
                ))

            collated = policy.collate_fn(processed)["inputs"]
            collated = _rec_to_dtype(collated, dtype=torch.bfloat16)

            with torch.inference_mode():
                bb_inputs, _ = policy.model.prepare_input(collated)
                ref_out = policy.model.backbone(bb_inputs)

            # Store inputs (on CPU) and reference output
            pv = bb_inputs["pixel_values"]
            samples.append({
                "input_ids": bb_inputs["input_ids"].detach().cpu().clone(),
                "attention_mask": bb_inputs["attention_mask"].detach().cpu().clone(),
                "pixel_values": [t.detach().cpu().clone() for t in pv] if isinstance(pv, list) else pv.detach().cpu().clone(),
                "ref_features": ref_out.backbone_features.detach().cpu().float(),
            })

    logger.info(f"Collected {len(samples)} samples, "
                f"feature shape: {samples[0]['ref_features'].shape}")
    return samples


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_stage_eager(policy, samples):
    """Stage B: PyTorch eager + FP32."""
    logger.info("[Stage B] PyTorch FP32 eager attention...")
    backbone_eager = deepcopy(policy.model.backbone)
    swap_flash_to_eager(backbone_eager)
    backbone_eager = backbone_eager.float().cuda().eval()

    results = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.float().cuda() for t in pv]
        else:
            pv_gpu = pv.float().cuda()
        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            out = backbone_eager(bb_in)
        results.append(out.backbone_features.detach().cpu().float())

    del backbone_eager
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_stage_onnx(onnx_path, samples):
    """Stage C: ONNX Runtime."""
    if not ORT_AVAILABLE:
        logger.warning("onnxruntime not available, skipping Stage C")
        return None
    if not os.path.exists(onnx_path):
        logger.warning(f"ONNX file not found: {onnx_path}, skipping Stage C")
        return None

    logger.info(f"[Stage C] ONNX Runtime: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = sess.get_providers()
    logger.info(f"  Active providers: {active}")

    # Auto-detect expected dtype and rank from ONNX model input metadata
    onnx_dtype_map = {"tensor(float16)": torch.float16, "tensor(float)": torch.float32}
    pv_input = [inp for inp in sess.get_inputs() if inp.name == "pixel_values"][0]
    onnx_pv_dtype = onnx_dtype_map.get(pv_input.type, torch.float32)
    onnx_pv_rank = len(pv_input.shape)  # 4D=[N,C,H,W] or 5D=[N,1,C,H,W]
    logger.info(f"  ONNX pixel_values dtype: {pv_input.type} -> torch {onnx_pv_dtype}, rank={onnx_pv_rank}")

    results = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            if onnx_pv_rank == 5:
                # ONNX expects [num_frames, batch, C, H, W] — stack preserving batch dim
                pv_np = torch.stack(pv, dim=0).to(onnx_pv_dtype).numpy()
            else:
                # ONNX expects [num_frames, C, H, W] — cat along frame dim
                pv_np = torch.cat(pv, dim=0).to(onnx_pv_dtype).numpy()
        else:
            pv_np = pv.to(onnx_pv_dtype).numpy()

        ort_out = sess.run(None, {
            "pixel_values": pv_np,
            "input_ids": s["input_ids"].numpy(),
            "attention_mask": s["attention_mask"].numpy(),
        })
        results.append(torch.from_numpy(ort_out[0]).float())

    del sess
    gc.collect()
    return results


def run_stage_trt(engine_path, samples, label="TRT"):
    """Stage D/E: TensorRT backbone engine."""
    if not TRT_AVAILABLE:
        logger.warning(f"TensorRT not available, skipping {label}")
        return None
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found: {engine_path}, skipping {label}")
        return None

    logger.info(f"[{label}] Loading: {engine_path}")

    # Import TRT wrapper from standalone_inference_script
    import importlib.util
    script_path = os.path.join(os.path.dirname(__file__), "standalone_inference_script.py")
    spec = importlib.util.spec_from_file_location("sis", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TensorRTBackboneWrapper = mod.TensorRTBackboneWrapper

    trt_wrapper = TensorRTBackboneWrapper(engine_path, device=0)

    # Cast pixel_values to the dtype expected by the TRT engine
    pv_dtype = trt_wrapper.input_dtypes.get("pixel_values", torch.float32)
    logger.info(f"  Engine expects pixel_values dtype: {pv_dtype}")

    results = []
    for s in samples:
        input_ids = s["input_ids"].cuda()
        attention_mask = s["attention_mask"].cuda()
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pixel_values = [t.to(dtype=pv_dtype).cuda() for t in pv]
        else:
            pixel_values = pv.to(dtype=pv_dtype).cuda()

        hidden_states, _attn_mask, _image_mask = trt_wrapper(
            input_ids, attention_mask, pixel_values
        )
        torch.cuda.synchronize()
        results.append(hidden_states.detach().cpu().float())

    del trt_wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Backbone pipeline benchmark")
    parser.add_argument("--model_path", default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--onnx_path", default="./groot_n1d6_onnx/backbone_model.onnx")
    parser.add_argument("--trt_fp16_path", default="./groot_n1d6_onnx/backbone_fp16_agx.trt")
    parser.add_argument("--trt_int8_path", default="./groot_n1d6_onnx/backbone_int8_agx.trt")
    parser.add_argument("--video_backend", default="torchcodec")
    parser.add_argument("--num_trajs", type=int, default=5)
    parser.add_argument("--steps_per_traj", type=int, default=2)
    parser.add_argument("--latency_iters", type=int, default=20)
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--skip_latency", action="store_true")
    parser.add_argument("--skip_onnx", action="store_true")
    parser.add_argument("--skip_trt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Load policy ----
    logger.info("=" * 80)
    logger.info("BACKBONE PIPELINE BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Samples: {args.num_trajs} trajs x {args.steps_per_traj} steps")
    logger.info("")

    logger.info("[Setup] Loading policy (flash_attention_2, BF16)...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )

    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    # ---- 2. Collect samples + reference (Stage A) ----
    logger.info("")
    logger.info("[Stage A] PyTorch BF16 flash_attention_2 (reference)")
    samples = collect_samples(policy, dataset, args.num_trajs, args.steps_per_traj)
    ref_outputs = [s["ref_features"] for s in samples]

    # ---- 3. Run stages ----
    stages = []

    # Stage A: reference (already collected)
    stages.append({
        "name": "A: PyTorch BF16 flash (ref)",
        "accuracy": None,  # Baseline
        "outputs": ref_outputs,
    })

    # Stage B: Eager
    logger.info("")
    eager_outputs = run_stage_eager(policy, samples)
    eager_metrics = [compute_metrics(r, t) for r, t in zip(ref_outputs, eager_outputs)]
    stages.append({
        "name": "B: PyTorch FP32 eager",
        "accuracy": aggregate_metrics(eager_metrics),
        "outputs": eager_outputs,
    })

    # Stage C: ONNX
    if not args.skip_onnx:
        logger.info("")
        onnx_outputs = run_stage_onnx(args.onnx_path, samples)
        if onnx_outputs is not None:
            onnx_metrics = [compute_metrics(r, t) for r, t in zip(ref_outputs, onnx_outputs)]
            stages.append({
                "name": "C: ONNX Runtime",
                "accuracy": aggregate_metrics(onnx_metrics),
                "outputs": onnx_outputs,
            })

    # Stage D: TRT FP16
    if not args.skip_trt:
        logger.info("")
        fp16_outputs = run_stage_trt(args.trt_fp16_path, samples, "Stage D: TRT FP16")
        if fp16_outputs is not None:
            fp16_metrics = [compute_metrics(r, t) for r, t in zip(ref_outputs, fp16_outputs)]
            stages.append({
                "name": "D: TRT FP16 engine",
                "accuracy": aggregate_metrics(fp16_metrics),
                "outputs": fp16_outputs,
            })

    # Stage E: TRT INT8
    if not args.skip_trt:
        logger.info("")
        int8_outputs = run_stage_trt(args.trt_int8_path, samples, "Stage E: TRT INT8")
        if int8_outputs is not None:
            int8_metrics = [compute_metrics(r, t) for r, t in zip(ref_outputs, int8_outputs)]
            stages.append({
                "name": "E: TRT INT8 engine",
                "accuracy": aggregate_metrics(int8_metrics),
                "outputs": int8_outputs,
            })

    # ---- 4. Latency benchmarks ----
    latencies = {}
    if not args.skip_latency and len(samples) > 0:
        logger.info("")
        logger.info("[Latency] Benchmarking with first sample...")
        s0 = samples[0]

        # Stage A latency: flash backbone
        bb_in_gpu = BatchFeature(data={
            "pixel_values": [t.cuda() for t in s0["pixel_values"]] if isinstance(s0["pixel_values"], list) else s0["pixel_values"].cuda(),
            "input_ids": s0["input_ids"].cuda(),
            "attention_mask": s0["attention_mask"].cuda(),
        })
        bb_in_bf16 = _rec_to_dtype(bb_in_gpu, torch.bfloat16)

        def run_flash():
            with torch.inference_mode():
                policy.model.backbone(bb_in_bf16)

        latencies["A"] = benchmark_latency(run_flash, args.latency_warmup, args.latency_iters)
        logger.info(f"  A (flash): {latencies['A']['median_ms']:.1f}ms median")

        # Stage B latency: eager
        backbone_eager = deepcopy(policy.model.backbone)
        swap_flash_to_eager(backbone_eager)
        backbone_eager = backbone_eager.float().cuda().eval()
        bb_in_fp32 = _rec_to_dtype(bb_in_gpu, torch.float32)

        def run_eager():
            with torch.inference_mode():
                backbone_eager(bb_in_fp32)

        latencies["B"] = benchmark_latency(run_eager, args.latency_warmup, args.latency_iters)
        logger.info(f"  B (eager): {latencies['B']['median_ms']:.1f}ms median")
        del backbone_eager
        gc.collect()
        torch.cuda.empty_cache()

        # Stage C latency: ONNX
        if not args.skip_onnx and ORT_AVAILABLE and os.path.exists(args.onnx_path):
            sess = ort.InferenceSession(
                args.onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            pv_input = [inp for inp in sess.get_inputs() if inp.name == "pixel_values"][0]
            onnx_dtype_map = {"tensor(float16)": torch.float16, "tensor(float)": torch.float32}
            onnx_pv_dtype = onnx_dtype_map.get(pv_input.type, torch.float32)
            onnx_pv_rank = len(pv_input.shape)
            pv = s0["pixel_values"]
            if isinstance(pv, list):
                if onnx_pv_rank == 5:
                    pv_np = torch.stack(pv, dim=0).to(onnx_pv_dtype).numpy()
                else:
                    pv_np = torch.cat(pv, dim=0).to(onnx_pv_dtype).numpy()
            else:
                pv_np = pv.to(onnx_pv_dtype).numpy()
            ort_in = {
                "pixel_values": pv_np,
                "input_ids": s0["input_ids"].numpy(),
                "attention_mask": s0["attention_mask"].numpy(),
            }

            def run_ort():
                sess.run(None, ort_in)

            latencies["C"] = benchmark_latency(run_ort, args.latency_warmup, args.latency_iters)
            logger.info(f"  C (ONNX):  {latencies['C']['median_ms']:.1f}ms median")
            del sess
            gc.collect()

        # Stage D/E latency: TRT
        if not args.skip_trt and TRT_AVAILABLE:
            import importlib.util
            script_path = os.path.join(os.path.dirname(__file__), "standalone_inference_script.py")
            spec = importlib.util.spec_from_file_location("sis", script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            TRTBBWrapper = mod.TensorRTBackboneWrapper

            for label, path in [("D", args.trt_fp16_path), ("E", args.trt_int8_path)]:
                if not os.path.exists(path):
                    continue
                trt_w = TRTBBWrapper(path, device=0)
                ids = s0["input_ids"].cuda()
                mask = s0["attention_mask"].cuda()
                pv = s0["pixel_values"]
                pv_gpu = [t.cuda() for t in pv] if isinstance(pv, list) else pv.cuda()

                def run_trt(w=trt_w, i=ids, m=mask, p=pv_gpu):
                    w(i, m, p)
                    torch.cuda.synchronize()

                latencies[label] = benchmark_latency(run_trt, args.latency_warmup, args.latency_iters)
                logger.info(f"  {label} (TRT):  {latencies[label]['median_ms']:.1f}ms median")
                del trt_w
                gc.collect()
                torch.cuda.empty_cache()

    # ---- 5. Summary table ----
    print("")
    print("=" * 100)
    n = samples[0]["ref_features"].shape if samples else "?"
    print(f"BACKBONE PIPELINE BENCHMARK ({len(samples)} samples, "
          f"{args.num_trajs} trajs x {args.steps_per_traj} steps, shape={n})")
    print("=" * 100)

    hdr = f"{'Stage':<30} {'MSE':>12} {'MAE':>12} {'Cos Sim':>10} {'Max Abs':>10}"
    if latencies:
        hdr += f" {'Median ms':>10} {'P90 ms':>8}"
    print(hdr)
    print("-" * len(hdr))

    stage_letter = ["A", "B", "C", "D", "E"]
    for i, st in enumerate(stages):
        letter = stage_letter[i] if i < len(stage_letter) else "?"
        acc = st["accuracy"]
        if acc is None:
            row = f"{st['name']:<30} {'baseline':>12} {'baseline':>12} {'baseline':>10} {'baseline':>10}"
        else:
            row = (f"{st['name']:<30} {acc['mse']:>12.6f} {acc['mae']:>12.6f} "
                   f"{acc['cos_sim']:>10.6f} {acc['max_abs']:>10.4f}")
        if latencies:
            lat = latencies.get(letter)
            if lat:
                row += f" {lat['median_ms']:>10.1f} {lat['p90_ms']:>8.1f}"
            else:
                row += f" {'—':>10} {'—':>8}"
        print(row)

    # ---- 6. Incremental degradation ----
    print("")
    print("Incremental degradation (each stage vs previous):")
    labels = [
        ("B vs A", "flash->eager", "attention implementation"),
        ("C vs B", "eager->ONNX", "ONNX tracing"),
        ("D vs C", "ONNX->TRT FP16", "TRT compilation + FP16"),
        ("E vs D", "FP16->INT8", "INT8 quantization"),
    ]
    for idx, (lbl, short, desc) in enumerate(labels):
        prev_idx = idx
        curr_idx = idx + 1
        if curr_idx >= len(stages):
            break
        prev_out = stages[prev_idx]["outputs"]
        curr_out = stages[curr_idx]["outputs"]
        inc_metrics = [compute_metrics(p, c) for p, c in zip(prev_out, curr_out)]
        inc_agg = aggregate_metrics(inc_metrics)
        marker = "  <-- BOTTLENECK" if inc_agg["mse"] > 0.01 else ""
        print(f"  {lbl} ({short}): MSE={inc_agg['mse']:.6f}, "
              f"cos_sim={inc_agg['cos_sim']:.6f}  ({desc}){marker}")

    print("=" * 100)


if __name__ == "__main__":
    main()
