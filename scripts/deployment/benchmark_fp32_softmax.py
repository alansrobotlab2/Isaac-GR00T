#!/usr/bin/env python3
"""
Quick benchmark for FP32-softmax-patched TRT backbone engines.

Compares patched FP16 TRT engine against:
  A: PyTorch BF16 flash (reference)
  B: Original TRT FP16 (broken, cos_sim~0.354)
  C: Patched TRT FP16 (target cos_sim > 0.99)

Usage (inside Docker):
    cd /workspace/gr00t/Isaac-GR00T

    python scripts/deployment/benchmark_fp32_softmax.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --original_trt groot_n1d6_onnx_sdpa_fp32/backbone_fp16_agx.trt \
        --patched_trt groot_n1d6_onnx_sdpa_fp32/backbone_fp16_fp32sm.trt
"""

import argparse
import gc
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cosine_sim(a, b):
    a_flat = a.reshape(a.shape[0], -1).float()
    b_flat = b.reshape(b.shape[0], -1).float()
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def compute_metrics(ref, test):
    ref_f = ref.float()
    test_f = test.float()
    diff = ref_f - test_f
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cos_sim": cosine_sim(ref, test),
    }


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def benchmark_latency(run_fn, warmup=5, num_iters=20):
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


def collect_samples(policy, dataset, num_trajs=3, steps_per_traj=2):
    """Collect backbone input/output samples."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType, VLAStepData

    logger.info(f"Collecting {num_trajs * steps_per_traj} samples...")
    samples = []
    max_traj = min(num_trajs, len(dataset))

    for traj_idx in range(max_traj):
        traj = dataset[traj_idx]
        traj_len = len(traj)
        step_indices = list(range(0, min(traj_len, steps_per_traj * 16), 16))[:steps_per_traj]

        for step_idx in step_indices:
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

            unbatched_obs = []
            batch_size = new_obs["video"][list(new_obs["video"].keys())[0]].shape[0]
            for i in range(batch_size):
                unbatched_obs.append({
                    "video": {k: v[i] for k, v in new_obs["video"].items()},
                    "state": {k: v[i] for k, v in new_obs["state"].items()},
                    "language": {k: v[i] for k, v in new_obs["language"].items()},
                })

            processed = []
            for o in unbatched_obs:
                vla = VLAStepData(
                    images=o["video"], states=o["state"], actions={},
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

            pv = bb_inputs["pixel_values"]
            samples.append({
                "input_ids": bb_inputs["input_ids"].detach().cpu().clone(),
                "attention_mask": bb_inputs["attention_mask"].detach().cpu().clone(),
                "pixel_values": [t.detach().cpu().clone() for t in pv] if isinstance(pv, list) else pv.detach().cpu().clone(),
                "ref_features": ref_out.backbone_features.detach().cpu().float(),
            })

    logger.info(f"Collected {len(samples)} samples")
    return samples


def run_trt_engine(engine_path, samples, label="TRT"):
    """Run backbone through TRT engine."""
    if not os.path.exists(engine_path):
        logger.warning(f"{label}: Engine not found: {engine_path}")
        return None, None

    logger.info(f"[{label}] Loading: {engine_path}")

    import importlib.util
    script_path = os.path.join(os.path.dirname(__file__), "standalone_inference_script.py")
    spec = importlib.util.spec_from_file_location("sis", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TRTWrapper = mod.TensorRTBackboneWrapper

    trt_w = TRTWrapper(engine_path, device=0)
    pv_dtype = trt_w.input_dtypes.get("pixel_values", torch.float32)
    logger.info(f"  Engine expects pixel_values dtype: {pv_dtype}")

    # Accuracy
    outputs = []
    for s in samples:
        ids = s["input_ids"].cuda()
        mask = s["attention_mask"].cuda()
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to(dtype=pv_dtype).cuda() for t in pv]
        else:
            pv_gpu = pv.to(dtype=pv_dtype).cuda()

        hs, _, _ = trt_w(ids, mask, pv_gpu)
        torch.cuda.synchronize()
        outputs.append(hs.detach().cpu().float())

    # Latency
    s0 = samples[0]
    ids = s0["input_ids"].cuda()
    mask = s0["attention_mask"].cuda()
    pv = s0["pixel_values"]
    pv_gpu = [t.to(dtype=pv_dtype).cuda() for t in pv] if isinstance(pv, list) else pv.to(dtype=pv_dtype).cuda()

    def run_fn():
        trt_w(ids, mask, pv_gpu)
        torch.cuda.synchronize()

    latency = benchmark_latency(run_fn)
    logger.info(f"  {label}: {latency['median_ms']:.1f}ms median")

    del trt_w
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", default="new_embodiment")
    parser.add_argument("--original_trt", default="groot_n1d6_onnx_sdpa_fp32/backbone_fp16_agx.trt")
    parser.add_argument("--patched_trt", default="groot_n1d6_onnx_sdpa_fp32/backbone_fp16_fp32sm.trt")
    parser.add_argument("--patched_aggr_trt", default="groot_n1d6_onnx_sdpa_fp32/backbone_fp16_fp32sm_aggr.trt")
    parser.add_argument("--video_backend", default="torchcodec")
    parser.add_argument("--num_trajs", type=int, default=3)
    parser.add_argument("--steps_per_traj", type=int, default=2)
    args = parser.parse_args()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logger.info("=" * 80)
    logger.info("FP32 SOFTMAX SURGERY BENCHMARK")
    logger.info("=" * 80)

    logger.info("Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
    )

    # Collect reference samples (flash attention)
    samples = collect_samples(policy, dataset, args.num_trajs, args.steps_per_traj)
    ref_outputs = [s["ref_features"] for s in samples]

    # Flash latency
    s0 = samples[0]
    bb_in = BatchFeature(data={
        "pixel_values": [t.cuda() for t in s0["pixel_values"]] if isinstance(s0["pixel_values"], list) else s0["pixel_values"].cuda(),
        "input_ids": s0["input_ids"].cuda(),
        "attention_mask": s0["attention_mask"].cuda(),
    })
    bb_in = _rec_to_dtype(bb_in, torch.bfloat16)

    def run_flash():
        with torch.inference_mode():
            policy.model.backbone(bb_in)

    flash_lat = benchmark_latency(run_flash)
    logger.info(f"Flash backbone: {flash_lat['median_ms']:.1f}ms median")

    # Test engines
    results = []
    results.append(("A: PyTorch BF16 flash", None, flash_lat, "baseline"))

    for label, path in [
        ("B: TRT FP16 (original)", args.original_trt),
        ("C: TRT FP16 (softmax-only patch)", args.patched_trt),
        ("D: TRT FP16 (aggressive patch)", args.patched_aggr_trt),
    ]:
        outputs, latency = run_trt_engine(path, samples, label)
        if outputs is not None:
            metrics_list = [compute_metrics(r, t) for r, t in zip(ref_outputs, outputs)]
            avg = {}
            for k in metrics_list[0]:
                vals = [m[k] for m in metrics_list]
                avg[k] = max(vals) if k == "max_abs" else float(np.mean(vals))
            results.append((label, avg, latency, None))

    # Print results
    print()
    print("=" * 110)
    print("FP32 SOFTMAX SURGERY RESULTS")
    print("=" * 110)
    print(f"{'Config':<40} {'MSE':>10} {'MAE':>10} {'Cos Sim':>10} {'Max Abs':>10} {'Median ms':>10} {'P90 ms':>8}")
    print("-" * 110)

    for name, acc, lat, note in results:
        if note == "baseline":
            print(f"{name:<40} {'baseline':>10} {'baseline':>10} {'baseline':>10} {'baseline':>10} "
                  f"{lat['median_ms']:>10.1f} {lat['p90_ms']:>8.1f}")
        else:
            cos_marker = " <<<" if acc["cos_sim"] > 0.99 else " !!!" if acc["cos_sim"] < 0.5 else ""
            print(f"{name:<40} {acc['mse']:>10.4f} {acc['mae']:>10.4f} {acc['cos_sim']:>10.6f} "
                  f"{acc['max_abs']:>10.2f} {lat['median_ms']:>10.1f} {lat['p90_ms']:>8.1f}{cos_marker}")

    print("=" * 110)
    print()
    print("Legend: <<< = cos_sim > 0.99 (SUCCESS), !!! = cos_sim < 0.5 (quality destroyed)")


if __name__ == "__main__":
    main()
