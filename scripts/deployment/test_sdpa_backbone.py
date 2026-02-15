#!/usr/bin/env python3
"""
Test SDPA attention for backbone: accuracy vs flash_attention_2 and latency with torch.compile.

This script answers two key questions:
1. Does swapping flash_attention_2 → sdpa close the MSE 3.76 gap we see with eager?
2. Does torch.compile on the SDPA backbone give a meaningful speedup?

Usage (inside Docker):
    python scripts/deployment/test_sdpa_backbone.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --video_backend torchcodec
"""

import argparse
import logging
import time
from copy import deepcopy

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def swap_attn_implementation(model, target="sdpa"):
    """Swap all attention implementations to target (sdpa or eager)."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation != target:
                old = module.config._attn_implementation
                module.config._attn_implementation = target
                count += 1
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation != target:
                module._attn_implementation = target
    logger.info(f"Swapped {count} attention implementations → {target}")


def compare_outputs(ref, test, label):
    """Compare two tensors and print accuracy metrics."""
    diff = ref - test
    mse = (diff**2).mean().item()
    mae = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1), test.reshape(1, -1)
    ).item()
    logger.info(f"  {label}:")
    logger.info(f"    MSE:        {mse:.6f}")
    logger.info(f"    MAE:        {mae:.6f}")
    logger.info(f"    Max abs:    {max_abs:.4f}")
    logger.info(f"    Cosine sim: {cos_sim:.6f}")
    logger.info(f"    Ref range:  [{ref.min():.2f}, {ref.max():.2f}]")
    logger.info(f"    Test range: [{test.min():.2f}, {test.max():.2f}]")
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "cos_sim": cos_sim}


def benchmark_backbone(backbone, bb_inputs, label, num_iters=20, warmup=5):
    """Benchmark backbone forward pass latency."""
    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = backbone(bb_inputs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = backbone(bb_inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    logger.info(f"  {label}: median={np.median(times):.1f}ms, "
                f"mean={np.mean(times):.1f}±{np.std(times):.1f}ms, "
                f"min={np.min(times):.1f}ms, max={np.max(times):.1f}ms")
    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description="Test SDPA backbone accuracy and latency")
    parser.add_argument("--model_path", type=str, default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", type=str, default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--num_iters", type=int, default=20, help="Latency benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--skip_compile", action="store_true", help="Skip torch.compile test")
    parser.add_argument("--skip_eager", action="store_true", help="Skip eager attention test")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SDPA BACKBONE TEST: Accuracy + Latency")
    logger.info("=" * 80)

    # ---- Load policy ----
    logger.info("\n[1] Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )

    # ---- Prepare input ----
    logger.info("\n[2] Preparing input data...")
    modality_config = policy.get_modality_config()
    modality_configs = deepcopy(modality_config)
    modality_configs.pop("action", None)

    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    traj = dataset[0]
    data_point = extract_step_data(traj, 0, modality_configs, policy.embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for key in modality_configs["language"].modality_keys:
        obs[key] = data_point.text

    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]

    unbatched_obs = []
    batch_size = new_obs["video"][list(new_obs["video"].keys())[0]].shape[0]
    for i in range(batch_size):
        unbatched_value = {
            "video": {k: v[i] for k, v in new_obs["video"].items()},
            "state": {k: v[i] for k, v in new_obs["state"].items()},
            "language": {k: v[i] for k, v in new_obs["language"].items()},
        }
        unbatched_obs.append(unbatched_value)

    processed_inputs = []
    for o in unbatched_obs:
        vla_step_data = VLAStepData(
            images=o["video"],
            states=o["state"],
            actions={},
            text=o["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = collated_inputs["inputs"]
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    # ================================================================
    # A. Reference: PyTorch BF16 backbone (flash_attention_2)
    # ================================================================
    logger.info("\n" + "=" * 80)
    logger.info("A. REFERENCE: PyTorch BF16 backbone (flash_attention_2)")
    logger.info("=" * 80)

    with torch.inference_mode():
        bb_inputs, _ = policy.model.prepare_input(collated_inputs)
        ref_output = policy.model.backbone(bb_inputs)

    ref_features = ref_output.backbone_features.detach().cpu().float()
    logger.info(f"  Output shape: {ref_features.shape}")
    logger.info(f"  Output range: [{ref_features.min():.2f}, {ref_features.max():.2f}]")

    # Benchmark flash attention
    logger.info("\n  Latency (flash_attention_2):")
    flash_ms = benchmark_backbone(
        policy.model.backbone, bb_inputs, "flash_attn_2",
        num_iters=args.num_iters, warmup=args.warmup
    )

    # ================================================================
    # B. Test: SDPA backbone (BF16, same weights)
    # ================================================================
    logger.info("\n" + "=" * 80)
    logger.info("B. TEST: SDPA backbone (BF16, same weights)")
    logger.info("=" * 80)

    # Swap attention to SDPA on the SAME backbone (no copy needed for accuracy test)
    swap_attn_implementation(policy.model.backbone, target="sdpa")

    with torch.inference_mode():
        bb_inputs_sdpa, _ = policy.model.prepare_input(collated_inputs)
        sdpa_output = policy.model.backbone(bb_inputs_sdpa)

    sdpa_features = sdpa_output.backbone_features.detach().cpu().float()

    metrics_sdpa = compare_outputs(ref_features, sdpa_features, "flash_attn_2 vs SDPA (BF16)")

    # Benchmark SDPA
    logger.info("\n  Latency (SDPA):")
    sdpa_ms = benchmark_backbone(
        policy.model.backbone, bb_inputs_sdpa, "sdpa",
        num_iters=args.num_iters, warmup=args.warmup
    )

    # ================================================================
    # C. Test: Eager backbone (BF16, same weights) - for comparison
    # ================================================================
    if not args.skip_eager:
        logger.info("\n" + "=" * 80)
        logger.info("C. TEST: Eager backbone (BF16, same weights)")
        logger.info("=" * 80)

        swap_attn_implementation(policy.model.backbone, target="eager")

        with torch.inference_mode():
            bb_inputs_eager, _ = policy.model.prepare_input(collated_inputs)
            eager_output = policy.model.backbone(bb_inputs_eager)

        eager_features = eager_output.backbone_features.detach().cpu().float()

        metrics_eager = compare_outputs(ref_features, eager_features, "flash_attn_2 vs Eager (BF16)")

        # Benchmark eager
        logger.info("\n  Latency (Eager BF16):")
        eager_ms = benchmark_backbone(
            policy.model.backbone, bb_inputs_eager, "eager",
            num_iters=args.num_iters, warmup=args.warmup
        )

        # Also compare SDPA vs eager directly
        logger.info("\n  Direct comparison:")
        compare_outputs(sdpa_features, eager_features, "SDPA vs Eager")
    else:
        eager_ms = None

    # ================================================================
    # D. Test: torch.compile + SDPA backbone
    # ================================================================
    if not args.skip_compile:
        logger.info("\n" + "=" * 80)
        logger.info("D. TEST: torch.compile + SDPA backbone")
        logger.info("=" * 80)

        # Switch back to SDPA
        swap_attn_implementation(policy.model.backbone, target="sdpa")

        logger.info("  Compiling backbone with torch.compile(mode='max-autotune')...")
        original_forward = policy.model.backbone.forward
        compiled_forward = torch.compile(original_forward, mode="max-autotune")
        policy.model.backbone.forward = compiled_forward

        # Extra warmup for compilation
        logger.info("  Warming up compiled backbone (this may take 30-60 seconds)...")
        for i in range(3):
            with torch.inference_mode():
                bb_inputs_compile, _ = policy.model.prepare_input(collated_inputs)
                compile_output = policy.model.backbone(bb_inputs_compile)
            torch.cuda.synchronize()
            logger.info(f"    Warmup {i+1}/3 done")

        # Accuracy check
        compile_features = compile_output.backbone_features.detach().cpu().float()
        metrics_compile = compare_outputs(ref_features, compile_features, "flash_attn_2 vs compile+SDPA")

        # Benchmark
        logger.info("\n  Latency (torch.compile + SDPA):")
        compile_ms = benchmark_backbone(
            policy.model.backbone, bb_inputs_compile, "compile+sdpa",
            num_iters=args.num_iters, warmup=args.warmup
        )

        # Restore original forward
        policy.model.backbone.forward = original_forward
    else:
        compile_ms = None

    # ================================================================
    # SUMMARY
    # ================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Mode':<25} {'MSE':>12} {'Cos Sim':>12} {'Latency':>10} {'vs flash':>10}")
    logger.info("-" * 69)
    logger.info(f"{'flash_attn_2 (ref)':<25} {'baseline':>12} {'baseline':>12} {flash_ms:>8.1f}ms {'1.00x':>10}")
    logger.info(f"{'SDPA (BF16)':<25} {metrics_sdpa['mse']:>12.6f} {metrics_sdpa['cos_sim']:>12.6f} {sdpa_ms:>8.1f}ms {flash_ms/sdpa_ms:>9.2f}x")

    if not args.skip_eager:
        logger.info(f"{'Eager (BF16)':<25} {metrics_eager['mse']:>12.6f} {metrics_eager['cos_sim']:>12.6f} {eager_ms:>8.1f}ms {flash_ms/eager_ms:>9.2f}x")

    if not args.skip_compile:
        logger.info(f"{'compile+SDPA':<25} {metrics_compile['mse']:>12.6f} {metrics_compile['cos_sim']:>12.6f} {compile_ms:>8.1f}ms {flash_ms/compile_ms:>9.2f}x")

    logger.info("\n" + "=" * 80)

    # Key takeaway
    if metrics_sdpa["mse"] < 0.01:
        logger.info("SDPA accuracy is EXCELLENT - viable for ONNX export path")
    elif metrics_sdpa["mse"] < 1.0:
        logger.info("SDPA accuracy is GOOD - may work for ONNX export")
    else:
        logger.info("SDPA accuracy is POOR - same issue as eager")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
