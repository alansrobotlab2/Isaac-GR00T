#!/usr/bin/env python3
"""
Test E2E action accuracy with SDPA/compile backbone vs flash_attention_2 baseline.

Answers the key question: does the backbone MSE 3.8 (flash vs SDPA) actually
matter for final action predictions?

Runs both policies on the same trajectories, comparing:
  1. ref (flash) vs ground truth
  2. test (SDPA) vs ground truth
  3. ref vs test (direct action divergence)

Also optionally tests with TRT DiT to measure the combined effect.

Usage (inside Docker):
    python scripts/deployment/test_sdpa_e2e.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --video_backend torchcodec \
        --traj_ids 0 1 2 3 4 \
        --trt_dit_path ./groot_n1d6_onnx/dit_fp16.trt
"""

import argparse
import gc
import logging
import time
from copy import deepcopy

import numpy as np
import torch

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
    """Swap all attention implementations to target."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation != target:
                module.config._attn_implementation = target
                count += 1
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation != target:
                module._attn_implementation = target
    logger.info(f"Swapped {count} attention implementations -> {target}")


def parse_observation_gr00t(obs, modality_configs):
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
    return new_obs


def prepare_observation(policy, dataset, traj_idx, step_idx, modality_configs):
    traj = dataset[traj_idx]
    data_point = extract_step_data(traj, step_idx, modality_configs, policy.embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for key in modality_configs["language"].modality_keys:
        obs[key] = data_point.text
    return parse_observation_gr00t(obs, modality_configs), traj


def evaluate_e2e(
    ref_policy, test_policy, dataset, traj_ids, modality_configs, action_keys,
    action_horizon=16, max_steps=200, label="test",
):
    """Run E2E evaluation comparing ref and test policies against ground truth."""
    modality_configs_no_action = deepcopy(modality_configs)
    modality_configs_no_action.pop("action", None)

    all_ref_mse = []
    all_test_mse = []
    all_ref_test_mse = []
    all_ref_mae = []
    all_test_mae = []

    for traj_id in traj_ids:
        traj = dataset[traj_id]
        traj_len = len(traj)
        actual_steps = min(max_steps, traj_len)

        ref_preds = []
        test_preds = []

        pred_horizon = action_horizon
        step_counts = list(range(0, actual_steps, pred_horizon))

        for step_count in step_counts:
            obs, _ = prepare_observation(
                ref_policy, dataset, traj_id, step_count, modality_configs_no_action
            )

            with torch.inference_mode():
                ref_action, _ = ref_policy.get_action(obs)
                test_action, _ = test_policy.get_action(obs)

            first_key = action_keys[0]
            ref_arr = np.array(ref_action[first_key])
            if ref_arr.ndim == 3:
                chunk_len = ref_arr.shape[1]
                pred_horizon = chunk_len
                for j in range(chunk_len):
                    ref_concat = np.concatenate(
                        [np.array(ref_action[k])[0, j] for k in action_keys], axis=0
                    )
                    test_concat = np.concatenate(
                        [np.array(test_action[k])[0, j] for k in action_keys], axis=0
                    )
                    ref_preds.append(ref_concat)
                    test_preds.append(test_concat)
            else:
                ref_concat = np.concatenate(
                    [np.array(ref_action[k]).flatten() for k in action_keys], axis=0
                )
                test_concat = np.concatenate(
                    [np.array(test_action[k]).flatten() for k in action_keys], axis=0
                )
                ref_preds.append(ref_concat)
                test_preds.append(test_concat)

        ref_preds = np.array(ref_preds)[:actual_steps]
        test_preds = np.array(test_preds)[:actual_steps]

        gt_actions = np.concatenate(
            [np.vstack(traj[f"action.{k}"].values) for k in action_keys], axis=-1
        )[:actual_steps]

        min_len = min(len(gt_actions), len(ref_preds), len(test_preds))
        gt_actions = gt_actions[:min_len]
        ref_preds = ref_preds[:min_len]
        test_preds = test_preds[:min_len]

        ref_mse = np.mean((gt_actions - ref_preds) ** 2)
        test_mse = np.mean((gt_actions - test_preds) ** 2)
        ref_test_mse = np.mean((ref_preds - test_preds) ** 2)
        ref_mae = np.mean(np.abs(gt_actions - ref_preds))
        test_mae = np.mean(np.abs(gt_actions - test_preds))

        all_ref_mse.append(ref_mse)
        all_test_mse.append(test_mse)
        all_ref_test_mse.append(ref_test_mse)
        all_ref_mae.append(ref_mae)
        all_test_mae.append(test_mae)

        logger.info(f"  Traj {traj_id}: ref_mse={ref_mse:.6f}, {label}_mse={test_mse:.6f}, "
                     f"ref_vs_{label}={ref_test_mse:.6f}")

    return {
        "ref_vs_gt_mse": np.mean(all_ref_mse),
        "test_vs_gt_mse": np.mean(all_test_mse),
        "ref_vs_test_mse": np.mean(all_ref_test_mse),
        "ref_vs_gt_mae": np.mean(all_ref_mae),
        "test_vs_gt_mae": np.mean(all_test_mae),
    }


def benchmark_latency(policy, dataset, modality_configs, traj_id=0, num_iters=20, warmup=5):
    """Quick latency benchmark."""
    modality_configs_no_action = deepcopy(modality_configs)
    modality_configs_no_action.pop("action", None)

    obs, _ = prepare_observation(policy, dataset, traj_id, 0, modality_configs_no_action)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = policy.get_action(obs)
    torch.cuda.synchronize()
    gc.collect()

    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = policy.get_action(obs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    logger.info(f"  Latency: median={np.median(times):.1f}ms, "
                f"mean={np.mean(times):.1f}+-{np.std(times):.1f}ms "
                f"({1000/np.median(times):.1f} Hz)")
    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description="Test SDPA backbone E2E action accuracy")
    parser.add_argument("--model_path", type=str, default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", type=str, default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--trt_dit_path", type=str, default=None,
                        help="TRT DiT engine. If set, also tests SDPA backbone + TRT DiT combo.")
    parser.add_argument("--num_latency_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_latency", action="store_true")
    parser.add_argument("--use_compile", action="store_true",
                        help="Apply torch.compile to SDPA backbone")
    args = parser.parse_args()

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("SDPA BACKBONE E2E ACCURACY TEST")
    logger.info("=" * 80)

    # ---- Load reference policy (flash_attention_2) ----
    logger.info("\n[1] Loading reference policy (flash_attention_2)...")
    ref_policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )

    modality_config = ref_policy.get_modality_config()
    action_keys = modality_config["action"].modality_keys
    action_horizon = ref_policy.model.action_head.action_horizon

    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    logger.info(f"Action keys: {action_keys}")
    logger.info(f"Action horizon: {action_horizon}")
    logger.info(f"Trajectories: {args.traj_ids}")

    results = []

    # ---- A. Flash backbone + PyTorch DiT (baseline) ----
    logger.info("\n" + "=" * 80)
    logger.info("A. BASELINE: flash_attn_2 backbone + PyTorch DiT")
    logger.info("=" * 80)

    if not args.skip_latency:
        flash_ms = benchmark_latency(
            ref_policy, dataset, modality_config,
            traj_id=args.traj_ids[0], num_iters=args.num_latency_iters
        )
    else:
        flash_ms = None

    # ---- B. SDPA backbone + PyTorch DiT ----
    logger.info("\n" + "=" * 80)
    logger.info("B. TEST: SDPA backbone + PyTorch DiT")
    logger.info("=" * 80)

    # Create a separate policy for SDPA to keep ref untouched
    logger.info("  Loading SDPA policy...")
    sdpa_policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )
    swap_attn_implementation(sdpa_policy.model.backbone, target="sdpa")

    if args.use_compile:
        logger.info("  Compiling SDPA backbone with torch.compile(mode='max-autotune')...")
        sdpa_policy.model.backbone.forward = torch.compile(
            sdpa_policy.model.backbone.forward, mode="max-autotune"
        )
        # Warmup compilation
        modality_configs_no_action = deepcopy(modality_config)
        modality_configs_no_action.pop("action", None)
        obs_warmup, _ = prepare_observation(
            sdpa_policy, dataset, args.traj_ids[0], 0, modality_configs_no_action
        )
        logger.info("  Warming up torch.compile (may take 1-3 minutes)...")
        for i in range(3):
            with torch.inference_mode():
                _ = sdpa_policy.get_action(obs_warmup)
            torch.cuda.synchronize()
            logger.info(f"    Warmup {i+1}/3 done")

    logger.info("  Running E2E evaluation...")
    sdpa_e2e = evaluate_e2e(
        ref_policy, sdpa_policy, dataset, args.traj_ids,
        modality_config, action_keys, action_horizon, args.max_steps,
        label="sdpa",
    )

    if not args.skip_latency:
        sdpa_ms = benchmark_latency(
            sdpa_policy, dataset, modality_config,
            traj_id=args.traj_ids[0], num_iters=args.num_latency_iters
        )
    else:
        sdpa_ms = None

    results.append(("SDPA backbone + PT DiT", sdpa_e2e, sdpa_ms))

    # ---- C. SDPA backbone + TRT DiT (if provided) ----
    if args.trt_dit_path:
        logger.info("\n" + "=" * 80)
        logger.info("C. TEST: SDPA backbone + TRT FP16 DiT")
        logger.info("=" * 80)

        from standalone_inference_script import replace_dit_with_tensorrt

        sdpa_trt_policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device="cuda",
        )
        swap_attn_implementation(sdpa_trt_policy.model.backbone, target="sdpa")
        replace_dit_with_tensorrt(sdpa_trt_policy, args.trt_dit_path)

        if args.use_compile:
            logger.info("  Compiling SDPA backbone...")
            sdpa_trt_policy.model.backbone.forward = torch.compile(
                sdpa_trt_policy.model.backbone.forward, mode="max-autotune"
            )
            modality_configs_no_action = deepcopy(modality_config)
            modality_configs_no_action.pop("action", None)
            obs_warmup, _ = prepare_observation(
                sdpa_trt_policy, dataset, args.traj_ids[0], 0, modality_configs_no_action
            )
            logger.info("  Warming up torch.compile...")
            for i in range(3):
                with torch.inference_mode():
                    _ = sdpa_trt_policy.get_action(obs_warmup)
                torch.cuda.synchronize()
                logger.info(f"    Warmup {i+1}/3 done")

        logger.info("  Running E2E evaluation...")
        sdpa_trt_e2e = evaluate_e2e(
            ref_policy, sdpa_trt_policy, dataset, args.traj_ids,
            modality_config, action_keys, action_horizon, args.max_steps,
            label="sdpa_trt",
        )

        if not args.skip_latency:
            sdpa_trt_ms = benchmark_latency(
                sdpa_trt_policy, dataset, modality_config,
                traj_id=args.traj_ids[0], num_iters=args.num_latency_iters
            )
        else:
            sdpa_trt_ms = None

        results.append(("SDPA backbone + TRT DiT", sdpa_trt_e2e, sdpa_trt_ms))

        # ---- D. Flash backbone + TRT DiT (for comparison) ----
        logger.info("\n" + "=" * 80)
        logger.info("D. COMPARE: flash backbone + TRT FP16 DiT (current best)")
        logger.info("=" * 80)

        flash_trt_policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device="cuda",
        )
        replace_dit_with_tensorrt(flash_trt_policy, args.trt_dit_path)

        logger.info("  Running E2E evaluation...")
        flash_trt_e2e = evaluate_e2e(
            ref_policy, flash_trt_policy, dataset, args.traj_ids,
            modality_config, action_keys, action_horizon, args.max_steps,
            label="flash_trt",
        )

        if not args.skip_latency:
            flash_trt_ms = benchmark_latency(
                flash_trt_policy, dataset, modality_config,
                traj_id=args.traj_ids[0], num_iters=args.num_latency_iters
            )
        else:
            flash_trt_ms = None

        results.append(("Flash backbone + TRT DiT", flash_trt_e2e, flash_trt_ms))

    # ================================================================
    # SUMMARY
    # ================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    header = f"{'Mode':<30} {'GT MSE (ref)':>12} {'GT MSE (test)':>14} {'ref vs test':>12} {'GT MAE (test)':>14} {'Latency':>10} {'Hz':>6}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    ref_gt_mse = results[0][1]["ref_vs_gt_mse"] if results else 0

    logger.info(f"{'Flash backbone (baseline)':<30} {ref_gt_mse:>12.6f} {'(same)':>14} {'0.000000':>12} {'--':>14} "
                f"{f'{flash_ms:.0f}ms' if flash_ms else 'n/a':>10} "
                f"{f'{1000/flash_ms:.1f}' if flash_ms else 'n/a':>6}")

    for mode, e2e, ms in results:
        latency_str = f"{ms:.0f}ms" if ms else "n/a"
        hz_str = f"{1000/ms:.1f}" if ms else "n/a"
        logger.info(f"{mode:<30} {e2e['ref_vs_gt_mse']:>12.6f} {e2e['test_vs_gt_mse']:>14.6f} "
                     f"{e2e['ref_vs_test_mse']:>12.6f} {e2e['test_vs_gt_mae']:>14.6f} "
                     f"{latency_str:>10} {hz_str:>6}")

    logger.info("\n" + "=" * 80)

    # Verdict
    if results:
        sdpa_e2e = results[0][1]
        degradation = sdpa_e2e["test_vs_gt_mse"] / (ref_gt_mse + 1e-12)
        logger.info(f"SDPA E2E MSE degradation vs flash baseline: {degradation:.2f}x")
        if degradation < 1.1:
            logger.info("VERDICT: SDPA backbone has NEGLIGIBLE E2E impact - viable for TRT path!")
        elif degradation < 1.5:
            logger.info("VERDICT: SDPA backbone has SMALL E2E impact - likely acceptable")
        elif degradation < 2.0:
            logger.info("VERDICT: SDPA backbone has MODERATE E2E impact - test on robot before using")
        else:
            logger.info("VERDICT: SDPA backbone has SIGNIFICANT E2E impact - may not be acceptable")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
