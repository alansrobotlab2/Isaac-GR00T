#!/usr/bin/env python3
"""
Benchmark quantization accuracy and latency for GR00T model components.

Measures BOTH accuracy AND latency per-component (backbone, DiT, E2E),
unlike benchmark_inference.py (latency only) or standalone_inference_script.py
(E2E accuracy only).

Accuracy metrics:
  - MSE, MAE, cosine similarity, max absolute error per component
  - Per-denoising-step error growth for DiT
  - E2E action prediction MSE/MAE vs ground truth

Latency metrics:
  - Component-wise timing with torch.cuda.synchronize() barriers
  - Median, P90, mean ± std

Usage:
    # Compare PyTorch BF16 vs TRT FP16 DiT:
    python benchmark_quantization.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path /path/to/dataset \
        --trt_dit_path ./groot_n1d6_onnx/dit_model_fp16.trt

    # Compare with TRT backbone too:
    python benchmark_quantization.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path /path/to/dataset \
        --trt_dit_path ./groot_n1d6_onnx/dit_int8.trt \
        --trt_backbone_path ./groot_n1d6_onnx/backbone_int8.trt
"""

import argparse
import gc
import logging
import os
import time
from copy import deepcopy
from typing import Any

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors (flattened per batch)."""
    a_flat = a.reshape(a.shape[0], -1).float()
    b_flat = b.reshape(b.shape[0], -1).float()
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """Compute accuracy metrics between reference and test tensors."""
    ref_f = ref.float()
    test_f = test.float()
    diff = ref_f - test_f
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "cosine_sim": cosine_similarity(ref, test),
    }


def _rec_to_dtype(x, dtype):
    """Recursively convert all floating point tensors to the given dtype."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


# ---------------------------------------------------------------------------
# Data preparation (reuses patterns from standalone_inference_script.py)
# ---------------------------------------------------------------------------


def parse_observation_gr00t(obs: dict[str, Any], modality_configs: dict[str, Any]) -> dict[str, Any]:
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
    """Prepare a single observation for inference."""
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


# Reuse prepare_model_inputs from benchmark_inference.py
def prepare_model_inputs(policy, observation):
    """Prepare inputs for the model, mimicking what happens inside _get_action."""
    from gr00t.data.types import MessageType, VLAStepData

    unbatched_obs = []
    batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
    for i in range(batch_size):
        unbatched_value = {
            "video": {k: v[i] for k, v in observation["video"].items()},
            "state": {k: v[i] for k, v in observation["state"].items()},
            "language": {k: v[i] for k, v in observation["language"].items()},
        }
        unbatched_obs.append(unbatched_value)

    processed_inputs = []
    for obs in unbatched_obs:
        vla_step_data = VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = collated_inputs["inputs"]
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
    return collated_inputs


# ---------------------------------------------------------------------------
# DiT input capture (reuses pattern from export_onnx_n1d6.py)
# ---------------------------------------------------------------------------


class DiTInputCaptureAll:
    """Capture DiT inputs at every denoising step (not just first call)."""

    def __init__(self):
        self.captured_steps: list[dict] = []

    def hook_fn(self, module, args, kwargs):
        self.captured_steps.append({
            "hidden_states": kwargs["hidden_states"].detach().clone(),
            "encoder_hidden_states": kwargs["encoder_hidden_states"].detach().clone(),
            "timestep": kwargs["timestep"].detach().clone(),
            "image_mask": kwargs.get("image_mask"),
            "backbone_attention_mask": kwargs.get("backbone_attention_mask"),
        })
        # Clone optional masks
        step = self.captured_steps[-1]
        if step["image_mask"] is not None:
            step["image_mask"] = step["image_mask"].detach().clone()
        if step["backbone_attention_mask"] is not None:
            step["backbone_attention_mask"] = step["backbone_attention_mask"].detach().clone()

    def reset(self):
        self.captured_steps = []


# ---------------------------------------------------------------------------
# Backbone accuracy benchmark
# ---------------------------------------------------------------------------


def benchmark_backbone_accuracy(
    ref_policy: Gr00tPolicy,
    test_policy: Gr00tPolicy,
    dataset: LeRobotEpisodeLoader,
    traj_ids: list[int],
    num_steps_per_traj: int = 3,
) -> dict[str, float]:
    """
    Compare backbone outputs between two policies (e.g. PyTorch vs TRT backbone).
    Feeds identical inputs to both and computes feature-level metrics.
    """
    logger.info("Benchmarking backbone accuracy...")
    modality_configs = deepcopy(ref_policy.get_modality_config())
    modality_configs.pop("action", None)

    all_metrics = []

    for traj_id in traj_ids:
        traj = dataset[traj_id]
        traj_len = len(traj)
        step_indices = list(range(0, min(traj_len, num_steps_per_traj * 16), 16))[:num_steps_per_traj]

        for step_idx in step_indices:
            obs, _ = prepare_observation(ref_policy, dataset, traj_id, step_idx, modality_configs)
            collated_inputs = prepare_model_inputs(ref_policy, obs)

            with torch.inference_mode():
                # Reference backbone
                bb_inputs_ref, _ = ref_policy.model.prepare_input(collated_inputs)
                bb_out_ref = ref_policy.model.backbone(bb_inputs_ref)

                # Test backbone (may be TRT-replaced)
                bb_inputs_test, _ = test_policy.model.prepare_input(collated_inputs)
                bb_out_test = test_policy.model.backbone(bb_inputs_test)

            metrics = compute_metrics(
                bb_out_ref.backbone_features, bb_out_test.backbone_features
            )
            all_metrics.append(metrics)

    # Average across all samples
    avg = {}
    for key in all_metrics[0]:
        if key == "max_abs_error":
            avg[key] = max(m[key] for m in all_metrics)
        else:
            avg[key] = np.mean([m[key] for m in all_metrics])

    logger.info(f"  Backbone accuracy ({len(all_metrics)} samples):")
    for k, v in avg.items():
        logger.info(f"    {k}: {v:.6f}")

    return avg


# ---------------------------------------------------------------------------
# DiT accuracy benchmark (per-denoising-step)
# ---------------------------------------------------------------------------


def benchmark_dit_accuracy(
    ref_policy: Gr00tPolicy,
    test_policy: Gr00tPolicy,
    dataset: LeRobotEpisodeLoader,
    traj_ids: list[int],
    num_steps_per_traj: int = 3,
) -> dict[str, Any]:
    """
    Compare DiT outputs between two policies at each denoising step.
    Captures DiT inputs from the reference policy and feeds them to both
    the reference DiT and the test DiT (which may be TRT-replaced).

    Returns per-step metrics + overall metrics.
    """
    logger.info("Benchmarking DiT accuracy (per-denoising-step)...")
    modality_configs = deepcopy(ref_policy.get_modality_config())
    modality_configs.pop("action", None)

    num_denoise_steps = ref_policy.model.action_head.num_inference_timesteps
    per_step_metrics: list[list[dict]] = [[] for _ in range(num_denoise_steps)]

    # Get reference to the DiT models
    ref_dit = ref_policy.model.action_head.model
    test_dit_forward = test_policy.model.action_head.model.forward

    for traj_id in traj_ids:
        traj = dataset[traj_id]
        traj_len = len(traj)
        step_indices = list(range(0, min(traj_len, num_steps_per_traj * 16), 16))[:num_steps_per_traj]

        for step_idx in step_indices:
            obs, _ = prepare_observation(ref_policy, dataset, traj_id, step_idx, modality_configs)
            collated_inputs = prepare_model_inputs(ref_policy, obs)

            # Capture DiT inputs from reference policy at every denoising step
            capture = DiTInputCaptureAll()
            hook = ref_dit.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)

            with torch.inference_mode():
                bb_inputs, action_inputs = ref_policy.model.prepare_input(collated_inputs)
                bb_out = ref_policy.model.backbone(bb_inputs)
                _ = ref_policy.model.action_head.get_action(bb_out, action_inputs)

            hook.remove()

            # Now feed captured inputs to both ref DiT and test DiT
            for step_i, captured in enumerate(capture.captured_steps):
                with torch.inference_mode():
                    ref_out = ref_dit(
                        hidden_states=captured["hidden_states"],
                        encoder_hidden_states=captured["encoder_hidden_states"],
                        timestep=captured["timestep"],
                        image_mask=captured["image_mask"],
                        backbone_attention_mask=captured["backbone_attention_mask"],
                    )
                    test_out = test_dit_forward(
                        hidden_states=captured["hidden_states"],
                        encoder_hidden_states=captured["encoder_hidden_states"],
                        timestep=captured["timestep"],
                        image_mask=captured["image_mask"],
                        backbone_attention_mask=captured["backbone_attention_mask"],
                    )

                metrics = compute_metrics(ref_out, test_out)
                per_step_metrics[step_i].append(metrics)

    # Average per step
    result = {"per_step": [], "overall": {}}
    all_mse = []

    for step_i in range(num_denoise_steps):
        if not per_step_metrics[step_i]:
            continue
        avg = {}
        for key in per_step_metrics[step_i][0]:
            if key == "max_abs_error":
                avg[key] = max(m[key] for m in per_step_metrics[step_i])
            else:
                avg[key] = np.mean([m[key] for m in per_step_metrics[step_i]])
        result["per_step"].append(avg)
        all_mse.append(avg["mse"])

        logger.info(f"  DiT step {step_i}: MSE={avg['mse']:.6f}, MAE={avg['mae']:.6f}, "
                     f"cos_sim={avg['cosine_sim']:.6f}, max_err={avg['max_abs_error']:.6f}")

    # Overall (average across all steps)
    if result["per_step"]:
        for key in result["per_step"][0]:
            if key == "max_abs_error":
                result["overall"][key] = max(s[key] for s in result["per_step"])
            else:
                result["overall"][key] = np.mean([s[key] for s in result["per_step"]])

    # Error growth analysis
    if len(all_mse) > 1:
        growth = all_mse[-1] / (all_mse[0] + 1e-12)
        logger.info(f"  Error growth (step {len(all_mse)-1} / step 0): {growth:.2f}x")
        result["error_growth"] = growth

    return result


# ---------------------------------------------------------------------------
# E2E accuracy benchmark
# ---------------------------------------------------------------------------


def benchmark_e2e_accuracy(
    ref_policy: Gr00tPolicy,
    test_policy: Gr00tPolicy,
    dataset: LeRobotEpisodeLoader,
    traj_ids: list[int],
    action_horizon: int = 16,
    max_steps: int = 200,
) -> dict[str, float]:
    """
    Compare E2E action predictions between ref and test policies,
    and also compute MSE/MAE vs ground truth actions.
    """
    logger.info("Benchmarking E2E accuracy...")
    modality_configs = deepcopy(ref_policy.get_modality_config())
    modality_configs_no_action = deepcopy(modality_configs)
    modality_configs_no_action.pop("action", None)

    action_keys = modality_configs["action"].modality_keys

    all_ref_vs_gt_mse = []
    all_test_vs_gt_mse = []
    all_ref_vs_test_mse = []
    all_ref_vs_gt_mae = []
    all_test_vs_gt_mae = []

    for traj_id in traj_ids:
        traj = dataset[traj_id]
        traj_len = len(traj)
        actual_steps = min(max_steps, traj_len)

        ref_preds = []
        test_preds = []

        # Determine prediction chunk size from first inference
        # get_action() returns (batch, pred_horizon, dim) per key
        pred_horizon = action_horizon
        step_counts = list(range(0, actual_steps, pred_horizon))

        for step_count in step_counts:
            obs, _ = prepare_observation(
                ref_policy, dataset, traj_id, step_count, modality_configs_no_action
            )

            with torch.inference_mode():
                ref_action, _ = ref_policy.get_action(obs)
                test_action, _ = test_policy.get_action(obs)

            # get_action() returns keys without "action." prefix.
            # Shape per key: (batch, pred_horizon, action_dim)
            first_key = action_keys[0]
            ref_arr = np.array(ref_action[first_key])
            if ref_arr.ndim == 3:
                # (batch, horizon, dim) - squeeze batch
                chunk_len = ref_arr.shape[1]
                pred_horizon = chunk_len  # update for subsequent steps
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
                # Flat array (horizon, dim) or (dim,)
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

        # Extract ground truth (dataset uses "action.{k}" keys)
        gt_actions = np.concatenate(
            [np.vstack(traj[f"action.{k}"].values) for k in action_keys], axis=-1
        )[:actual_steps]

        # Shapes may not match exactly due to chunking; truncate to minimum
        min_len = min(len(gt_actions), len(ref_preds), len(test_preds))
        gt_actions = gt_actions[:min_len]
        ref_preds = ref_preds[:min_len]
        test_preds = test_preds[:min_len]

        ref_vs_gt_mse = np.mean((gt_actions - ref_preds) ** 2)
        test_vs_gt_mse = np.mean((gt_actions - test_preds) ** 2)
        ref_vs_test_mse = np.mean((ref_preds - test_preds) ** 2)
        ref_vs_gt_mae = np.mean(np.abs(gt_actions - ref_preds))
        test_vs_gt_mae = np.mean(np.abs(gt_actions - test_preds))

        all_ref_vs_gt_mse.append(ref_vs_gt_mse)
        all_test_vs_gt_mse.append(test_vs_gt_mse)
        all_ref_vs_test_mse.append(ref_vs_test_mse)
        all_ref_vs_gt_mae.append(ref_vs_gt_mae)
        all_test_vs_gt_mae.append(test_vs_gt_mae)

        logger.info(f"  Traj {traj_id}: ref_mse={ref_vs_gt_mse:.6f}, test_mse={test_vs_gt_mse:.6f}, "
                     f"ref_vs_test_mse={ref_vs_test_mse:.6f}")

    result = {
        "ref_vs_gt_mse": np.mean(all_ref_vs_gt_mse),
        "test_vs_gt_mse": np.mean(all_test_vs_gt_mse),
        "ref_vs_test_mse": np.mean(all_ref_vs_test_mse),
        "ref_vs_gt_mae": np.mean(all_ref_vs_gt_mae),
        "test_vs_gt_mae": np.mean(all_test_vs_gt_mae),
    }

    logger.info(f"  E2E average: ref_mse={result['ref_vs_gt_mse']:.6f}, "
                 f"test_mse={result['test_vs_gt_mse']:.6f}")

    return result


# ---------------------------------------------------------------------------
# Latency benchmark (reuses patterns from benchmark_inference.py)
# ---------------------------------------------------------------------------


def benchmark_latency(
    policy: Gr00tPolicy,
    dataset: LeRobotEpisodeLoader,
    traj_id: int = 0,
    num_iterations: int = 20,
    warmup: int = 5,
) -> dict[str, np.ndarray]:
    """
    Benchmark component-wise latency with torch.cuda.synchronize() barriers.
    Returns dict with arrays of timing values in ms.
    """
    logger.info("Benchmarking latency...")
    modality_configs = deepcopy(policy.get_modality_config())
    modality_configs.pop("action", None)

    obs, _ = prepare_observation(policy, dataset, traj_id, 0, modality_configs)
    collated_inputs = prepare_model_inputs(policy, obs)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            bb_in, act_in = policy.model.prepare_input(collated_inputs)
            bb_out = policy.model.backbone(bb_in)
            _ = policy.model.action_head.get_action(bb_out, act_in)
    torch.cuda.synchronize()
    gc.collect()

    backbone_times = []
    action_head_times = []

    for _ in range(num_iterations):
        collated_inputs = prepare_model_inputs(policy, obs)

        # Backbone
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            bb_in, act_in = policy.model.prepare_input(collated_inputs)
            bb_out = policy.model.backbone(bb_in)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        backbone_times.append((t1 - t0) * 1000)

        # Action head (DiT)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.action_head.get_action(bb_out, act_in)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        action_head_times.append((t3 - t2) * 1000)

    backbone_arr = np.array(backbone_times)
    action_head_arr = np.array(action_head_times)
    e2e_arr = backbone_arr + action_head_arr

    result = {
        "backbone": backbone_arr,
        "action_head": action_head_arr,
        "e2e": e2e_arr,
    }

    logger.info(f"  Backbone:    median={np.median(backbone_arr):.1f} ms, "
                 f"P90={np.percentile(backbone_arr, 90):.1f} ms")
    logger.info(f"  Action Head: median={np.median(action_head_arr):.1f} ms, "
                 f"P90={np.percentile(action_head_arr, 90):.1f} ms")
    logger.info(f"  E2E:         median={np.median(e2e_arr):.1f} ms "
                 f"({1000 / np.median(e2e_arr):.1f} Hz)")

    return result


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------


def print_summary_table(results: list[dict]):
    """Print a markdown comparison table with all results."""
    print("\n" + "=" * 120)
    print("QUANTIZATION BENCHMARK RESULTS")
    print("=" * 120)

    # Header
    print("\n| Mode | BB feat MSE | BB cos-sim | DiT MSE (last) | DiT cos-sim | "
          "E2E MSE (ref) | E2E MSE (test) | E2E MAE | BB ms | DiT ms | E2E ms | Hz |")
    print("|------|------------|-----------|----------------|------------|"
          "--------------|----------------|---------|-------|--------|--------|-----|")

    for r in results:
        mode = r["mode"]
        bb = r.get("backbone_accuracy", {})
        dit = r.get("dit_accuracy", {})
        e2e = r.get("e2e_accuracy", {})
        lat = r.get("latency", {})

        bb_mse = f"{bb.get('mse', float('nan')):.6f}" if bb else "n/a"
        bb_cos = f"{bb.get('cosine_sim', float('nan')):.6f}" if bb else "n/a"

        dit_overall = dit.get("overall", {})
        dit_mse = f"{dit_overall.get('mse', float('nan')):.6f}" if dit_overall else "n/a"
        dit_cos = f"{dit_overall.get('cosine_sim', float('nan')):.6f}" if dit_overall else "n/a"

        e2e_ref_mse = f"{e2e.get('ref_vs_gt_mse', float('nan')):.6f}" if e2e else "n/a"
        e2e_test_mse = f"{e2e.get('test_vs_gt_mse', float('nan')):.6f}" if e2e else "n/a"
        e2e_mae = f"{e2e.get('test_vs_gt_mae', float('nan')):.6f}" if e2e else "n/a"

        bb_ms = f"{np.median(lat['backbone']):.0f}" if lat and "backbone" in lat else "n/a"
        dit_ms = f"{np.median(lat['action_head']):.0f}" if lat and "action_head" in lat else "n/a"
        e2e_ms = f"{np.median(lat['e2e']):.0f}" if lat and "e2e" in lat else "n/a"
        hz = f"{1000 / np.median(lat['e2e']):.1f}" if lat and "e2e" in lat else "n/a"

        print(f"| {mode} | {bb_mse} | {bb_cos} | {dit_mse} | {dit_cos} | "
              f"{e2e_ref_mse} | {e2e_test_mse} | {e2e_mae} | {bb_ms} | {dit_ms} | {e2e_ms} | {hz} |")

    # Per-step error growth
    print("\n### Per-Denoising-Step DiT Error\n")
    print("| Mode | Step 0 MSE | Step 1 MSE | Step 2 MSE | Step 3 MSE | Growth |")
    print("|------|-----------|-----------|-----------|-----------|--------|")

    for r in results:
        dit = r.get("dit_accuracy", {})
        steps = dit.get("per_step", [])
        if not steps:
            continue
        step_strs = [f"{s['mse']:.6f}" for s in steps]
        while len(step_strs) < 4:
            step_strs.append("n/a")
        growth = dit.get("error_growth", float("nan"))
        print(f"| {r['mode']} | {step_strs[0]} | {step_strs[1]} | {step_strs[2]} | {step_strs[3]} | {growth:.2f}x |")

    print("\n" + "=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantization accuracy and latency")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--embodiment_tag", type=str, default="gr1")
    parser.add_argument("--trt_dit_path", type=str, default=None,
                        help="Path to TensorRT DiT engine. If not set, skips TRT DiT benchmarks.")
    parser.add_argument("--trt_backbone_path", type=str, default=None,
                        help="Path to TensorRT backbone engine. If not set, skips backbone TRT benchmarks.")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=[0],
                        help="Trajectory IDs to evaluate")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_latency_iterations", type=int, default=20)
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_e2e", action="store_true", help="Skip E2E accuracy (slow)")
    parser.add_argument("--skip_latency", action="store_true", help="Skip latency benchmarks")
    args = parser.parse_args()

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default dataset path
    if args.dataset_path is None:
        repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
        args.dataset_path = os.path.join(repo_path, "demo_data/gr1.PickNPlace")

    logger.info("=" * 80)
    logger.info("QUANTIZATION BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"TRT DiT: {args.trt_dit_path or 'none'}")
    logger.info(f"TRT Backbone: {args.trt_backbone_path or 'none'}")
    logger.info(f"Trajectories: {args.traj_ids}")

    # Load dataset
    logger.info("\nLoading reference policy (PyTorch BF16)...")
    ref_policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device=device,
    )

    modality_config = ref_policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    action_horizon = ref_policy.model.action_head.action_horizon
    num_denoise = ref_policy.model.action_head.num_inference_timesteps
    logger.info(f"Action Horizon: {action_horizon}, Denoising Steps: {num_denoise}")

    all_results = []

    # ---- 1. PyTorch BF16 baseline (latency only, accuracy = self) ----
    logger.info("\n" + "=" * 80)
    logger.info("=== PyTorch BF16 Baseline ===")
    logger.info("=" * 80)

    baseline_result = {"mode": "PyTorch BF16"}

    if not args.skip_latency:
        baseline_result["latency"] = benchmark_latency(
            ref_policy, dataset, args.traj_ids[0],
            args.num_latency_iterations, args.latency_warmup,
        )

    if not args.skip_e2e:
        # E2E vs ground truth (ref vs ref is just to get the baseline MSE)
        baseline_result["e2e_accuracy"] = benchmark_e2e_accuracy(
            ref_policy, ref_policy, dataset, args.traj_ids,
            action_horizon, args.max_steps,
        )

    all_results.append(baseline_result)

    # ---- 2. TRT DiT (if provided) ----
    if args.trt_dit_path and os.path.exists(args.trt_dit_path):
        logger.info("\n" + "=" * 80)
        logger.info(f"=== TRT DiT: {args.trt_dit_path} ===")
        logger.info("=" * 80)

        from standalone_inference_script import replace_dit_with_tensorrt

        trt_dit_policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
        )
        replace_dit_with_tensorrt(trt_dit_policy, args.trt_dit_path)

        trt_dit_result = {"mode": f"TRT DiT ({os.path.basename(args.trt_dit_path)})"}

        # DiT accuracy (per-step)
        trt_dit_result["dit_accuracy"] = benchmark_dit_accuracy(
            ref_policy, trt_dit_policy, dataset, args.traj_ids,
        )

        # E2E accuracy
        if not args.skip_e2e:
            trt_dit_result["e2e_accuracy"] = benchmark_e2e_accuracy(
                ref_policy, trt_dit_policy, dataset, args.traj_ids,
                action_horizon, args.max_steps,
            )

        # Latency
        if not args.skip_latency:
            trt_dit_result["latency"] = benchmark_latency(
                trt_dit_policy, dataset, args.traj_ids[0],
                args.num_latency_iterations, args.latency_warmup,
            )

        all_results.append(trt_dit_result)

    # ---- 3. TRT Backbone (if provided) ----
    if args.trt_backbone_path and os.path.exists(args.trt_backbone_path):
        logger.info("\n" + "=" * 80)
        logger.info(f"=== TRT Backbone: {args.trt_backbone_path} ===")
        logger.info("=" * 80)

        from standalone_inference_script import replace_backbone_with_tensorrt

        trt_bb_policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
        )
        replace_backbone_with_tensorrt(trt_bb_policy, args.trt_backbone_path)

        trt_bb_result = {"mode": f"TRT Backbone ({os.path.basename(args.trt_backbone_path)})"}

        # Backbone accuracy
        trt_bb_result["backbone_accuracy"] = benchmark_backbone_accuracy(
            ref_policy, trt_bb_policy, dataset, args.traj_ids,
        )

        # E2E accuracy (TRT backbone + PyTorch DiT)
        if not args.skip_e2e:
            trt_bb_result["e2e_accuracy"] = benchmark_e2e_accuracy(
                ref_policy, trt_bb_policy, dataset, args.traj_ids,
                action_horizon, args.max_steps,
            )

        # Latency
        if not args.skip_latency:
            trt_bb_result["latency"] = benchmark_latency(
                trt_bb_policy, dataset, args.traj_ids[0],
                args.num_latency_iterations, args.latency_warmup,
            )

        all_results.append(trt_bb_result)

    # ---- 4. Full INT8 (both TRT backbone + TRT DiT) ----
    if (args.trt_dit_path and os.path.exists(args.trt_dit_path) and
            args.trt_backbone_path and os.path.exists(args.trt_backbone_path)):
        logger.info("\n" + "=" * 80)
        logger.info("=== Full TRT INT8 (Backbone + DiT) ===")
        logger.info("=" * 80)

        from standalone_inference_script import replace_dit_with_tensorrt, replace_backbone_with_tensorrt

        full_trt_policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
        )
        replace_backbone_with_tensorrt(full_trt_policy, args.trt_backbone_path)
        replace_dit_with_tensorrt(full_trt_policy, args.trt_dit_path)

        full_result = {"mode": "Full TRT INT8"}

        # Backbone accuracy
        full_result["backbone_accuracy"] = benchmark_backbone_accuracy(
            ref_policy, full_trt_policy, dataset, args.traj_ids,
        )

        # DiT accuracy
        full_result["dit_accuracy"] = benchmark_dit_accuracy(
            ref_policy, full_trt_policy, dataset, args.traj_ids,
        )

        # E2E accuracy
        if not args.skip_e2e:
            full_result["e2e_accuracy"] = benchmark_e2e_accuracy(
                ref_policy, full_trt_policy, dataset, args.traj_ids,
                action_horizon, args.max_steps,
            )

        # Latency
        if not args.skip_latency:
            full_result["latency"] = benchmark_latency(
                full_trt_policy, dataset, args.traj_ids[0],
                args.num_latency_iterations, args.latency_warmup,
            )

        all_results.append(full_result)

    # ---- Print results ----
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
