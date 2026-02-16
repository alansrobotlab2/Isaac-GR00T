#!/usr/bin/env python3
"""
SmoothQuant for Eagle Backbone → FP16 TensorRT on Orin AGX

The Eagle backbone's internal activations overflow FP16's ±65504 range, making
TRT FP16 unusable (cos_sim=0.354). SmoothQuant mathematically compresses activation
dynamic range by redistributing it to weights via per-channel scaling:

    Y = X @ W  →  Y = (X / s) @ (s * W)

The transformation is mathematically equivalent but shifts outlier magnitude from
activations (which overflow FP16) to weights (which are well-behaved). After smoothing,
activations fit within FP16 range, enabling FP16 tensor cores at ~149ms.

Phases:
  1. profile  - Analyze per-layer activation ranges, identify FP16 overflow points
  2. smooth   - Apply SmoothQuant to model weights, export smoothed ONNX
  3. verify   - Verify smoothed model matches original (PyTorch-level sanity check)

Usage:
    # Phase 1: Profile activations (understand where overflow happens)
    python scripts/deployment/smoothquant_backbone.py profile \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --num_samples 20

    # Phase 2: Apply SmoothQuant + export ONNX
    python scripts/deployment/smoothquant_backbone.py smooth \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --output_dir groot_n1d6_onnx_smoothquant \
        --alpha 0.5 \
        --num_samples 50

    # Phase 3: Verify smoothed model matches original
    python scripts/deployment/smoothquant_backbone.py verify \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --output_dir groot_n1d6_onnx_smoothquant \
        --alpha 0.5 \
        --num_samples 50

    # Then build FP16 TRT engine (inside Docker):
    python scripts/deployment/build_tensorrt_engine.py \
        --onnx groot_n1d6_onnx_smoothquant/backbone_model.onnx \
        --engine groot_n1d6_onnx_smoothquant/backbone_smoothquant_fp16.trt \
        --precision fp16 \
        --calib-data calibration_data_backbone/calib_data.npz \
        --max-seq-len 512

    # Benchmark (inside Docker):
    python scripts/deployment/benchmark_backbone_pipeline.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --trt_fp16_path groot_n1d6_onnx_smoothquant/backbone_smoothquant_fp16.trt \
        --skip_onnx
"""

import argparse
import gc
import json
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collection (reused from benchmark_backbone_pipeline.py)
# ---------------------------------------------------------------------------

def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def prepare_observation(policy, dataset, traj_idx, step_idx):
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


def collect_backbone_inputs(policy, dataset, num_samples=20):
    """Run inference and collect backbone inputs via hooks."""
    logger.info(f"Collecting {num_samples} backbone input samples...")

    samples = []
    modality_config = policy.get_modality_config()
    num_trajs = min(num_samples, len(dataset))

    for traj_idx in range(num_trajs):
        traj = dataset[traj_idx]
        traj_len = len(traj)
        steps_needed = max(1, num_samples // num_trajs)
        step_indices = list(range(0, min(traj_len, steps_needed * 16), 16))[:steps_needed]

        for step_idx in step_indices:
            if len(samples) >= num_samples:
                break

            obs = prepare_observation(policy, dataset, traj_idx, step_idx)

            # Process through VLA pipeline (same as policy.get_action)
            unbatched = []
            video_keys = list(obs["video"].keys())
            batch_size = obs["video"][video_keys[0]].shape[0]
            for i in range(batch_size):
                unbatched.append({
                    "video": {k: v[i] for k, v in obs["video"].items()},
                    "state": {k: v[i] for k, v in obs["state"].items()},
                    "language": {k: v[i] for k, v in obs["language"].items()},
                })

            processed = []
            for o in unbatched:
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

            pv = bb_inputs["pixel_values"]
            samples.append({
                "input_ids": bb_inputs["input_ids"].detach().cpu(),
                "attention_mask": bb_inputs["attention_mask"].detach().cpu(),
                "pixel_values": [t.detach().cpu() for t in pv] if isinstance(pv, list) else pv.detach().cpu(),
                "ref_features": ref_out.backbone_features.detach().cpu().float(),
            })

        if len(samples) >= num_samples:
            break

    logger.info(f"Collected {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Phase 1: Activation Range Profiling
# ---------------------------------------------------------------------------

class ActivationProfiler:
    """Hook-based profiler that records per-channel activation statistics."""

    def __init__(self):
        self.stats = defaultdict(lambda: {"max": [], "mean": [], "shape": None, "overflow_count": 0})
        self.hooks = []

    def register(self, model):
        """Register hooks on all linear and norm layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._record(name, mod, inp, out)
                )
                self.hooks.append(hook)
            # Also catch RMSNorm (Qwen2)
            elif "RMSNorm" in type(module).__name__:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._record(name, mod, inp, out)
                )
                self.hooks.append(hook)

    def _record(self, name, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        stats = self.stats[name]
        flat = output.detach().float()
        # Per-channel max absolute value (last dim = channels)
        if flat.dim() >= 2:
            channel_max = flat.abs().reshape(-1, flat.shape[-1]).max(dim=0).values.cpu()
        else:
            channel_max = flat.abs().cpu()

        stats["max"].append(channel_max)
        stats["mean"].append(flat.abs().mean().item())
        stats["shape"] = tuple(output.shape)

        # Count FP16 overflows
        fp16_max = 65504.0
        overflow = (flat.abs() > fp16_max).sum().item()
        stats["overflow_count"] += overflow

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def summarize(self):
        """Return sorted summary of activation ranges."""
        summary = []
        for name, stats in self.stats.items():
            if not stats["max"]:
                continue
            all_max = torch.stack(stats["max"])
            global_max = all_max.max().item()
            mean_max = all_max.max(dim=1).values.mean().item()
            channel_max_avg = all_max.mean(dim=0)
            p99 = torch.quantile(all_max.max(dim=1).values.float(), 0.99).item()

            summary.append({
                "name": name,
                "global_max": global_max,
                "mean_max": mean_max,
                "p99_max": p99,
                "mean_abs": np.mean(stats["mean"]),
                "shape": stats["shape"],
                "overflow_count": stats["overflow_count"],
                "num_channels": channel_max_avg.shape[0] if channel_max_avg.dim() > 0 else 1,
                "outlier_ratio": (channel_max_avg > 65504).float().mean().item() if channel_max_avg.dim() > 0 else 0,
                "channel_max_avg": channel_max_avg,
            })

        summary.sort(key=lambda x: x["global_max"], reverse=True)
        return summary


def phase_profile(policy, dataset, num_samples):
    """Profile per-layer activation ranges to identify FP16 overflow sources."""
    logger.info("=" * 80)
    logger.info("PHASE 1: Activation Range Profiling")
    logger.info("=" * 80)

    profiler = ActivationProfiler()
    profiler.register(policy.model.backbone)

    # Run calibration samples through backbone
    samples = collect_backbone_inputs(policy, dataset, num_samples)

    profiler.remove_hooks()
    summary = profiler.summarize()

    # Print results
    FP16_MAX = 65504.0
    print("\n" + "=" * 120)
    print(f"ACTIVATION RANGE PROFILE ({len(samples)} samples)")
    print("=" * 120)
    print(f"{'Layer':<70} {'Max':>12} {'P99 Max':>12} {'Overflow':>10} {'FP16?':>6}")
    print("-" * 120)

    overflow_layers = []
    safe_layers = []
    for s in summary:
        is_overflow = s["global_max"] > FP16_MAX
        marker = " OVERFLOW" if is_overflow else "      OK"
        short_name = s["name"]
        if len(short_name) > 68:
            short_name = "..." + short_name[-65:]
        print(f"{short_name:<70} {s['global_max']:>12.1f} {s['p99_max']:>12.1f} "
              f"{s['overflow_count']:>10} {marker}")
        if is_overflow:
            overflow_layers.append(s)
        else:
            safe_layers.append(s)

    print("-" * 120)
    print(f"\nTotal layers profiled: {len(summary)}")
    print(f"Layers with FP16 overflow (max > {FP16_MAX}): {len(overflow_layers)}")
    print(f"Safe layers: {len(safe_layers)}")

    if overflow_layers:
        print(f"\n{'='*80}")
        print("TOP OVERFLOW LAYERS (sorted by max activation)")
        print(f"{'='*80}")
        for s in overflow_layers[:20]:
            print(f"  {s['name']}")
            print(f"    max={s['global_max']:.1f}, p99={s['p99_max']:.1f}, "
                  f"overflow_count={s['overflow_count']}, "
                  f"outlier_channel_ratio={s['outlier_ratio']:.3f}")

    # Categorize overflows
    vision_overflow = [s for s in overflow_layers if "vision" in s["name"]]
    language_overflow = [s for s in overflow_layers if "language" in s["name"]]
    other_overflow = [s for s in overflow_layers if "vision" not in s["name"] and "language" not in s["name"]]

    print(f"\nOverflow breakdown:")
    print(f"  Vision encoder (SigLIP2): {len(vision_overflow)} layers")
    print(f"  Language model (Qwen2):   {len(language_overflow)} layers")
    print(f"  Other (MLP1/embed):       {len(other_overflow)} layers")

    # SmoothQuant viability assessment
    print(f"\n{'='*80}")
    print("SMOOTHQUANT VIABILITY ASSESSMENT")
    print(f"{'='*80}")

    norm_linear_overflows = 0
    attention_overflows = 0
    mlp_overflows = 0
    other_type_overflows = 0

    for s in overflow_layers:
        name = s["name"]
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]):
            attention_overflows += 1
        elif any(x in name for x in ["fc1", "fc2", "gate_proj", "up_proj", "down_proj"]):
            mlp_overflows += 1
        elif any(x in name for x in ["norm", "layernorm"]):
            norm_linear_overflows += 1
        else:
            other_type_overflows += 1

    print(f"  Attention projection overflows: {attention_overflows} (SmoothQuant CAN help)")
    print(f"  MLP/FFN overflows:              {mlp_overflows} (SmoothQuant CAN help)")
    print(f"  Norm output overflows:          {norm_linear_overflows} (SmoothQuant targets these)")
    print(f"  Other overflows:                {other_type_overflows} (SmoothQuant may NOT help)")

    smoothable = attention_overflows + mlp_overflows + norm_linear_overflows
    total_ov = len(overflow_layers)
    if total_ov > 0:
        coverage = smoothable / total_ov * 100
        print(f"\n  SmoothQuant coverage: {smoothable}/{total_ov} overflow layers ({coverage:.0f}%)")
        if coverage > 80:
            print("  --> PROMISING: Most overflows are in SmoothQuant-targetable layers")
        elif coverage > 50:
            print("  --> MIXED: Some overflows are in non-smoothable locations")
        else:
            print("  --> UNLIKELY TO HELP: Most overflows are in non-smoothable locations")
    else:
        print("\n  No FP16 overflows detected! Model may work with FP16 TRT directly.")

    return summary, samples


# ---------------------------------------------------------------------------
# Phase 2: SmoothQuant Application + ONNX Export
# ---------------------------------------------------------------------------

class SmoothQuantCalibrator:
    """
    Collect per-channel activation statistics for SmoothQuant.

    For each (Norm → Linear) pair, records max |activation| per input channel
    of the Linear layer across calibration samples.
    """

    def __init__(self):
        self.act_scales = {}  # layer_name -> Tensor[channels]
        self.hooks = []

    def register(self, model):
        """Register hooks on all Linear layers to capture input activation scales."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self._record_input(n, inp)
                )
                self.hooks.append(hook)

    def _record_input(self, name, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        x = inputs[0].detach().float()
        # Per input-channel max absolute value
        if x.dim() >= 2:
            channel_max = x.abs().reshape(-1, x.shape[-1]).max(dim=0).values
        else:
            channel_max = x.abs()

        if name in self.act_scales:
            self.act_scales[name] = torch.max(self.act_scales[name], channel_max.cpu())
        else:
            self.act_scales[name] = channel_max.cpu()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def find_norm_linear_pairs(model):
    """
    Find (Norm, [Linear...]) pairs in the Eagle backbone.

    Returns list of dicts with:
        - norm_name: full module name of the norm layer
        - norm_module: the norm module
        - linears: list of (name, module) for the linear layers fed by this norm
    """
    pairs = []
    modules = dict(model.named_modules())

    # SigLIP2 vision encoder: layer_norm1 → {q_proj, k_proj, v_proj, out_proj}, layer_norm2 → mlp.{fc1}
    for name, mod in modules.items():
        if name.endswith(".layer_norm1"):
            # Find the corresponding self_attn projections
            prefix = name.rsplit(".layer_norm1", 1)[0]
            linears = []
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.out_proj"]:
                full = f"{prefix}.{proj}"
                if full in modules:
                    linears.append((full, modules[full]))
            # Also check for down_proj (some SigLIP2 layers have it)
            for proj in ["self_attn.down_proj"]:
                full = f"{prefix}.{proj}"
                if full in modules:
                    linears.append((full, modules[full]))
            if linears:
                pairs.append({"norm_name": name, "norm_module": mod, "linears": linears})

        elif name.endswith(".layer_norm2"):
            prefix = name.rsplit(".layer_norm2", 1)[0]
            linears = []
            for proj in ["mlp.fc1"]:
                full = f"{prefix}.{proj}"
                if full in modules:
                    linears.append((full, modules[full]))
            if linears:
                pairs.append({"norm_name": name, "norm_module": mod, "linears": linears})

    # Qwen2 language model: input_layernorm → {q_proj, k_proj, v_proj}
    # post_attention_layernorm → {gate_proj, up_proj}
    for name, mod in modules.items():
        if name.endswith(".input_layernorm"):
            prefix = name.rsplit(".input_layernorm", 1)[0]
            linears = []
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
                full = f"{prefix}.{proj}"
                if full in modules:
                    linears.append((full, modules[full]))
            if linears:
                pairs.append({"norm_name": name, "norm_module": mod, "linears": linears})

        elif name.endswith(".post_attention_layernorm"):
            prefix = name.rsplit(".post_attention_layernorm", 1)[0]
            linears = []
            for proj in ["mlp.gate_proj", "mlp.up_proj"]:
                full = f"{prefix}.{proj}"
                if full in modules:
                    linears.append((full, modules[full]))
            if linears:
                pairs.append({"norm_name": name, "norm_module": mod, "linears": linears})

    # MLP1 projection: mlp1.0 (LayerNorm) → mlp1.1 (Linear)
    for name, mod in modules.items():
        if name.endswith("mlp1.0") and isinstance(mod, nn.LayerNorm):
            linear_name = name.replace("mlp1.0", "mlp1.1")
            if linear_name in modules:
                pairs.append({
                    "norm_name": name,
                    "norm_module": mod,
                    "linears": [(linear_name, modules[linear_name])],
                })

    return pairs


def apply_smoothquant(model, act_scales, alpha=0.5, clip_ratio=None):
    """
    Apply SmoothQuant transformation to norm-linear pairs.

    For each (Norm → [Linear_1, Linear_2, ...]) group:
        1. Compute shared smoothing factor s from activation and weight statistics
        2. Norm: weight /= s, bias /= s (absorb 1/s into preceding norm)
        3. Each Linear_i: weight *= s along input dim (compensate with s on weights)

    The transformation is mathematically equivalent: Y = X @ W = (X/s) @ (s*W)
    but compresses activation dynamic range into FP16-representable territory.

    Args:
        model: The backbone model (modified in-place)
        act_scales: Dict mapping linear layer name → per-channel max |activation|
        alpha: Smoothing strength (0=all on activations, 1=all on weights). Higher
               values push more dynamic range to weights, compressing activations more.
        clip_ratio: If set, clip extreme smoothing factors to this ratio of median

    Returns:
        Dict with smoothing statistics
    """
    pairs = find_norm_linear_pairs(model)
    logger.info(f"Found {len(pairs)} norm-linear pairs for SmoothQuant")

    stats = {
        "num_pairs": len(pairs),
        "alpha": alpha,
        "groups_smoothed": 0,
        "groups_skipped": 0,
        "linears_smoothed": 0,
        "max_smooth_factor": 0,
        "min_smooth_factor": float("inf"),
        "details": [],
    }

    for pair in pairs:
        norm_name = pair["norm_name"]
        norm = pair["norm_module"]
        linears = pair["linears"]

        # All downstream linears see the same norm output (same activation distribution).
        # Verify they all have matching activation scales (they should, since act_scales
        # captures the LINEAR INPUT, which is the norm output).
        available_linears = [(n, m) for n, m in linears if n in act_scales]
        if not available_linears:
            stats["groups_skipped"] += 1
            continue

        # Activation scale: max across all downstream linears (should be identical
        # since they all see the same norm output, but take max for safety)
        act_scale = act_scales[available_linears[0][0]].clone()
        for linear_name, _ in available_linears[1:]:
            s = act_scales[linear_name]
            if s.shape == act_scale.shape:
                act_scale = torch.max(act_scale, s)

        # Weight scale: max across ALL downstream linears per input channel.
        # This ensures the shared smooth factor doesn't overflow any linear's weights.
        wt_scale = None
        for linear_name, linear_mod in available_linears:
            w = linear_mod.weight.detach().float()  # [out_features, in_features]
            ws = w.abs().max(dim=0).values  # [in_features]
            if wt_scale is None:
                wt_scale = ws
            elif ws.shape == wt_scale.shape:
                wt_scale = torch.max(wt_scale, ws)
            else:
                logger.warning(f"  Weight shape mismatch in {norm_name} group: "
                               f"{ws.shape} vs {wt_scale.shape}")

        if wt_scale is None or act_scale.shape != wt_scale.shape:
            logger.warning(f"  Skipping {norm_name}: dimension mismatch "
                           f"act={act_scale.shape} wt={wt_scale.shape if wt_scale is not None else None}")
            stats["groups_skipped"] += 1
            continue

        # Compute shared smoothing factor: s = act^alpha / wt^(1-alpha)
        # Large alpha → s is large → more compression on activations, more expansion on weights
        eps = 1e-5
        smooth = (act_scale.clamp(min=eps).pow(alpha) /
                  wt_scale.clamp(min=eps).pow(1 - alpha))

        # Optional: clip extreme factors to prevent weight overflow
        if clip_ratio is not None:
            median_s = smooth.median()
            smooth = smooth.clamp(
                min=median_s / clip_ratio,
                max=median_s * clip_ratio
            )

        # Safety check: ensure smoothed weights don't overflow FP16 for ANY downstream linear
        for linear_name, linear_mod in available_linears:
            w = linear_mod.weight.detach().float()
            smoothed_wt_max = (w * smooth.unsqueeze(0)).abs().max().item()
            if smoothed_wt_max > 65504:
                safety = 65504 / smoothed_wt_max * 0.95
                smooth = smooth * safety
                logger.info(f"  {norm_name}: reduced smooth factors by {safety:.3f} "
                            f"to prevent weight FP16 overflow in {linear_name}")

        stats["max_smooth_factor"] = max(stats["max_smooth_factor"], smooth.max().item())
        stats["min_smooth_factor"] = min(stats["min_smooth_factor"], smooth.min().item())

        # --- Apply transformation ---

        # 1. Scale norm: divide weight (and bias) by s
        #    This means norm output X_new = X_old / s
        with torch.no_grad():
            device = norm.weight.device
            dtype = norm.weight.dtype
            s_device = smooth.to(device=device, dtype=dtype)

            norm.weight.div_(s_device)
            if hasattr(norm, 'bias') and norm.bias is not None:
                norm.bias.div_(s_device)

        # 2. Scale each downstream linear: multiply weight's input dim by s
        #    Linear weight shape: [out_features, in_features]
        #    Multiply each column (input channel j) by s[j]
        #    This means Y_new = X_new @ W_new = (X/s) @ (s*W) = X @ W = Y_old
        for linear_name, linear_mod in available_linears:
            with torch.no_grad():
                linear_mod.weight.mul_(
                    s_device.unsqueeze(0).to(dtype=linear_mod.weight.dtype)
                )
                # Bias is NOT scaled (it's added after the matmul, not affected by input scaling)

            stats["linears_smoothed"] += 1
            w_after = linear_mod.weight.detach().float()
            stats["details"].append({
                "norm": norm_name,
                "linear": linear_name,
                "smooth_min": smooth.min().item(),
                "smooth_max": smooth.max().item(),
                "smooth_mean": smooth.mean().item(),
                "act_max_before": act_scale.max().item(),
                "act_max_after": (act_scale / smooth).max().item(),
                "wt_max_after": w_after.abs().max().item(),
            })

        stats["groups_smoothed"] += 1

    logger.info(f"\nSmoothQuant applied:")
    logger.info(f"  Groups smoothed: {stats['groups_smoothed']}/{stats['num_pairs']}")
    logger.info(f"  Linear layers smoothed: {stats['linears_smoothed']}")
    logger.info(f"  Groups skipped: {stats['groups_skipped']}")
    if stats['linears_smoothed'] > 0:
        logger.info(f"  Smooth factor range: [{stats['min_smooth_factor']:.4f}, "
                     f"{stats['max_smooth_factor']:.4f}]")

    return stats


def export_smoothed_onnx(policy, output_dir, samples):
    """Export the smoothed backbone to ONNX (FP32 weights)."""
    from scripts.deployment.export_backbone_onnx import (
        BackboneInputCapture, export_backbone_to_onnx
    )

    # Use first sample as representative input
    s0 = samples[0]

    # Create a fake capture object
    capture = BackboneInputCapture()
    capture.captured = True
    capture.input_ids = s0["input_ids"]
    capture.attention_mask = s0["attention_mask"]
    capture.pixel_values = s0["pixel_values"]

    output_path = os.path.join(output_dir, "backbone_model.onnx")
    export_backbone_to_onnx(
        policy=policy,
        captured_inputs=capture,
        output_path=output_path,
        export_dtype="fp32",  # FP32 weights, let TRT handle precision
    )
    logger.info(f"Smoothed ONNX exported to {output_path}")
    return output_path


def phase_smooth(policy, dataset, output_dir, alpha, num_samples, clip_ratio=None):
    """Apply SmoothQuant and export smoothed ONNX."""
    logger.info("=" * 80)
    logger.info(f"PHASE 2: SmoothQuant (alpha={alpha})")
    logger.info("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Collect calibration samples
    samples = collect_backbone_inputs(policy, dataset, num_samples)

    # Step 2: Calibrate activation scales
    logger.info("\n[Step 2] Calibrating activation scales...")
    calibrator = SmoothQuantCalibrator()
    calibrator.register(policy.model.backbone)

    # Run samples through backbone again to collect per-linear activation stats
    for i, s in enumerate(samples):
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.bfloat16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.bfloat16)

        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            policy.model.backbone(bb_in)

        if (i + 1) % 10 == 0:
            logger.info(f"  Calibrated {i + 1}/{len(samples)} samples")

    calibrator.remove_hooks()
    logger.info(f"  Collected activation scales for {len(calibrator.act_scales)} linear layers")

    # Step 3: Analyze activation scales
    logger.info("\n[Step 3] Activation scale analysis...")
    overflow_linears = 0
    for name, scale in sorted(calibrator.act_scales.items()):
        max_val = scale.max().item()
        if max_val > 65504:
            overflow_linears += 1
            logger.info(f"  OVERFLOW: {name}: max_act={max_val:.1f}")

    logger.info(f"  {overflow_linears} linear layers have FP16-overflowing inputs")

    # Step 4: Switch to SDPA attention for ONNX export
    logger.info("\n[Step 4] Switching backbone to SDPA attention...")
    backbone = policy.model.backbone
    count = 0
    for name, module in backbone.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation == "flash_attention_2":
                module.config._attn_implementation = "sdpa"
                count += 1
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation == "flash_attention_2":
                module._attn_implementation = "sdpa"
                count += 1
    logger.info(f"  Switched {count} modules from flash_attention_2 to sdpa")

    # Step 5: Apply SmoothQuant
    logger.info("\n[Step 5] Applying SmoothQuant transformation...")
    smooth_stats = apply_smoothquant(
        backbone, calibrator.act_scales, alpha=alpha, clip_ratio=clip_ratio
    )

    # Save smoothing stats
    stats_path = os.path.join(output_dir, "smoothquant_stats.json")
    serializable_stats = {k: v for k, v in smooth_stats.items() if k != "details"}
    serializable_stats["details"] = smooth_stats["details"]
    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2, default=str)
    logger.info(f"  Stats saved to {stats_path}")

    # Step 6: Export smoothed ONNX
    logger.info("\n[Step 6] Exporting smoothed backbone to ONNX...")
    onnx_path = export_smoothed_onnx(policy, output_dir, samples)

    logger.info("\n" + "=" * 80)
    logger.info("SMOOTHQUANT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  ONNX: {onnx_path}")
    logger.info(f"  Stats: {stats_path}")
    logger.info(f"\nNext steps (inside Docker):")
    logger.info(f"  # Build FP16 TRT engine:")
    logger.info(f"  python scripts/deployment/build_tensorrt_engine.py \\")
    logger.info(f"    --onnx {onnx_path} \\")
    logger.info(f"    --engine {output_dir}/backbone_smoothquant_fp16.trt \\")
    logger.info(f"    --precision fp16 \\")
    logger.info(f"    --calib-data calibration_data_backbone/calib_data.npz \\")
    logger.info(f"    --max-seq-len 512")
    logger.info(f"\n  # Benchmark:")
    logger.info(f"  python scripts/deployment/benchmark_backbone_pipeline.py \\")
    logger.info(f"    --model_path alfie-gr00t/checkpoint-10000 \\")
    logger.info(f"    --dataset_path alfiebot.CanDoChallenge \\")
    logger.info(f"    --embodiment_tag new_embodiment \\")
    logger.info(f"    --trt_fp16_path {output_dir}/backbone_smoothquant_fp16.trt \\")
    logger.info(f"    --skip_onnx")

    return onnx_path, smooth_stats


# ---------------------------------------------------------------------------
# Phase 3: Verification
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a_flat = a.reshape(a.shape[0], -1).float()
    b_flat = b.reshape(b.shape[0], -1).float()
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def phase_verify(policy, dataset, output_dir, alpha, num_samples, clip_ratio=None):
    """
    Verify smoothed model against original in PyTorch (before ONNX/TRT).

    Loads the model fresh, applies SmoothQuant, and compares outputs.
    This checks that the mathematical transformation is correct.
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: Verification (smoothed vs original in PyTorch)")
    logger.info("=" * 80)

    # Collect reference outputs from original model
    logger.info("[Step 1] Collecting reference outputs from original backbone...")
    samples = collect_backbone_inputs(policy, dataset, num_samples)
    ref_outputs = [s["ref_features"] for s in samples]

    # Calibrate
    logger.info("\n[Step 2] Calibrating activation scales...")
    calibrator = SmoothQuantCalibrator()
    calibrator.register(policy.model.backbone)

    for i, s in enumerate(samples):
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.bfloat16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.bfloat16)

        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            policy.model.backbone(bb_in)

    calibrator.remove_hooks()

    # Apply SmoothQuant (modifies model in-place)
    logger.info("\n[Step 3] Applying SmoothQuant...")
    apply_smoothquant(policy.model.backbone, calibrator.act_scales,
                      alpha=alpha, clip_ratio=clip_ratio)

    # Run smoothed model
    logger.info("\n[Step 4] Running smoothed backbone...")
    smoothed_outputs = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.bfloat16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.bfloat16)

        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            out = policy.model.backbone(bb_in)
        smoothed_outputs.append(out.backbone_features.detach().cpu().float())

    # Compare
    logger.info("\n[Step 5] Comparing outputs...")
    mses = []
    cos_sims = []
    max_abs_errors = []
    for ref, smooth in zip(ref_outputs, smoothed_outputs):
        diff = ref - smooth
        mses.append((diff ** 2).mean().item())
        cos_sims.append(cosine_sim(ref, smooth))
        max_abs_errors.append(diff.abs().max().item())

    print("\n" + "=" * 80)
    print("SMOOTHQUANT VERIFICATION RESULTS (PyTorch BF16)")
    print("=" * 80)
    print(f"  Samples:     {len(samples)}")
    print(f"  Alpha:       {alpha}")
    print(f"  MSE:         {np.mean(mses):.6f} (should be ~0 if math is correct)")
    print(f"  Cos Sim:     {np.mean(cos_sims):.6f} (should be ~1.0)")
    print(f"  Max Abs Err: {max(max_abs_errors):.6f}")
    print()

    if np.mean(cos_sims) > 0.999:
        print("  PASS: SmoothQuant transformation is mathematically correct")
        print("  The smoothed model produces identical outputs to the original.")
    elif np.mean(cos_sims) > 0.99:
        print("  MARGINAL: Small numerical differences (expected from BF16 precision)")
        print("  The smoothed model is very close but not identical. This is likely")
        print("  due to floating-point order-of-operations differences in BF16.")
    else:
        print("  WARNING: Significant difference detected!")
        print("  The SmoothQuant transformation may have a bug or the smoothing")
        print("  factors are too aggressive.")

    # Profile activation ranges AFTER smoothing
    logger.info("\n[Step 6] Profiling activation ranges after smoothing...")
    profiler = ActivationProfiler()
    profiler.register(policy.model.backbone)

    for s in samples[:5]:  # Just a few samples
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.bfloat16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.bfloat16)

        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            policy.model.backbone(bb_in)

    profiler.remove_hooks()
    summary = profiler.summarize()

    overflow_after = sum(1 for s in summary if s["global_max"] > 65504)
    total = len(summary)
    print(f"\n  Activation range after smoothing:")
    print(f"    Total layers:            {total}")
    print(f"    FP16 overflow layers:    {overflow_after}")
    print(f"    Top 5 max activations:")
    for s in summary[:5]:
        marker = " OVERFLOW!" if s["global_max"] > 65504 else ""
        print(f"      {s['name'][:60]:<60} max={s['global_max']:.1f}{marker}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_policy(args, attn_implementation=None):
    """Load the GR00T policy."""
    logger.info("Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
        attn_implementation=attn_implementation,
    )
    return policy


def load_dataset(args, policy):
    """Load the dataset."""
    logger.info("Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
    )
    logger.info(f"Dataset loaded ({len(dataset)} trajectories)")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="SmoothQuant for Eagle Backbone FP16 TensorRT"
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_path", default="alfie-gr00t/checkpoint-10000")
    common.add_argument("--dataset_path", default="alfiebot.CanDoChallenge")
    common.add_argument("--embodiment_tag", default="new_embodiment")
    common.add_argument("--video_backend", default="torchcodec")
    common.add_argument("--num_samples", type=int, default=20)
    common.add_argument("--seed", type=int, default=42)

    # Phase 1: Profile
    p1 = subparsers.add_parser("profile", parents=[common],
                                help="Profile per-layer activation ranges")

    # Phase 2: Smooth + Export
    p2 = subparsers.add_parser("smooth", parents=[common],
                                help="Apply SmoothQuant and export ONNX")
    p2.add_argument("--output_dir", default="groot_n1d6_onnx_smoothquant")
    p2.add_argument("--alpha", type=float, default=0.5,
                    help="Smoothing strength (0=all on activations, 1=all on weights)")
    p2.add_argument("--clip_ratio", type=float, default=None,
                    help="Clip extreme smooth factors to this ratio of median")

    # Phase 3: Verify
    p3 = subparsers.add_parser("verify", parents=[common],
                                help="Verify smoothed model matches original")
    p3.add_argument("--output_dir", default="groot_n1d6_onnx_smoothquant")
    p3.add_argument("--alpha", type=float, default=0.5)
    p3.add_argument("--clip_ratio", type=float, default=None)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.phase == "profile":
        policy = load_policy(args)
        dataset = load_dataset(args, policy)
        phase_profile(policy, dataset, args.num_samples)

    elif args.phase == "smooth":
        policy = load_policy(args)
        dataset = load_dataset(args, policy)
        phase_smooth(policy, dataset, args.output_dir, args.alpha,
                     args.num_samples, args.clip_ratio)

    elif args.phase == "verify":
        policy = load_policy(args)
        dataset = load_dataset(args, policy)
        phase_verify(policy, dataset, args.output_dir, args.alpha,
                     args.num_samples, args.clip_ratio)


if __name__ == "__main__":
    main()
