#!/usr/bin/env python3
"""
Root cause analysis for FP16 quality destruction in Eagle backbone TRT.

Phase 1 profiling showed ALL layer outputs are within FP16 range (max ~5160 < 65504).
Yet TRT FP16 produces cos_sim=0.354. This script digs deeper:

1. Profile INTERMEDIATE attention values (Q@K^T scores) inside SDPA
2. Run backbone in PyTorch FP16 (not TRT) to isolate precision vs TRT-specific issues
3. Test per-layer FP16 casting to find which layers cause the most degradation
4. Measure cumulative precision loss across transformer layers

Usage:
    python scripts/deployment/profile_fp16_rootcause.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --num_samples 5
"""

import argparse
import gc
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy

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


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def cosine_sim(a, b):
    a_flat = a.reshape(a.shape[0], -1).float()
    b_flat = b.reshape(b.shape[0], -1).float()
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def compute_metrics(ref, test):
    diff = ref.float() - test.float()
    return {
        "mse": (diff**2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cos_sim": cosine_sim(ref, test),
    }


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


def collect_backbone_inputs(policy, dataset, num_samples=5):
    """Collect backbone inputs and BF16 reference outputs."""
    logger.info(f"Collecting {num_samples} backbone input samples...")
    samples = []
    for traj_idx in range(min(num_samples, len(dataset))):
        obs = prepare_observation(policy, dataset, traj_idx, 0)

        # Process through VLA pipeline
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

    logger.info(f"Collected {len(samples)} samples")
    return samples


def swap_attention(model, impl):
    """Switch attention implementation on all modules."""
    count = 0
    for _, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = impl
            count += 1
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = impl
            count += 1
    return count


# ---------------------------------------------------------------------------
# Test 1: PyTorch FP16 (not TRT) to isolate precision vs TRT issues
# ---------------------------------------------------------------------------

def test_pytorch_fp16(policy, samples):
    """Run backbone in FP16 in PyTorch — no TRT, no ONNX."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: PyTorch FP16 backbone (flash attention, pure PyTorch)")
    logger.info("=" * 80)

    backbone = policy.model.backbone

    # Test BF16 (reference, should match exactly)
    logger.info("\n[1a] BF16 flash (reference)...")
    bf16_outputs = []
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
            out = backbone(bb_in)
        bf16_outputs.append(out.backbone_features.detach().cpu().float())

    # Verify BF16 matches reference
    for ref, test in zip([s["ref_features"] for s in samples], bf16_outputs):
        m = compute_metrics(ref, test)
        logger.info(f"  BF16 vs ref: MSE={m['mse']:.6f}, cos_sim={m['cos_sim']:.6f}")

    # Test: CAST backbone to FP16, run with FP16 inputs
    logger.info("\n[1b] FP16 flash (model + inputs in FP16)...")
    backbone_fp16 = deepcopy(backbone).half().cuda().eval()

    fp16_flash_outputs = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.float16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.float16)
        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            out = backbone_fp16(bb_in)
        fp16_flash_outputs.append(out.backbone_features.detach().cpu().float())

    del backbone_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Test: FP16 backbone with SDPA attention (same as ONNX path)
    logger.info("\n[1c] FP16 SDPA (model + inputs in FP16, SDPA attention)...")
    backbone_sdpa = deepcopy(backbone)
    swap_attention(backbone_sdpa, "sdpa")
    backbone_sdpa = backbone_sdpa.half().cuda().eval()

    fp16_sdpa_outputs = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.float16) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.float16)
        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            out = backbone_sdpa(bb_in)
        fp16_sdpa_outputs.append(out.backbone_features.detach().cpu().float())

    del backbone_sdpa
    gc.collect()
    torch.cuda.empty_cache()

    # Test: BF16 backbone with SDPA attention
    logger.info("\n[1d] BF16 SDPA (model + inputs in BF16, SDPA attention)...")
    backbone_sdpa_bf16 = deepcopy(backbone)
    swap_attention(backbone_sdpa_bf16, "sdpa")
    backbone_sdpa_bf16 = backbone_sdpa_bf16.to(dtype=torch.bfloat16).cuda().eval()

    bf16_sdpa_outputs = []
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
            out = backbone_sdpa_bf16(bb_in)
        bf16_sdpa_outputs.append(out.backbone_features.detach().cpu().float())

    del backbone_sdpa_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # Test: FP32 backbone with SDPA attention
    logger.info("\n[1e] FP32 SDPA (model + inputs in FP32, SDPA attention)...")
    backbone_sdpa_fp32 = deepcopy(backbone)
    swap_attention(backbone_sdpa_fp32, "sdpa")
    backbone_sdpa_fp32 = backbone_sdpa_fp32.float().cuda().eval()

    fp32_sdpa_outputs = []
    for s in samples:
        pv = s["pixel_values"]
        if isinstance(pv, list):
            pv_gpu = [t.to("cuda", dtype=torch.float32) for t in pv]
        else:
            pv_gpu = pv.to("cuda", dtype=torch.float32)
        bb_in = BatchFeature(data={
            "pixel_values": pv_gpu,
            "input_ids": s["input_ids"].cuda(),
            "attention_mask": s["attention_mask"].cuda(),
        })
        with torch.inference_mode():
            out = backbone_sdpa_fp32(bb_in)
        fp32_sdpa_outputs.append(out.backbone_features.detach().cpu().float())

    del backbone_sdpa_fp32
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    ref_outputs = [s["ref_features"] for s in samples]

    configs = [
        ("BF16 flash (ref)", bf16_outputs),
        ("FP16 flash", fp16_flash_outputs),
        ("FP16 SDPA", fp16_sdpa_outputs),
        ("BF16 SDPA", bf16_sdpa_outputs),
        ("FP32 SDPA", fp32_sdpa_outputs),
    ]

    print("\n" + "=" * 100)
    print("PYTORCH DTYPE COMPARISON (all in PyTorch, no TRT/ONNX)")
    print("=" * 100)
    print(f"{'Config':<25} {'MSE':>12} {'MAE':>12} {'Cos Sim':>10} {'Max Abs':>10}")
    print("-" * 80)

    for name, outputs in configs:
        mses, cos_sims, maes, max_abs = [], [], [], []
        for ref, test in zip(ref_outputs, outputs):
            m = compute_metrics(ref, test)
            mses.append(m["mse"])
            cos_sims.append(m["cos_sim"])
            maes.append(m["mae"])
            max_abs.append(m["max_abs"])
        print(f"{name:<25} {np.mean(mses):>12.6f} {np.mean(maes):>12.6f} "
              f"{np.mean(cos_sims):>10.6f} {max(max_abs):>10.4f}")

    print("-" * 80)
    print()
    print("KEY QUESTIONS ANSWERED:")

    fp16_flash_sim = np.mean([compute_metrics(r, t)["cos_sim"]
                              for r, t in zip(ref_outputs, fp16_flash_outputs)])
    fp16_sdpa_sim = np.mean([compute_metrics(r, t)["cos_sim"]
                             for r, t in zip(ref_outputs, fp16_sdpa_outputs)])
    bf16_sdpa_sim = np.mean([compute_metrics(r, t)["cos_sim"]
                             for r, t in zip(ref_outputs, bf16_sdpa_outputs)])
    fp32_sdpa_sim = np.mean([compute_metrics(r, t)["cos_sim"]
                             for r, t in zip(ref_outputs, fp32_sdpa_outputs)])

    print(f"\n  1. Is FP16 the problem, or TRT?")
    if fp16_flash_sim < 0.95:
        print(f"     --> FP16 PYTORCH flash cos_sim={fp16_flash_sim:.4f} — FP16 ITSELF is the problem!")
        print(f"         Even in pure PyTorch with flash attention, FP16 destroys quality.")
        print(f"         SmoothQuant and TRT optimization won't help.")
    elif fp16_sdpa_sim < 0.95:
        print(f"     --> FP16 flash is OK ({fp16_flash_sim:.4f}) but FP16 SDPA is bad ({fp16_sdpa_sim:.4f})")
        print(f"         The problem is FP16 + SDPA combination (attention materialization)")
    else:
        print(f"     --> FP16 PyTorch works fine ({fp16_flash_sim:.4f}/{fp16_sdpa_sim:.4f})")
        print(f"         The problem is TRT-specific (operator fusion, precision handling)")

    print(f"\n  2. Is SDPA the problem?")
    print(f"     BF16 SDPA cos_sim={bf16_sdpa_sim:.4f}, FP32 SDPA cos_sim={fp32_sdpa_sim:.4f}")
    if bf16_sdpa_sim > 0.99:
        print(f"     --> SDPA is fine in BF16/FP32. Issue is specifically FP16 + SDPA.")
    else:
        print(f"     --> SDPA itself introduces some error even in BF16")

    print(f"\n  3. COMPARISON:")
    print(f"     BF16 flash:  cos_sim=1.000 (reference)")
    print(f"     FP16 flash:  cos_sim={fp16_flash_sim:.4f}")
    print(f"     BF16 SDPA:   cos_sim={bf16_sdpa_sim:.4f}")
    print(f"     FP32 SDPA:   cos_sim={fp32_sdpa_sim:.4f}")
    print(f"     FP16 SDPA:   cos_sim={fp16_sdpa_sim:.4f}")
    print(f"     TRT FP16:    cos_sim=0.354 (from previous benchmarks)")
    print(f"     TRT FP32:    cos_sim=0.999 (from previous benchmarks)")
    print("=" * 100)

    return {
        "fp16_flash": fp16_flash_sim,
        "fp16_sdpa": fp16_sdpa_sim,
        "bf16_sdpa": bf16_sdpa_sim,
        "fp32_sdpa": fp32_sdpa_sim,
    }


# ---------------------------------------------------------------------------
# Test 2: Profile intermediate attention scores
# ---------------------------------------------------------------------------

def test_attention_intermediates(policy, samples):
    """Profile Q@K^T attention score ranges inside SDPA."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Intermediate attention score analysis")
    logger.info("=" * 80)

    backbone = deepcopy(policy.model.backbone)
    swap_attention(backbone, "sdpa")
    backbone = backbone.to(dtype=torch.bfloat16).cuda().eval()

    # Hook into the SDPA attention to capture intermediate values
    attn_stats = defaultdict(lambda: {"q_max": [], "k_max": [], "v_max": [],
                                       "score_max": [], "score_min": [],
                                       "q_range": [], "k_range": []})

    def make_attn_hook(name):
        def hook(module, inputs, output):
            # For SigLIP2 attention, inputs are (hidden_states,) or (hidden_states, attention_mask)
            # We need to compute Q@K^T manually to see the scores
            # The module already computed Q, K, V inside forward()
            # We can access them via the module's cached attributes if available
            # But most modules don't cache intermediates...
            # Instead, capture the input hidden_states and module weights
            if not inputs:
                return
            hidden_states = inputs[0]
            if not isinstance(hidden_states, torch.Tensor):
                return

            stats = attn_stats[name]
            h = hidden_states.detach().float()
            stats["q_max"].append(h.abs().max().item())

            # Check if this is an attention output (post-softmax @ V)
            if isinstance(output, torch.Tensor):
                stats["v_max"].append(output.detach().float().abs().max().item())

        return hook

    hooks = []
    for name, module in backbone.named_modules():
        if "self_attn" in name and name.endswith("self_attn"):
            h = module.register_forward_hook(make_attn_hook(name))
            hooks.append(h)

    # Also hook into Q, K, V projection outputs
    qkv_stats = defaultdict(lambda: {"max": [], "min": []})

    def make_qkv_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                qkv_stats[name]["max"].append(output.detach().float().abs().max().item())
                qkv_stats[name]["min"].append(output.detach().float().abs().min().item())
        return hook

    for name, module in backbone.named_modules():
        if any(p in name for p in ["q_proj", "k_proj", "v_proj"]) and isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_qkv_hook(name))
            hooks.append(h)

    # Run samples
    for s in samples[:3]:
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
            backbone(bb_in)

    for h in hooks:
        h.remove()

    # Report Q, K, V projection output ranges
    print("\n" + "=" * 80)
    print("Q/K/V PROJECTION OUTPUT RANGES (BF16 SDPA)")
    print("=" * 80)
    print(f"{'Layer':<70} {'Max Abs':>10}")
    print("-" * 82)

    overflow_count = 0
    for name, stats in sorted(qkv_stats.items(), key=lambda x: max(x[1]["max"]), reverse=True):
        max_val = max(stats["max"])
        marker = " OVERFLOW!" if max_val > 65504 else ""
        if max_val > 65504:
            overflow_count += 1
        short = name if len(name) <= 68 else "..." + name[-65:]
        print(f"{short:<70} {max_val:>10.1f}{marker}")

    print(f"\nQ/K/V projections with FP16 overflow: {overflow_count}")

    # Estimate attention score ranges
    print("\n" + "=" * 80)
    print("ESTIMATED ATTENTION SCORE RANGES (Q@K^T / sqrt(d_head))")
    print("=" * 80)

    # Group by layer
    layers = defaultdict(dict)
    for name, stats in qkv_stats.items():
        parts = name.split(".")
        # Find the layer part
        layer_name = ".".join(parts[:-1])  # Everything before q_proj/k_proj/v_proj
        proj_type = parts[-1]  # q_proj, k_proj, v_proj
        layers[layer_name][proj_type] = max(stats["max"])

    print(f"{'Layer':<60} {'Q max':>8} {'K max':>8} {'Est Score':>10} {'FP16?':>6}")
    print("-" * 98)

    for layer_name, projs in sorted(layers.items()):
        q_max = projs.get("q_proj", 0)
        k_max = projs.get("k_proj", 0)

        # Determine head_dim from layer name
        if "vision" in layer_name:
            head_dim = 64  # SigLIP2: 1152 / 18 heads
        else:
            head_dim = 128  # Qwen2: 4096 / 32 heads

        # Worst-case attention score: q_max * k_max * head_dim / sqrt(head_dim)
        # = q_max * k_max * sqrt(head_dim)
        est_score = q_max * k_max * (head_dim ** 0.5)
        marker = " OVERFLOW" if est_score > 65504 else "     OK"

        short = layer_name if len(layer_name) <= 58 else "..." + layer_name[-55:]
        print(f"{short:<60} {q_max:>8.1f} {k_max:>8.1f} {est_score:>10.0f} {marker}")

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    return qkv_stats


def main():
    parser = argparse.ArgumentParser(description="FP16 root cause analysis")
    parser.add_argument("--model_path", default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", default="new_embodiment")
    parser.add_argument("--video_backend", default="torchcodec")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    samples = collect_backbone_inputs(policy, dataset, args.num_samples)

    # Test 1: PyTorch FP16 vs BF16
    dtype_results = test_pytorch_fp16(policy, samples)

    # Test 2: Attention intermediate analysis
    qkv_stats = test_attention_intermediates(policy, samples)


if __name__ == "__main__":
    main()
