#!/usr/bin/env python3
"""
Phase 1 experiment: torch.quantize_dynamic on the GR00T backbone.

Tests whether INT8 weight-only quantization gives free latency on the
PyTorch path. quantize_dynamic replaces nn.Linear weight tensors with INT8
storage — at inference, weights are dequantized to compute dtype on-the-fly.
This halves DRAM bandwidth for weight loads (relevant at batch=1).

Known risks:
  - quantize_dynamic may not compose with flash_attention_2 CUDA kernels
  - qnnpack (ARM) targets CPU; GPU path uses fbgemm or falls back to dequant→matmul
  - On GPU, dequantization overhead might eat bandwidth savings (zero net speedup)
  - May error out entirely (quantized tensors on GPU not supported for all ops)

Go/No-Go:
  - Go:    latency < 145ms AND cos_sim > 0.99
  - No-Go: errors on CUDA, or latency >= 155ms, or cos_sim < 0.99

Usage (host, not Docker — no TRT needed):
    python scripts/deployment/test_dynamic_quant_backbone.py \
        --model_path alfie-gr00t/checkpoint-10000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment
"""

import argparse
import copy
import gc
import os
import time

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.6",
)

import numpy as np
import torch
import torch.quantization

torch.set_float32_matmul_precision("high")

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy


def _rec_to_dtype(x, dtype):
    """Recursively convert all floating point tensors to the given dtype."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def prepare_model_inputs(policy, observation):
    """Prepare inputs for the model (observation dict -> collated model inputs)."""
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


def get_backbone_inputs(policy, collated_inputs):
    """Run prepare_input to get backbone-ready tensors."""
    with torch.inference_mode():
        backbone_inputs, _ = policy.model.prepare_input(collated_inputs)
    return backbone_inputs


def benchmark_backbone(backbone, backbone_inputs, num_iterations=20, warmup=5, label=""):
    """Benchmark backbone forward pass with proper GPU sync barriers."""
    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = backbone(backbone_inputs)
    torch.cuda.synchronize()
    gc.collect()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = backbone(backbone_inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    print(f"\n  {label} Backbone Latency ({num_iterations} iters):")
    print(f"    Median: {np.median(times):.1f} ms")
    print(f"    Mean:   {np.mean(times):.1f} +/- {np.std(times):.1f} ms")
    print(f"    Min:    {np.min(times):.1f} ms")
    print(f"    Max:    {np.max(times):.1f} ms")
    print(f"    P90:    {np.percentile(times, 90):.1f} ms")
    return times


def compare_accuracy(backbone_ref, backbone_test, backbone_inputs, label=""):
    """Compare backbone outputs: MSE, MAE, cosine similarity, max abs error."""
    with torch.inference_mode():
        out_ref = backbone_ref(backbone_inputs)
        out_test = backbone_test(backbone_inputs)

    feat_ref = out_ref["backbone_features"].float()
    feat_test = out_test["backbone_features"].float()

    mse = torch.mean((feat_ref - feat_test) ** 2).item()
    mae = torch.mean(torch.abs(feat_ref - feat_test)).item()
    max_abs = torch.max(torch.abs(feat_ref - feat_test)).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        feat_ref.flatten(0, 1), feat_test.flatten(0, 1)
    ).mean().item()

    print(f"\n  {label} Accuracy vs Flash Baseline:")
    print(f"    MSE:       {mse:.6f}")
    print(f"    MAE:       {mae:.6f}")
    print(f"    Max Abs:   {max_abs:.6f}")
    print(f"    Cos Sim:   {cos_sim:.6f}")

    return {"mse": mse, "mae": mae, "max_abs": max_abs, "cos_sim": cos_sim}


def count_linear_layers(module):
    """Count nn.Linear layers in a module."""
    count = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            count += 1
    return count


def count_quantized_layers(module):
    """Count quantized Linear layers in a module."""
    count = 0
    for m in module.modules():
        cls_name = type(m).__name__
        if "Quantized" in cls_name or "Dynamic" in cls_name:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Test dynamic quantization on GR00T backbone")
    parser.add_argument("--model_path", type=str, default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", type=str, default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("PHASE 1: Dynamic Quantization Backbone Experiment")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model:  {args.model_path}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ========================================
    # Step 1: Load policy
    # ========================================
    print("Loading policy (flash_attention_2)...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device=device,
        strict=True,
    )
    print(f"  Backbone type: {type(policy.model.backbone).__name__}")
    backbone_dtype = next(iter(policy.model.backbone.parameters())).dtype
    print(f"  Backbone dtype: {backbone_dtype}")
    num_linear = count_linear_layers(policy.model.backbone)
    print(f"  nn.Linear layers: {num_linear}")

    # ========================================
    # Step 2: Load dataset + prepare inputs
    # ========================================
    print("\nLoading dataset...")
    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    episode_data = dataset[0]
    step_data = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        allow_padding=False,
    )

    observation = {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
        "state": {k: step_data.states[k][None] for k in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }

    print("Preparing model inputs...")
    collated_inputs = prepare_model_inputs(policy, observation)
    backbone_inputs = get_backbone_inputs(policy, collated_inputs)

    # Print input shapes
    print("  Backbone input shapes:")
    for k, v in backbone_inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape} ({v.dtype})")

    # ========================================
    # Step 3: Baseline benchmark (flash attention BF16)
    # ========================================
    print("\n" + "-" * 60)
    print("BASELINE: PyTorch BF16 with flash_attention_2")
    print("-" * 60)

    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"  GPU memory allocated: {mem_before:.0f} MB")

    baseline_times = benchmark_backbone(
        policy.model.backbone, backbone_inputs,
        num_iterations=args.num_iterations, warmup=args.warmup,
        label="Baseline"
    )

    # Capture reference output for accuracy comparison
    with torch.inference_mode():
        ref_output = policy.model.backbone(backbone_inputs)
    ref_features = ref_output["backbone_features"].clone()
    print(f"  Output shape: {ref_features.shape}, dtype: {ref_features.dtype}")
    print(f"  Output range: [{ref_features.min().item():.4f}, {ref_features.max().item():.4f}]")

    # ========================================
    # Step 4: Dynamic quantization
    # ========================================
    print("\n" + "-" * 60)
    print("EXPERIMENT: torch.quantize_dynamic (INT8 weight-only)")
    print("-" * 60)

    # Copy backbone to CPU for quantization (quantize_dynamic is CPU-only)
    print("  Copying backbone to CPU for quantization...")
    backbone_cpu = copy.deepcopy(policy.model.backbone).cpu().float()

    print("  Applying torch.quantize_dynamic...")
    try:
        backbone_quantized = torch.quantization.quantize_dynamic(
            backbone_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as e:
        print(f"\n  FAILED: quantize_dynamic raised {type(e).__name__}: {e}")
        print("  Verdict: No-Go — dynamic quantization not compatible")
        return

    num_quantized = count_quantized_layers(backbone_quantized)
    print(f"  Quantized layers: {num_quantized} / {num_linear} nn.Linear")

    # Try moving quantized model to CUDA
    print("  Moving quantized backbone to CUDA...")
    try:
        backbone_quantized = backbone_quantized.cuda()
    except Exception as e:
        print(f"\n  FAILED to move to CUDA: {type(e).__name__}: {e}")
        print("  Quantized tensors on GPU not supported.")

        # Try CPU-only benchmark instead
        print("\n  Falling back to CPU benchmark...")
        backbone_inputs_cpu = {
            k: v.cpu().float() if isinstance(v, torch.Tensor) else v
            for k, v in backbone_inputs.items()
        }

        print("  Running CPU forward pass...")
        try:
            cpu_start = time.perf_counter()
            with torch.inference_mode():
                out_cpu = backbone_quantized(backbone_inputs_cpu)
            cpu_time = (time.perf_counter() - cpu_start) * 1000
            print(f"  CPU forward pass: {cpu_time:.0f} ms (not useful for Orin deployment)")

            # Accuracy on CPU
            feat_q = out_cpu["backbone_features"].float()
            feat_r = ref_features.cpu().float()
            cos_sim = torch.nn.functional.cosine_similarity(
                feat_r.flatten(0, 1), feat_q.flatten(0, 1)
            ).mean().item()
            mse = torch.mean((feat_r - feat_q) ** 2).item()
            print(f"  CPU accuracy — MSE: {mse:.6f}, cos_sim: {cos_sim:.6f}")
        except Exception as e2:
            print(f"  CPU forward pass also failed: {type(e2).__name__}: {e2}")

        print("\n  Verdict: No-Go — quantized tensors cannot run on CUDA")
        return

    mem_after_quant = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"  GPU memory allocated: {mem_after_quant:.0f} MB (was {mem_before:.0f} MB)")

    # ========================================
    # Step 5: Benchmark quantized backbone on CUDA
    # ========================================
    print("\n  Running quantized backbone forward pass...")
    try:
        with torch.inference_mode():
            test_out = backbone_quantized(backbone_inputs)
        print("  Forward pass succeeded!")
    except Exception as e:
        print(f"\n  FAILED: forward pass raised {type(e).__name__}: {e}")
        print("  Verdict: No-Go — quantized backbone cannot run on CUDA")
        return

    quant_times = benchmark_backbone(
        backbone_quantized, backbone_inputs,
        num_iterations=args.num_iterations, warmup=args.warmup,
        label="Quantized"
    )

    # ========================================
    # Step 6: Accuracy comparison
    # ========================================
    accuracy = compare_accuracy(
        policy.model.backbone, backbone_quantized, backbone_inputs,
        label="Quantized"
    )

    # ========================================
    # Step 7: Summary
    # ========================================
    baseline_median = np.median(baseline_times)
    quant_median = np.median(quant_times)
    speedup = baseline_median / quant_median

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print("| Config | Median ms | P90 ms | cos_sim | MSE | Memory MB |")
    print("|--------|-----------|--------|---------|-----|-----------|")
    print(
        f"| BF16 flash (baseline) | {baseline_median:.1f} | "
        f"{np.percentile(baseline_times, 90):.1f} | 1.000 | 0.000 | {mem_before:.0f} |"
    )
    print(
        f"| INT8 dynamic quant | {quant_median:.1f} | "
        f"{np.percentile(quant_times, 90):.1f} | {accuracy['cos_sim']:.3f} | "
        f"{accuracy['mse']:.6f} | {mem_after_quant:.0f} |"
    )
    print()
    print(f"Speedup: {speedup:.2f}x ({baseline_median:.1f} -> {quant_median:.1f} ms)")
    print()

    # Go/No-Go decision
    go = quant_median < 145 and accuracy["cos_sim"] > 0.99
    if go:
        print("VERDICT: GO — integrate into standalone_inference_script.py")
    elif accuracy["cos_sim"] < 0.99:
        print(f"VERDICT: No-Go — cos_sim {accuracy['cos_sim']:.4f} < 0.99 threshold")
    elif quant_median >= 155:
        print(f"VERDICT: No-Go — latency {quant_median:.1f}ms >= 155ms threshold")
    else:
        print(f"VERDICT: Marginal — latency {quant_median:.1f}ms, cos_sim {accuracy['cos_sim']:.4f}")
        print("  Between go (< 145ms) and no-go (>= 155ms). Consider E2E testing.")


if __name__ == "__main__":
    main()
