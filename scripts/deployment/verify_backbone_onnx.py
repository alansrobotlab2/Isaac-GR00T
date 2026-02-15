#!/usr/bin/env python3
"""
Verify an exported ONNX backbone by comparing its outputs to the original PyTorch model.

Loads a sample observation, runs it through both the PyTorch backbone and the ONNX model
(via onnxruntime), and reports numerical differences. This confirms the ONNX export
preserves model behavior.

Usage:
    python scripts/deployment/verify_backbone_onnx.py \
        --model_path cando/checkpoint-2000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --onnx_path ./groot_n1d6_onnx/backbone_model.onnx \
        --attn_implementation eager
"""

import argparse
import logging
import os
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def prepare_observation(policy, dataset, traj_idx=0):
    """Prepare a single observation for inference."""
    traj = dataset[traj_idx]
    modality_configs = policy.get_modality_config()

    data_point = extract_step_data(
        traj,
        0,
        modality_configs=modality_configs,
        embodiment_tag=policy.embodiment_tag,
    )

    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in modality_configs["language"].modality_keys:
        observation[key] = data_point.text

    return parse_observation_gr00t(observation, modality_configs)


class BackboneIOCapture:
    """Captures both inputs and outputs of the backbone forward pass."""

    def __init__(self):
        self.captured = False
        self.input_ids = None
        self.attention_mask = None
        self.pixel_values = None
        self.output_hidden_states = None
        self.output_attn_mask = None
        self.output_image_mask = None

    def pre_hook(self, module, args, kwargs):
        if self.captured:
            return
        vl_input = args[0] if args else kwargs.get("vl_input")
        self.input_ids = vl_input["input_ids"].detach().cpu().clone()
        self.attention_mask = vl_input["attention_mask"].detach().cpu().clone()
        pv = vl_input["pixel_values"]
        if isinstance(pv, list):
            self.pixel_values = [t.detach().cpu().clone() for t in pv]
        else:
            self.pixel_values = pv.detach().cpu().clone()

    def post_hook(self, module, args, output):
        if self.captured:
            return
        # EagleBackbone.forward returns BatchFeature with these keys
        self.output_hidden_states = output["backbone_features"].detach().cpu().clone()
        self.output_attn_mask = output["backbone_attention_mask"].detach().cpu().clone()
        self.output_image_mask = output["image_mask"].detach().cpu().clone()
        self.captured = True


def run_pytorch_reference(backbone, capture, model_dtype, device="cuda"):
    """
    Re-run the backbone in the same way the ONNX wrapper does, cast to the export dtype.

    This ensures we compare apples-to-apples: the same computation graph, same dtype.
    The ONNX wrapper calls backbone.model(...) directly (not EagleBackbone.forward),
    so we do the same here.
    """
    backbone.eval()
    backbone.to(device=device, dtype=model_dtype)
    eagle_model = backbone.model
    image_token_index = eagle_model.config.image_token_index

    input_ids = capture.input_ids.to(device)
    attention_mask = capture.attention_mask.to(device)

    if isinstance(capture.pixel_values, list):
        pixel_values_list = [t.to(device=device, dtype=model_dtype) for t in capture.pixel_values]
    else:
        pixel_values_list = [capture.pixel_values[i].to(device=device, dtype=model_dtype)
                             for i in range(capture.pixel_values.shape[0])]

    with torch.inference_mode():
        outputs = eagle_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_list,
            output_hidden_states=True,
        )

    hidden_states = outputs["hidden_states"][-1]
    attn_mask = attention_mask == 1
    image_mask = input_ids == image_token_index

    return (
        hidden_states.cpu().float().numpy(),
        attn_mask.cpu().numpy(),
        image_mask.cpu().numpy(),
    )


def run_onnx_model(onnx_path, capture, model_dtype, force_cpu=False):
    """
    Run the ONNX model via onnxruntime and return outputs as numpy arrays.

    Returns (hidden_states, attn_mask, image_mask, used_cuda) where used_cuda
    indicates whether the CUDA provider was actually used (affects comparison dtype).
    """
    if force_cpu:
        providers_to_try = ["CPUExecutionProvider"]
    else:
        providers_to_try = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    available = ort.get_available_providers()

    logger.info(f"Loading ONNX model from {onnx_path}")
    logger.info(f"onnxruntime available providers: {available}")
    logger.info(f"Requested providers: {providers_to_try}")

    session = ort.InferenceSession(onnx_path, providers=providers_to_try)
    active_providers = session.get_providers()
    used_cuda = "CUDAExecutionProvider" in active_providers

    logger.info(f"Active providers: {active_providers}")
    if not used_cuda:
        logger.info("ONNX running on CPU.")

    # Prepare inputs — must match the ONNX wrapper's input signature
    input_ids = capture.input_ids.numpy()
    attention_mask = capture.attention_mask.numpy()

    # Stack pixel_values into [num_frames, C, H, W] tensor
    pv = capture.pixel_values
    if isinstance(pv, list):
        pv_stacked = torch.stack(pv)
    else:
        pv_stacked = pv

    # When running on CPU, onnxruntime casts FP16 inputs to FP32 internally.
    # Feed FP16 to match the ONNX graph's declared input dtype regardless.
    if model_dtype == torch.float16:
        pv_np = pv_stacked.to(torch.float16).numpy()
    else:
        pv_np = pv_stacked.to(torch.float32).numpy()

    logger.info("ONNX input shapes:")
    logger.info(f"  input_ids: {input_ids.shape} ({input_ids.dtype})")
    logger.info(f"  attention_mask: {attention_mask.shape} ({attention_mask.dtype})")
    logger.info(f"  pixel_values: {pv_np.shape} ({pv_np.dtype})")

    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pv_np,
    }

    hidden_states, attn_mask, image_mask = session.run(None, onnx_inputs)
    return hidden_states, attn_mask, image_mask, used_cuda


def compare_outputs(name, pytorch_out, onnx_out, is_float=True, atol=1e-2, cosine_thresh=0.999):
    """Compare a single output tensor pair and print metrics."""
    print(f"\n  --- {name} ---")
    print(f"  PyTorch shape: {pytorch_out.shape}  dtype: {pytorch_out.dtype}")
    print(f"  ONNX    shape: {onnx_out.shape}  dtype: {onnx_out.dtype}")

    if pytorch_out.shape != onnx_out.shape:
        print(f"  SHAPE MISMATCH!")
        return False

    if is_float:
        # Cast both to float32 for comparison
        pt = pytorch_out.astype(np.float32)
        ox = onnx_out.astype(np.float32)

        abs_diff = np.abs(pt - ox)
        max_err = abs_diff.max()
        mean_err = abs_diff.mean()

        # Relative error (element-wise, ignoring near-zero values)
        scale = np.maximum(np.abs(pt), np.abs(ox))
        rel_mask = scale > 1e-6
        rel_err = np.zeros_like(abs_diff)
        rel_err[rel_mask] = abs_diff[rel_mask] / scale[rel_mask]
        mean_rel_err = rel_err[rel_mask].mean() if rel_mask.any() else 0.0
        max_rel_err = rel_err[rel_mask].max() if rel_mask.any() else 0.0

        # Cosine similarity (flatten)
        pt_flat = pt.flatten()
        ox_flat = ox.flatten()
        dot = np.dot(pt_flat, ox_flat)
        norm_pt = np.linalg.norm(pt_flat)
        norm_ox = np.linalg.norm(ox_flat)
        cosine_sim = dot / (norm_pt * norm_ox + 1e-12)

        # Percentile breakdown of absolute errors
        p50 = np.percentile(abs_diff, 50)
        p90 = np.percentile(abs_diff, 90)
        p99 = np.percentile(abs_diff, 99)
        p999 = np.percentile(abs_diff, 99.9)

        print(f"  Max absolute error:  {max_err:.6e}")
        print(f"  Mean absolute error: {mean_err:.6e}")
        print(f"  Max relative error:  {max_rel_err:.6e}")
        print(f"  Mean relative error: {mean_rel_err:.6e}")
        print(f"  Percentiles (abs):   p50={p50:.6e}  p90={p90:.6e}  p99={p99:.6e}  p99.9={p999:.6e}")
        print(f"  Cosine similarity:   {cosine_sim:.8f}")
        print(f"  Any NaN (PyTorch):   {np.any(np.isnan(pt))}")
        print(f"  Any NaN (ONNX):      {np.any(np.isnan(ox))}")

        has_nan = np.any(np.isnan(ox))
        cosine_pass = cosine_sim >= cosine_thresh
        p99_pass = p99 < atol

        # Pass criteria: cosine similarity above threshold, p99 within tolerance,
        # and no NaNs. A few FP16 outliers in the tail are expected when comparing
        # CPU vs GPU execution, so we use p99 rather than max.
        passed = cosine_pass and p99_pass and not has_nan
        print(f"  PASS (cosine>={cosine_thresh}, p99<{atol}, no NaN): {passed}")
        if not cosine_pass:
            print(f"    FAIL: cosine similarity {cosine_sim:.6f} < {cosine_thresh}")
        if not p99_pass:
            print(f"    FAIL: p99 error {p99:.6e} >= atol {atol}")
        return passed
    else:
        # Boolean/integer comparison — must be exact
        match = np.array_equal(pytorch_out, onnx_out)
        if not match:
            mismatches = np.sum(pytorch_out != onnx_out)
            total = pytorch_out.size
            print(f"  MISMATCH: {mismatches}/{total} elements differ")
        else:
            print(f"  Exact match: True")
        return match


def main(args):
    print("=" * 80)
    print("ONNX Backbone Verification")
    print("=" * 80)

    if not os.path.exists(args.onnx_path):
        logger.error(f"ONNX file not found: {args.onnx_path}")
        return

    # Resolve export dtype
    if args.export_dtype == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    # Step 1: Load policy with eager attention (must match export)
    logger.info("[Step 1] Loading PyTorch model...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
        attn_implementation=args.attn_implementation,
    )

    # Convert backbone to export dtype for fair comparison
    backbone = policy.model.backbone
    if policy.model_dtype != model_dtype:
        logger.info(f"Converting backbone from {policy.model_dtype} to {model_dtype}...")
        backbone.to(dtype=model_dtype)

    # Step 2: Load dataset and prepare observation
    logger.info("[Step 2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    observation = prepare_observation(policy, dataset, traj_idx=0)

    # Step 3: Capture backbone I/O from a real inference pass
    logger.info("[Step 3] Running PyTorch inference to capture backbone inputs...")
    capture = BackboneIOCapture()
    pre_handle = backbone.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post_handle = backbone.register_forward_hook(capture.post_hook)

    with torch.inference_mode():
        _ = policy.get_action(observation)

    pre_handle.remove()
    post_handle.remove()

    if not capture.captured:
        logger.error("Failed to capture backbone I/O!")
        return

    logger.info("Captured backbone inputs and outputs")

    # Step 4: Run ONNX model
    force_cpu = args.device == "cpu"
    logger.info(f"[Step 4] Running ONNX model (device={args.device})...")
    ox_hidden, ox_attn, ox_image, onnx_used_cuda = run_onnx_model(
        args.onnx_path, capture, model_dtype, force_cpu=force_cpu,
    )

    # Step 5: Run PyTorch reference.
    # When --device cpu, both ONNX and PyTorch run on CPU. The ONNX graph has
    # FP16 weights, so onnxruntime on CPU computes with those FP16 values
    # (internally upcasting ops to FP32 but keeping FP16 intermediates).
    # For PyTorch on CPU, FP16 matmul is extremely slow and some ops silently
    # upcast to FP32, so we run PyTorch in FP32 on CPU. The comparison then
    # isolates ONNX graph correctness from precision noise — any error beyond
    # FP16 quantization noise (~1e-3) indicates a graph conversion issue.
    onnx_out_dtype = ox_hidden.dtype
    logger.info(f"ONNX output dtype: {onnx_out_dtype}")

    if force_cpu:
        ref_device = "cpu"
        ref_dtype = torch.float32
    elif onnx_used_cuda:
        ref_device = "cuda"
        ref_dtype = model_dtype
    else:
        ref_device = "cuda"
        ref_dtype = model_dtype

    logger.info(f"[Step 5] Running PyTorch reference on {ref_device} in {ref_dtype}...")

    pt_hidden, pt_attn, pt_image = run_pytorch_reference(
        backbone, capture, ref_dtype, device=ref_device
    )

    # Step 6: Compare
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    if force_cpu:
        print("  MODE: Both running on CPU. PyTorch=FP32, ONNX=FP16 weights.")
        print("  This isolates ONNX graph correctness from device-specific numerics.")
        print("  Expect small errors from FP16 weight quantization only.")
    elif not onnx_used_cuda:
        print("  NOTE: ONNX fell back to CPU, PyTorch ran on CUDA. Both used FP16 precision.")
        print("  Differences come from ONNX graph optimizations and CPU vs GPU numerics.")
        print("  Use --device cpu for a cleaner comparison, or install onnxruntime-gpu with CUDA 12.")

    results = []
    results.append(compare_outputs(
        "hidden_states", pt_hidden, ox_hidden,
        is_float=True, atol=args.atol, cosine_thresh=args.cosine_thresh,
    ))
    results.append(compare_outputs("attn_mask", pt_attn, ox_attn, is_float=False))
    results.append(compare_outputs("image_mask", pt_image, ox_image, is_float=False))

    # Per-token-type error breakdown to diagnose where errors concentrate
    print("\n  --- Per-token-type error breakdown ---")
    pt_h = pt_hidden.astype(np.float32)
    ox_h = ox_hidden.astype(np.float32)
    abs_diff = np.abs(pt_h - ox_h)  # (B, seq_len, hidden)

    img_mask = pt_image[0]   # (seq_len,) bool
    txt_mask = ~img_mask & pt_attn[0]  # text tokens (non-image, non-padding)
    pad_mask = ~pt_attn[0]   # padding tokens

    for label, mask in [("image", img_mask), ("text", txt_mask), ("padding", pad_mask)]:
        n_tokens = mask.sum()
        if n_tokens == 0:
            print(f"  {label:>8s}: 0 tokens")
            continue
        token_errs = abs_diff[0, mask, :]  # (n_tokens, hidden)
        print(f"  {label:>8s}: {n_tokens} tokens, "
              f"mean_abs={token_errs.mean():.6e}, "
              f"max_abs={token_errs.max():.6e}, "
              f"p99_abs={np.percentile(token_errs, 99):.6e}")

    print("\n" + "=" * 80)
    if all(results):
        print("OVERALL: PASS — ONNX export matches PyTorch model")
    else:
        print("OVERALL: FAIL — ONNX export does NOT match PyTorch model")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX backbone export against PyTorch")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument(
        "--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.GR1, help="Embodiment tag"
    )
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to exported backbone ONNX model")
    parser.add_argument(
        "--attn_implementation", type=str, default="eager",
        help="Attention implementation (must match what was used during export)",
    )
    parser.add_argument(
        "--video_backend", type=str, default="torchcodec",
        help="Options: ['decord', 'torchvision_av', 'torchcodec']",
    )
    parser.add_argument(
        "--export_dtype", type=str, default="fp16", choices=["fp16", "fp32"],
        help="Export dtype (must match what was used during export)",
    )
    parser.add_argument(
        "--atol", type=float, default=5e-1,
        help="Absolute tolerance for p99 error (default: 0.5 for FP16 CPU vs GPU comparison)",
    )
    parser.add_argument(
        "--cosine_thresh", type=float, default=0.999,
        help="Minimum cosine similarity threshold (default: 0.999)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for comparison. 'cpu' forces both ONNX and PyTorch to run on CPU "
             "(isolates graph correctness from device numerics). 'auto' uses CUDA if available.",
    )

    args = parser.parse_args()
    main(args)
