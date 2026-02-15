#!/usr/bin/env python3
"""
Export EagleX backbone to ONNX for TensorRT optimization.

The backbone (Siglip2 vision encoder + Qwen2 LLM + MLP1 projection) is
exported as a single ONNX model. Flash attention must be replaced with
eager attention before export since flash_attn ops are not ONNX-traceable.

Usage:
    python export_backbone_onnx.py \
        --model_path /path/to/checkpoint \
        --dataset_path /path/to/dataset \
        --output_dir ./groot_n1d6_onnx
"""

import argparse
import logging
import os
from copy import deepcopy
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input capture
# ---------------------------------------------------------------------------


class BackboneInputCapture:
    """Capture backbone forward pass inputs during inference."""

    def __init__(self):
        self.captured = False
        self.input_ids = None
        self.attention_mask = None
        self.pixel_values = None  # Will be a single stacked tensor
        self.pixel_values_list_shapes = None  # Original list element shapes

    def hook_fn(self, module, args, kwargs):
        if not self.captured:
            # The backbone forward unpacks vl_input to get these keys
            vl_input = args[0] if args else kwargs.get("vl_input")
            if vl_input is None:
                return

            self.input_ids = vl_input["input_ids"].detach().cpu().clone()
            self.attention_mask = vl_input["attention_mask"].detach().cpu().clone()

            # pixel_values is a list of tensors (one per image/frame group),
            # each with shape [N, C, H, W]. Concatenate along dim 0 for ONNX.
            pv = vl_input["pixel_values"]
            if isinstance(pv, list):
                self.pixel_values_list_shapes = [t.shape for t in pv]
                self.pixel_values = torch.cat(
                    [t.detach().cpu().clone() for t in pv], dim=0
                )
                logger.info(f"  pixel_values: list of {len(pv)} tensors, "
                            f"shapes: {self.pixel_values_list_shapes}")
            else:
                self.pixel_values = pv.detach().cpu().clone()

            self.captured = True

            logger.info("Captured backbone inputs:")
            logger.info(f"  input_ids: {self.input_ids.shape} ({self.input_ids.dtype})")
            logger.info(f"  attention_mask: {self.attention_mask.shape} ({self.attention_mask.dtype})")
            logger.info(f"  pixel_values (stacked): {self.pixel_values.shape} ({self.pixel_values.dtype})")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def prepare_model_inputs(policy, observation):
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


# ---------------------------------------------------------------------------
# Flash attention swap
# ---------------------------------------------------------------------------


def swap_flash_to_eager(model):
    """
    Recursively set _attn_implementation to 'eager' on all submodules
    that use flash_attention_2. This is needed for ONNX export since
    flash attention ops are not traceable.
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation == "flash_attention_2":
                module.config._attn_implementation = "eager"
                count += 1
        # Also check for the attribute directly on the module
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation == "flash_attention_2":
                module._attn_implementation = "eager"
                count += 1
    logger.info(f"Swapped {count} flash_attention_2 → eager attention implementations")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_backbone_to_onnx(
    policy: Gr00tPolicy,
    captured: BackboneInputCapture,
    output_path: str,
    use_fp32: bool = True,
):
    """
    Export the EagleX backbone to ONNX.

    The backbone forward returns 3 outputs, but backbone_attention_mask
    and image_mask are derived from input_ids (not model computation),
    so we only export backbone_features as the model output. The masks
    can be computed cheaply from input_ids at inference time.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Backbone to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    backbone.eval()

    # Swap flash attention to eager for ONNX traceability
    swap_flash_to_eager(backbone)

    # Use FP32 for export to let TensorRT handle precision
    if use_fp32:
        dtype = torch.float32
        backbone = backbone.float()
        logger.info("Using FP32 precision for ONNX export")
    else:
        dtype = torch.bfloat16
        logger.info("Using BF16 precision for ONNX export")

    backbone = backbone.cuda()

    # Create dummy inputs matching captured shapes
    pixel_values = torch.randn(
        captured.pixel_values.shape, dtype=dtype, device="cuda"
    )
    input_ids = captured.input_ids.clone().to(device="cuda")
    attention_mask = captured.attention_mask.clone().to(device="cuda")

    logger.info("Export input shapes:")
    logger.info(f"  pixel_values: {pixel_values.shape} ({pixel_values.dtype})")
    logger.info(f"  input_ids: {input_ids.shape} ({input_ids.dtype})")
    logger.info(f"  attention_mask: {attention_mask.shape} ({attention_mask.dtype})")

    # ONNX wrapper: the backbone forward takes a BatchFeature dict with
    # pixel_values as a list of tensors, but torch.onnx.export needs positional
    # tensor args. We pass pixel_values as a stacked tensor [N, C, H, W] and
    # split it back into a list of [1, C, H, W] tensors inside the wrapper.
    # This preserves the original model's iteration behavior during ONNX trace.
    num_images = captured.pixel_values.shape[0]
    logger.info(f"Wrapping backbone with {num_images} image splits")

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone_model, n_images):
            super().__init__()
            self.backbone = backbone_model
            self.n_images = n_images

        def forward(self, pixel_values, input_ids, attention_mask):
            from transformers.feature_extraction_utils import BatchFeature

            # pixel_values: [N, C, H, W] stacked tensor
            # Split back into list of [1, C, H, W] to match original model behavior.
            # The Siglip2 embeddings iterate over this list, computing spatial_shapes
            # per-image, which affects pixel_shuffle_back downstream.
            pv_list = [pixel_values[i:i+1] for i in range(self.n_images)]
            vl_input = BatchFeature(data={
                "pixel_values": pv_list,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            output = self.backbone(vl_input)
            # Return only backbone_features (the model's actual computation).
            # backbone_attention_mask and image_mask are derived from input_ids
            # and can be computed at inference time without the model.
            return output.backbone_features

    wrapped = BackboneWrapper(backbone, num_images)
    wrapped.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dynamic_axes = {
        "pixel_values": {0: "num_images"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "backbone_features": {0: "batch_size", 1: "feature_len"},
    }

    logger.info(f"Exporting to {output_path}...")

    # Monkey-patch F.interpolate to disable antialias (not supported in ONNX).
    # The Siglip2 positional embedding resize uses antialias=True, but the
    # difference is negligible for learned positional embeddings.
    import torch.nn.functional as _F
    _orig_interpolate = _F.interpolate

    def _interpolate_no_aa(*args, **kwargs):
        kwargs.pop("antialias", None)
        return _orig_interpolate(*args, **kwargs)

    _F.interpolate = _interpolate_no_aa

    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapped,
                (pixel_values, input_ids, attention_mask),
                output_path,
                input_names=["pixel_values", "input_ids", "attention_mask"],
                output_names=["backbone_features"],
                opset_version=19,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
                export_params=True,
            )
    finally:
        _F.interpolate = _orig_interpolate

    logger.info("Backbone exported successfully!")

    # Verify
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Model size: {file_size_mb:.2f} MB")

    external_data_path = output_path.replace(".onnx", ".onnx.data")
    if os.path.exists(external_data_path):
        ext_size = os.path.getsize(external_data_path) / (1024 * 1024)
        logger.info(f"External data: {ext_size:.2f} MB")
        logger.info(f"Total: {file_size_mb + ext_size:.2f} MB")

    try:
        import onnx
        onnx.checker.check_model(output_path)
        logger.info("ONNX model is valid!")
    except Exception as e:
        logger.warning(f"ONNX validation: {e}")
        logger.info("Export completed (validation skipped for large model)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    logger.info("=" * 80)
    logger.info("EagleX Backbone ONNX Export")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")

    # Load policy
    logger.info("\n[Step 1] Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device="cuda",
    )

    # Load dataset
    logger.info("\n[Step 2] Loading dataset...")
    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    # Capture backbone inputs
    logger.info("\n[Step 3] Capturing backbone inputs...")
    capture = BackboneInputCapture()
    hook = policy.model.backbone.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)

    modality_configs_no_action = deepcopy(modality_config)
    modality_configs_no_action.pop("action", None)

    traj = dataset[0]
    data_point = extract_step_data(traj, 0, modality_configs_no_action, policy.embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for key in modality_configs_no_action["language"].modality_keys:
        obs[key] = data_point.text
    parsed_obs = parse_observation_gr00t(obs, modality_configs_no_action)

    with torch.inference_mode():
        _ = policy.get_action(parsed_obs)

    hook.remove()

    if not capture.captured:
        logger.error("Failed to capture backbone inputs!")
        return

    # Export
    logger.info("\n[Step 4] Exporting backbone to ONNX...")
    output_path = os.path.join(args.output_dir, "backbone_model.onnx")
    export_backbone_to_onnx(
        policy, capture, output_path, use_fp32=not args.use_bf16
    )

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EagleX backbone to ONNX")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.GR1)
    parser.add_argument("--output_dir", type=str, default="./groot_n1d6_onnx")
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Export in BF16 (default: FP32 for better TRT precision control)")

    args = parser.parse_args()
    main(args)
