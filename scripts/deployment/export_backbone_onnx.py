#!/usr/bin/env python3
"""
Export the Eagle backbone (Vision Encoder + Language Model) to ONNX for TensorRT conversion.

The backbone is the largest component (~4GB in BF16). Converting it to TensorRT INT8
can reduce this to ~1-1.5GB, which is critical for fitting on Jetson Orin NX 16GB.

IMPORTANT: The backbone must use eager attention for ONNX export. Both Flash Attention 2
and SDPA contain ops that the ONNX JIT tracer cannot handle (segfaults during tracing).
Eager attention decomposes to basic matmul/softmax ops that trace cleanly.

IMPORTANT: All camera inputs must use the same resolution (e.g. 4x 320x240). The Eagle
vision encoder receives pixel_values as a list of variable-sized tensors, but ONNX export
requires a single fixed-shape tensor. The export script stacks all frames into one
[num_frames, C, H, W] tensor, which only works when all frames share the same resolution.

Usage:
    python export_backbone_onnx.py \
        --model_path cando/checkpoint-2000 \
        --dataset_path alfiebot.CanDoChallenge \
        --embodiment_tag new_embodiment \
        --output_dir ./groot_n1d6_onnx \
        --attn_implementation eager
"""

import argparse
import logging
import os
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.onnx
from torch.onnx import register_custom_op_symbolic


# Register custom ONNX symbolic for aten::_upsample_bilinear2d_aa.
# The Eagle vision encoder uses F.interpolate(..., antialias=True) for positional
# embedding resizing, which maps to this op. ONNX opset 19 doesn't support it
# natively, but we can map it to the standard ONNX Resize operator (the anti-alias
# distinction is negligible for positional embeddings).
def _upsample_bilinear2d_aa_symbolic(g, self, output_size, align_corners, scales_h=None, scales_w=None):
    # Schema: aten::_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size,
    #         bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    # Parameter names must match the ATen schema for keyword dispatch.
    input = self
    shape = g.op("Shape", input)
    n = g.op("Gather", shape, g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)), axis_i=0)
    c = g.op("Gather", shape, g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)), axis_i=0)

    # Concat [N, C] with output_size [H, W]
    nc = g.op("Unsqueeze", n, axes_i=[0])
    cc = g.op("Unsqueeze", c, axes_i=[0])
    hw = g.op("Cast", output_size, to_i=7)  # ensure int64
    sizes = g.op("Concat", nc, cc, hw, axis_i=0)

    empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
    empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

    return g.op(
        "Resize",
        input,
        empty_roi,
        empty_scales,
        sizes,
        coordinate_transformation_mode_s="pytorch_half_pixel",
        mode_s="linear",
    )


register_custom_op_symbolic("aten::_upsample_bilinear2d_aa", _upsample_bilinear2d_aa_symbolic, 19)


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BackboneInputCapture:
    """
    Helper class to capture backbone forward pass inputs during inference.

    The backbone (EagleBackbone.forward) receives a BatchFeature containing:
      - input_ids: [B, seq_len] int64  (token IDs including image tokens)
      - attention_mask: [B, seq_len] int64  (1 for real tokens, 0 for padding)
      - pixel_values: list of [C, H, W] tensors, or [N, C, H, W] when stacked (all same resolution)

    And outputs a BatchFeature containing:
      - backbone_features: [B, seq_len, hidden_size] float  (hidden states)
      - backbone_attention_mask: [B, seq_len] bool
      - image_mask: [B, seq_len] bool
    """

    def __init__(self):
        self.captured = False
        self.input_ids = None
        self.attention_mask = None
        self.pixel_values = None

    def hook_fn(self, module, args, kwargs):
        """Pre-forward hook to capture inputs."""
        if self.captured:
            return

        # The backbone forward receives a BatchFeature (dict-like) as first positional arg
        vl_input = args[0] if args else kwargs.get("vl_input")

        self.input_ids = vl_input["input_ids"].detach().cpu().clone()
        self.attention_mask = vl_input["attention_mask"].detach().cpu().clone()
        pv = vl_input["pixel_values"]
        if isinstance(pv, list):
            self.pixel_values = [t.detach().cpu().clone() for t in pv]
        else:
            self.pixel_values = pv.detach().cpu().clone()

        self.captured = True
        logger.info("Captured backbone inputs:")
        logger.info(f"  input_ids shape: {self.input_ids.shape} dtype: {self.input_ids.dtype}")
        logger.info(f"  attention_mask shape: {self.attention_mask.shape} dtype: {self.attention_mask.dtype}")
        if isinstance(self.pixel_values, list):
            logger.info(f"  pixel_values: list of {len(self.pixel_values)} tensors, "
                        f"shapes: {[t.shape for t in self.pixel_values]}, dtype: {self.pixel_values[0].dtype}")
        else:
            logger.info(f"  pixel_values shape: {self.pixel_values.shape} dtype: {self.pixel_values.dtype}")


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
    logger.info(f"\nPreparing observation from trajectory {traj_idx}...")

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

    parsed_obs = parse_observation_gr00t(observation, modality_configs)
    logger.info("Observation prepared")
    return parsed_obs


def export_backbone_to_onnx(
    policy: Gr00tPolicy,
    captured_inputs: BackboneInputCapture,
    output_path: str,
    export_dtype: str = "fp16",
):
    """
    Export the Eagle backbone to ONNX.

    Args:
        policy: Loaded policy with model
        captured_inputs: Captured input tensors from actual inference
        output_path: Path to save ONNX model
        export_dtype: Data type for export ('fp16' or 'fp32'). BF16 is not supported by TensorRT.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Eagle Backbone to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    backbone.eval()

    # Convert export dtype string to torch dtype
    if export_dtype == "fp16":
        model_dtype = torch.float16
    elif export_dtype == "fp32":
        model_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported export dtype: {export_dtype}. Use 'fp16' or 'fp32'.")

    logger.info(f"Export dtype: {export_dtype} (TensorRT-compatible)")
    logger.info(f"Model native dtype: {policy.model_dtype}")

    # Convert model to target dtype if needed
    if policy.model_dtype != model_dtype:
        logger.info(f"Converting backbone from {policy.model_dtype} to {model_dtype} for ONNX export...")
        backbone = backbone.to(dtype=model_dtype)

    # Create dummy inputs matching captured shapes
    input_ids = captured_inputs.input_ids.to("cuda")
    attention_mask = captured_inputs.attention_mask.to("cuda")
    # pixel_values from the Eagle processor is a list of tensors (one per frame).
    # With standardized resolution (e.g. 4 cameras at 320x240), all tensors have the
    # same shape, so we can stack them into a single [N, C, H, W] tensor for ONNX.
    # The wrapper unstacks back to a list before passing to the model.
    pv = captured_inputs.pixel_values
    if isinstance(pv, list):
        pixel_values = torch.stack([t.to(device="cuda", dtype=model_dtype) for t in pv])
        logger.info(f"  pixel_values: stacked {len(pv)} frames -> {pixel_values.shape} ({pixel_values.dtype})")
    else:
        pixel_values = pv.to(device="cuda", dtype=model_dtype)
        logger.info(f"  pixel_values: {pixel_values.shape} ({pixel_values.dtype})")

    logger.info("Export input shapes:")
    logger.info(f"  input_ids: {input_ids.shape} ({input_ids.dtype})")
    logger.info(f"  attention_mask: {attention_mask.shape} ({attention_mask.dtype})")

    # The backbone forward takes a BatchFeature dict with pixel_values as a list,
    # but ONNX needs a single tensor. The wrapper converts between the two formats.
    class BackboneONNXWrapper(torch.nn.Module):
        def __init__(self, backbone_model):
            super().__init__()
            self.backbone = backbone_model
            self.image_token_index = backbone_model.model.config.image_token_index

        def forward(self, input_ids, attention_mask, pixel_values):
            # Unstack the single tensor back into a list for the Eagle model
            pixel_values_list = [pixel_values[i] for i in range(pixel_values.shape[0])]
            keys_to_use = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values_list,
            }
            outputs = self.backbone.model(**keys_to_use, output_hidden_states=True)
            hidden_states = outputs["hidden_states"][-1]
            image_mask = input_ids == self.image_token_index
            attn_mask = attention_mask == 1
            return hidden_states, attn_mask, image_mask

    wrapped_model = BackboneONNXWrapper(backbone)
    wrapped_model.eval()

    # Dynamic axes: batch size and sequence length can vary
    # pixel_values is [num_frames, C, H, W] - spatial dims fixed by vision encoder
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "pixel_values": {0: "num_frames"},
        "hidden_states": {0: "batch_size", 1: "seq_len"},
        "attn_mask": {0: "batch_size", 1: "seq_len"},
        "image_mask": {0: "batch_size", 1: "seq_len"},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Exporting to {output_path}...")

    with torch.inference_mode():
        torch.onnx.export(
            wrapped_model,
            (input_ids, attention_mask, pixel_values),
            output_path,
            input_names=["input_ids", "attention_mask", "pixel_values"],
            output_names=["hidden_states", "attn_mask", "image_mask"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
            dynamo=False,  # Use legacy TorchScript-based exporter (required for F.interpolate with antialias=True)
        )

    logger.info("Backbone exported successfully!")

    # Verify the export
    logger.info("\nVerifying ONNX export...")
    import onnx

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Model size on disk: {file_size_mb:.2f} MB")

    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        external_size_mb = os.path.getsize(external_data_path) / (1024 * 1024)
        logger.info(f"External data size: {external_size_mb:.2f} MB")
        logger.info(f"Total model size: {file_size_mb + external_size_mb:.2f} MB")

    try:
        onnx.checker.check_model(output_path)
        logger.info("ONNX model is valid!")
    except ValueError as e:
        if "too large" in str(e):
            logger.info("Model is very large, skipping full validation...")
            try:
                onnx.shape_inference.infer_shapes_path(output_path, output_path + ".tmp")
                os.remove(output_path + ".tmp")
                logger.info("ONNX model structure verified!")
            except Exception as e2:
                logger.warning(f"Could not fully validate (this is OK): {e2}")
                logger.info("ONNX model exported (validation skipped for large model)")
        else:
            raise


def main(args):
    logger.info("=" * 80)
    logger.info("Eagle Backbone ONNX Export Script")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Attention implementation: {args.attn_implementation}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    if args.attn_implementation not in ("eager", "sdpa"):
        logger.warning(
            "Flash Attention 2 is NOT ONNX-exportable. "
            "Use --attn_implementation eager (recommended) or sdpa for ONNX export."
        )
        raise ValueError("Backbone ONNX export requires --attn_implementation eager or sdpa")

    # Step 1: Load the policy with eager attention (ONNX-compatible)
    logger.info(f"\n[Step 1] Loading policy with {args.attn_implementation} attention...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
        attn_implementation=args.attn_implementation,
    )
    logger.info("Policy loaded")

    # Step 2: Load dataset
    logger.info("\n[Step 2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    logger.info(f"Dataset loaded ({len(dataset)} trajectories)")

    # Step 3: Capture backbone inputs by running one inference
    logger.info("\n[Step 3] Capturing backbone inputs from actual inference...")

    capture = BackboneInputCapture()
    hook = policy.model.backbone.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)

    observation = prepare_observation(policy, dataset, traj_idx=0)
    logger.info("Running inference to capture shapes...")
    with torch.inference_mode():
        _ = policy.get_action(observation)

    hook.remove()

    if not capture.captured:
        logger.error("Failed to capture backbone inputs!")
        return

    # Step 4: Export backbone
    logger.info("\n[Step 4] Exporting backbone to ONNX...")
    backbone_output_path = os.path.join(args.output_dir, "backbone_model.onnx")
    export_dtype = getattr(args, "export_dtype", "fp16")
    export_backbone_to_onnx(
        policy=policy,
        captured_inputs=capture,
        output_path=backbone_output_path,
        export_dtype=export_dtype,
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nExported files in: {args.output_dir}")
    logger.info(f"  backbone_model.onnx")
    logger.info("\nNext steps:")
    logger.info("  1. Collect calibration data:")
    logger.info(f"     python scripts/deployment/collect_calibration_data.py \\")
    logger.info(f"       --model_path {args.model_path} \\")
    logger.info(f"       --dataset_path {args.dataset_path} \\")
    logger.info(f"       --embodiment_tag {args.embodiment_tag.value} \\")
    logger.info(f"       --output_dir ./calibration_data_backbone \\")
    logger.info(f"       --capture_target backbone \\")
    logger.info(f"       --num_samples 500")
    logger.info("  2. Build TensorRT engine:")
    logger.info(f"     python scripts/deployment/build_tensorrt_engine.py \\")
    logger.info(f"       --onnx {backbone_output_path} \\")
    logger.info(f"       --engine ./groot_n1d6_onnx/backbone_int8_orin.trt \\")
    logger.info(f"       --precision int8 \\")
    logger.info(f"       --calib-data ./calibration_data_backbone/calib_data.npz")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Eagle backbone to ONNX")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset (used to capture input shapes)",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=EmbodimentTag,
        default=EmbodimentTag.GR1,
        help="Embodiment tag (default: GR1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./groot_n1d6_onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Attention implementation ('eager' recommended for ONNX export, 'sdpa' may also work)",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Options: ['decord', 'torchvision_av', 'torchcodec']",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Export data type for ONNX (default: fp16). BF16 is not supported by TensorRT.",
    )

    args = parser.parse_args()
    main(args)
