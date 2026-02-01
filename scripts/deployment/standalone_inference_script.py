from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import gc
import logging
import os
import sys
from pathlib import Path
import random
import re
import time
from typing import Any, Literal
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

# Jetson unified memory optimization: configure PyTorch CUDA allocator before any CUDA ops.
# - expandable_segments: reduces fragmentation by growing existing segments instead of new blocks
# - garbage_collection_threshold: 0.6 triggers GC earlier (default 0.8) to reclaim unused cache
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6",
)

"""
Combined inference script supporting both PyTorch and TensorRT modes.

Example commands:
 
# PyTorch mode (default):
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch

# TensorRT mode:
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode tensorrt \
  --trt_engine_path ./groot_n1d6_onnx/dit_model_bf16.trt
"""

###############################################################################
# TENSORRT Module Wrappers
###############################################################################


def set_seed(seed: int = 0):
    """
    Set seed for all random number generators.
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    # For cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch requires this to be set for some CUDA kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TensorRTDiTWrapper:
    """Wrapper for TensorRT DiT engine."""

    def __init__(self, engine_path: str, device: int = 0, use_fp16_output: bool = False):
        import tensorrt as trt

        self.device = device
        self.use_fp16_output = use_fp16_output

        # Ensures CUDA driver is properly loaded
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)  # Set the specified CUDA device
            logging.info(f"CUDA initialized via PyTorch: device {device}")
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Read and deserialize separately so we can free the file buffer immediately
        # On Jetson unified memory, the file buffer competes with GPU allocation
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        del engine_data
        gc.collect()

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Create a dedicated CUDA stream for TRT execution
        # This avoids synchronization overhead from using the default stream
        self.stream = torch.cuda.Stream(device=device)
        logging.info(f"Created dedicated CUDA stream for TRT execution")

        # Detect input dtypes from engine so we can cast inputs to match
        import tensorrt as trt
        self.input_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_dtypes[name] = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))

        # Detect output dtype from engine
        # TensorRT outputs dequantized values even for INT8 engines
        output_dtype = self.engine.get_tensor_dtype("output")
        self.engine_output_dtype = self._trt_dtype_to_torch(output_dtype)

        # Action decoder requires BF16 (Eagle backbone uses Flash Attention which needs BF16)
        # If TRT outputs FP16, we convert to BF16 after execution
        self.convert_to_bf16 = (self.engine_output_dtype == torch.float16)
        if self.convert_to_bf16:
            logging.info(f"TensorRT output dtype: {output_dtype} -> will convert FP16 to BF16 for action decoder")
        else:
            logging.info(f"TensorRT output dtype: {output_dtype} -> torch {self.engine_output_dtype}")

        # Pre-allocated output buffer — reused across diffusion steps to avoid
        # repeated alloc/free on Jetson unified memory. Reallocated only on shape change.
        self._output_buf = None
        self._output_shape = None


    def _trt_dtype_to_torch(self, trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        import tensorrt as trt

        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
        }
        return dtype_map.get(trt_dtype, torch.bfloat16)  # Default to bf16

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        """Forward pass through TensorRT DiT."""
        # Ensure default stream ops (backbone, action_encoder, etc.) complete
        self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            # Cast to engine's expected dtypes (no device move — Jetson unified memory)
            sa_embs = sa_embs.to(dtype=self.input_dtypes.get("sa_embs", sa_embs.dtype))
            if not sa_embs.is_contiguous():
                sa_embs = sa_embs.contiguous()
            vl_embs = vl_embs.to(dtype=self.input_dtypes.get("vl_embs", vl_embs.dtype))
            if not vl_embs.is_contiguous():
                vl_embs = vl_embs.contiguous()
            timestep = timestep.to(dtype=self.input_dtypes.get("timestep", timestep.dtype))
            if not timestep.is_contiguous():
                timestep = timestep.contiguous()

            if image_mask is not None:
                image_mask = image_mask.to(dtype=self.input_dtypes.get("image_mask", image_mask.dtype))
                if not image_mask.is_contiguous():
                    image_mask = image_mask.contiguous()
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.to(dtype=self.input_dtypes.get("backbone_attention_mask", backbone_attention_mask.dtype))
                if not backbone_attention_mask.is_contiguous():
                    backbone_attention_mask = backbone_attention_mask.contiguous()

            self.context.set_input_shape("sa_embs", sa_embs.shape)
            self.context.set_input_shape("vl_embs", vl_embs.shape)
            self.context.set_input_shape("timestep", timestep.shape)
            if image_mask is not None:
                self.context.set_input_shape("image_mask", image_mask.shape)
            if backbone_attention_mask is not None:
                self.context.set_input_shape("backbone_attention_mask", backbone_attention_mask.shape)

            self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
            self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
            self.context.set_tensor_address("timestep", timestep.data_ptr())
            if image_mask is not None:
                self.context.set_tensor_address("image_mask", image_mask.data_ptr())
            if backbone_attention_mask is not None:
                self.context.set_tensor_address(
                    "backbone_attention_mask", backbone_attention_mask.data_ptr()
                )

            # Reuse output buffer across diffusion steps (avoids alloc/free churn)
            output_shape = tuple(self.context.get_tensor_shape("output"))
            if self._output_shape != output_shape:
                self._output_buf = torch.empty(
                    output_shape, dtype=self.engine_output_dtype, device=f"cuda:{self.device}"
                )
                self._output_shape = output_shape
            self.context.set_tensor_address("output", self._output_buf.data_ptr())

            # Execute on our dedicated stream
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError("TensorRT inference failed")

            # Convert FP16 -> BF16 if needed (action decoder requires BF16)
            if self.convert_to_bf16:
                output = self._output_buf.to(torch.bfloat16)
            else:
                output = self._output_buf.clone()

        # Record event on TRT stream and make default stream wait for it
        event = self.stream.record_event()
        torch.cuda.current_stream().wait_event(event)

        return output


def replace_dit_with_tensorrt(policy: Gr00tPolicy | Any, trt_engine_path: str, device: int = 0, preloaded_trt: TensorRTDiTWrapper | None = None):
    """Replace the DiT forward method with TensorRT inference.

    Args:
        policy: The Gr00tPolicy instance
        trt_engine_path: Path to the TensorRT engine file
        device: CUDA device index
        preloaded_trt: Optional pre-loaded TensorRT wrapper (for memory-constrained systems)
    """
    # Free the PyTorch DiT weights if they exist (may already be deleted in TRT loading flow)
    if hasattr(policy.model.action_head, 'model') and policy.model.action_head.model is not None:
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory before DiT deletion: {mem_before:.2f} GB")

        del policy.model.action_head.model
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory after DiT deletion: {mem_after:.2f} GB (freed {mem_before - mem_after:.2f} GB)")

    # Use preloaded TRT engine if provided, otherwise load now
    if preloaded_trt is not None:
        trt_dit = preloaded_trt
        logging.info("Using pre-loaded TensorRT engine")
    else:
        trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        """
        TensorRT wrapper matching DiT forward signature.

        Maps DiT parameter names to ONNX export names:
        - hidden_states -> sa_embs
        - encoder_hidden_states -> vl_embs
        - timestep -> timestep
        - image_mask, backbone_attention_mask passed through
        """
        output = trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

        # DiT returns (output, all_hidden_states) when return_all_hidden_states=True
        if return_all_hidden_states:
            # TensorRT only returns the final output, not intermediate states
            # For inference, we don't need intermediate states, so raise
            # as this seems invalid config for inference
            raise RuntimeError("TensorRT only returns the final output. Check inference config")
        else:
            return output

    # Create a simple object to hold the TensorRT forward method
    class TRTModel:
        def __init__(self, forward_fn):
            self._forward = forward_fn

        def forward(self, *args, **kwargs):
            return self._forward(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._forward(*args, **kwargs)

    policy.model.action_head.model = TRTModel(trt_forward)
    logging.info(" DiT replaced with TensorRT engine")


class TensorRTBackboneWrapper:
    """Wrapper for TensorRT backbone engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Read and deserialize separately so we can free the file buffer immediately
        # On Jetson unified memory, the file buffer competes with GPU allocation
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        del engine_data
        gc.collect()

        if self.engine is None:
            raise RuntimeError(f"Failed to load backbone TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        # Detect input/output dtypes
        self.input_dtypes = {}
        self.output_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                trt_dtype = self.engine.get_tensor_dtype(name)
                self.output_dtypes[name] = self._trt_dtype_to_torch(trt_dtype)
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_dtypes[name] = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))

        logging.info(f"Backbone TRT engine loaded with outputs: {list(self.output_dtypes.keys())}")

        # Pre-allocated output buffers — reused across calls to avoid alloc churn
        self._output_bufs = {}
        self._output_shapes = {}

        # Profile validation is done once on first call, then skipped (shapes are constant)
        self._validated = False

    def _trt_dtype_to_torch(self, trt_dtype):
        import tensorrt as trt
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }
        return dtype_map.get(trt_dtype, torch.bfloat16)

    def __call__(self, input_ids, attention_mask, pixel_values):
        """Forward pass through TensorRT backbone.

        Uses pre-allocated buffers and skips repeated shape validation for
        minimal per-call overhead. Input shapes are validated on first call only.

        Returns:
            hidden_states, attn_mask, image_mask
        """
        self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            # No device move — Jetson unified memory; only ensure contiguous
            if not input_ids.is_contiguous():
                input_ids = input_ids.contiguous()
            if not attention_mask.is_contiguous():
                attention_mask = attention_mask.contiguous()

            # pixel_values can be a list of tensors (one per camera view) or a single tensor
            if isinstance(pixel_values, list):
                frames = [t.squeeze(0) if t.ndim == 4 else t for t in pixel_values]
                pixel_values = torch.stack(frames).unsqueeze(1)
            if not pixel_values.is_contiguous():
                pixel_values = pixel_values.contiguous()

            # Validate shapes against engine profile on first call only
            if not self._validated:
                for inp_name, inp_tensor in [
                    ("input_ids", input_ids),
                    ("attention_mask", attention_mask),
                    ("pixel_values", pixel_values),
                ]:
                    profile_min = self.engine.get_tensor_profile_shape(inp_name, 0)[0]
                    profile_max = self.engine.get_tensor_profile_shape(inp_name, 0)[2]
                    actual = tuple(inp_tensor.shape)
                    for dim_idx, (a, lo, hi) in enumerate(zip(actual, profile_min, profile_max)):
                        if a < lo or a > hi:
                            logging.error(
                                f"[Backbone TRT] Shape mismatch! {inp_name} dim {dim_idx}: "
                                f"actual={a}, profile min={lo}, max={hi}"
                            )
                self._validated = True

            self.context.set_input_shape("input_ids", input_ids.shape)
            self.context.set_input_shape("attention_mask", attention_mask.shape)
            self.context.set_input_shape("pixel_values", pixel_values.shape)

            self.context.set_tensor_address("input_ids", input_ids.data_ptr())
            self.context.set_tensor_address("attention_mask", attention_mask.data_ptr())
            self.context.set_tensor_address("pixel_values", pixel_values.data_ptr())

            # Reuse output buffers (reallocate only on shape change)
            for name, dtype in self.output_dtypes.items():
                shape = tuple(self.context.get_tensor_shape(name))
                if self._output_shapes.get(name) != shape:
                    self._output_bufs[name] = torch.empty(shape, dtype=dtype, device=f"cuda:{self.device}")
                    self._output_shapes[name] = shape
                self.context.set_tensor_address(name, self._output_bufs[name].data_ptr())

            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError("Backbone TensorRT inference failed")

        # Use event-based sync (non-blocking) instead of stream.synchronize()
        event = self.stream.record_event()
        torch.cuda.current_stream().wait_event(event)
        return self._output_bufs.get("hidden_states"), self._output_bufs.get("attn_mask"), self._output_bufs.get("image_mask")


def replace_backbone_with_tensorrt(
    policy: Gr00tPolicy | Any,
    trt_engine_path: str,
    device: int = 0,
    preloaded_trt: TensorRTBackboneWrapper | None = None,
):
    """Replace the backbone with TensorRT inference."""
    from transformers.feature_extraction_utils import BatchFeature

    if hasattr(policy.model, 'backbone') and policy.model.backbone is not None:
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory before backbone deletion: {mem_before:.2f} GB")

        del policy.model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory after backbone deletion: {mem_after:.2f} GB (freed {mem_before - mem_after:.2f} GB)")

    trt_backbone = preloaded_trt or TensorRTBackboneWrapper(trt_engine_path, device=device)

    class TRTBackbone:
        def __init__(self, trt_wrapper):
            self._trt = trt_wrapper

        def prepare_input(self, batch):
            return BatchFeature(data=batch)

        def __call__(self, vl_input):
            hidden_states, attn_mask, image_mask = self._trt(
                vl_input["input_ids"],
                vl_input["attention_mask"],
                vl_input["pixel_values"],
            )
            return BatchFeature(data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": attn_mask,
                "image_mask": image_mask,
            })

    policy.model.backbone = TRTBackbone(trt_backbone)
    logging.info(" Backbone replaced with TensorRT engine")


###############################################################################
# TENSORRT Module Wrappers End
###############################################################################


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


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
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def prepare_observation_data(
    traj: pd.DataFrame,
    step_count: int,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
    loader: LeRobotEpisodeLoader,
) -> dict[str, Any]:
    """
    Prepare observation data for inference (CPU-only operations).

    This function is designed to run asynchronously on CPU while GPU performs inference.

    Args:
        traj: Trajectory data
        step_count: Current step in trajectory
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        loader: Data loader with modality configs

    Returns:
        Parsed observation ready for inference
    """
    # Extract step data from trajectory
    data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)

    # Build observation dictionary
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v  # (T, D)
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
    for language_key in loader.modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text

    # Parse observation to expected format
    parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)

    return parsed_obs


def prepare_model_inputs(
    policy,
    traj: pd.DataFrame,
    step_count: int,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
    loader: LeRobotEpisodeLoader,
) -> tuple[dict, list]:
    """
    Full preprocessing pipeline: observation extraction + VLA processor + collation.

    CPU-only operations, safe to run in a background thread while GPU processes
    the previous step. Returns model-ready inputs that can be passed directly
    to policy.run_inference().
    """
    parsed_obs = prepare_observation_data(traj, step_count, modality_configs, embodiment_tag, loader)
    collated_inputs, states = policy.prepare_inputs(parsed_obs)
    return collated_inputs, states


def measure_episode_memory(traj: pd.DataFrame) -> dict[str, float]:
    """Measure memory consumption of a loaded episode DataFrame, in MB."""
    video_bytes = 0
    state_action_bytes = 0
    other_bytes = 0

    for col in traj.columns:
        col_bytes = 0
        for val in traj[col]:
            if hasattr(val, "nbytes"):
                # numpy array
                col_bytes += val.nbytes
            elif hasattr(val, "size") and hasattr(val, "mode"):
                # PIL Image — estimate from dimensions and mode
                w, h = val.size
                channels = len(val.getbands())
                col_bytes += w * h * channels
            else:
                col_bytes += sys.getsizeof(val)

        if col.startswith("video."):
            video_bytes += col_bytes
        elif col.startswith("state.") or col.startswith("action."):
            state_action_bytes += col_bytes
        else:
            other_bytes += col_bytes

    total = video_bytes + state_action_bytes + other_bytes
    return {
        "video_mb": video_bytes / (1024 * 1024),
        "state_action_mb": state_action_bytes / (1024 * 1024),
        "other_mb": other_bytes / (1024 * 1024),
        "total_mb": total / (1024 * 1024),
    }


def run_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps=300,
    action_horizon=16,
    skip_timing_steps=1,
    get_performance_stats=False,
):
    """
    Run inference on a single trajectory.

    Args:
        skip_timing_steps: Number of initial inference steps to skip when calculating timing statistics

    Returns: tuple: (
        state_keys,
        action_keys,
        pred_action_across_time,
        traj,
        actual_steps,
        timing_dict,
    )
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"=== Running Trajectory {traj_id} ===")
    logging.info("=" * 80)

    # Timing accumulators
    timing_dict = {
        "episode_load_time": 0.0,
        "data_prep_times": [],
        "inference_times": [],
    }

    # Load episode
    episode_load_start = time.time()
    traj = loader[traj_id]
    timing_dict["episode_load_time"] = time.time() - episode_load_start

    if get_performance_stats:
        mem_stats = measure_episode_memory(traj)
        timing_dict["episode_memory"] = mem_stats
        logging.info(f"Episode memory: {mem_stats['total_mb']:.1f} MB total "
                     f"(video: {mem_stats['video_mb']:.1f} MB, "
                     f"state/action: {mem_stats['state_action_mb']:.1f} MB, "
                     f"other: {mem_stats['other_mb']:.1f} MB)")

    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    # Inference loop with deep async prefetching
    # The full preprocessing chain (observation extraction + VLA processor + collation)
    # runs in a background thread, overlapping with GPU inference on the previous step.
    # This hides ~33ms of CPU preprocessing behind the ~235ms backbone execution.
    num_inference_steps = len(range(0, actual_steps, action_horizon))
    logging.info(f"\nRunning {num_inference_steps} inference steps...")
    logging.info(f"(Skipping first {skip_timing_steps} step(s) for timing statistics)")
    logging.info("Using deep async prefetching: full preprocessing overlaps with GPU inference")
    logging.info("-" * 80)

    # Create thread pool for async data preparation (single worker is sufficient)
    executor = ThreadPoolExecutor(max_workers=1)

    # List of step counts to process
    step_counts = list(range(0, actual_steps, action_horizon))

    # Deep prefetch: run full preprocessing (obs extraction + VLA processor + collation)
    future_inputs = executor.submit(
        prepare_model_inputs,
        policy,
        traj,
        step_counts[0],
        modality_configs,
        embodiment_tag,
        loader,
    )

    for step_idx, step_count in enumerate(step_counts):
        logging.info(
            f"\n[Step {step_idx + 1}/{num_inference_steps}] Processing timestep {step_count}"
        )

        # Wait for full preprocessing to complete (should be ready from prefetch)
        data_prep_start = time.time()
        collated_inputs, states = future_inputs.result()  # Blocks until ready
        data_prep_time = time.time() - data_prep_start

        # Deep prefetch NEXT step's full preprocessing while GPU runs inference
        if step_idx + 1 < len(step_counts):
            next_step_count = step_counts[step_idx + 1]
            future_inputs = executor.submit(
                prepare_model_inputs,
                policy,
                traj,
                next_step_count,
                modality_configs,
                embodiment_tag,
                loader,
            )

        # Inference timing (GPU only - preprocessing already done)
        inference_start = time.time()
        _action_chunk = policy.run_inference(collated_inputs, states)
        inference_time = time.time() - inference_start

        # Only record timing after skipping the first N steps (warmup)
        if step_idx >= skip_timing_steps:
            timing_dict["data_prep_times"].append(data_prep_time)
            timing_dict["inference_times"].append(inference_time)

        # Action processing
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    # Clean up thread pool
    executor.shutdown(wait=True)
    del collated_inputs, states

    logging.info("\n" + "-" * 80)
    logging.info(f"All inference steps completed for current trajectory-id {traj_id}")

    # Drop video columns to free PIL Image memory (~500MB-1.5GB).
    # evaluate_predictions only needs state.* and action.* columns.
    video_cols = [c for c in traj.columns if c.startswith("video.")]
    if video_cols:
        traj.drop(columns=video_cols, inplace=True)
        gc.collect()

    return (
        state_keys,
        action_keys,
        np.array(pred_action_across_time),
        traj,
        actual_steps,
        timing_dict,
    )


def evaluate_predictions(
    state_keys,
    action_keys,
    pred_action_across_time,
    traj,
    traj_id,
    actual_steps,
    action_horizon,
    save_plot_path=None,
):
    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/stand_alone_inference/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GR1
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str = "./groot_n1d6_onnx/dit_model_bf16.trt"
    """Path to TensorRT engine file (.trt). Used only when inference_mode='tensorrt'."""

    trt_use_fp16: bool = False
    """DEPRECATED: No longer used. Pure FP16 pipeline is now default for TensorRT mode."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    skip_timing_steps: int = 1
    """Number of initial inference steps to skip when calculating timing statistics (default: 1 to exclude warmup)."""

    get_performance_stats: bool = True
    """Agreegate and summarize timing and accuracy stats across several runs"""

    seed: int = 42
    """Seed to use for reproducibility."""

    attn_implementation: str | None = None
    """Override backbone attention implementation. Options: 'flash_attention_2' (default), 'sdpa' (ONNX/TRT-compatible)."""

    backbone_trt_engine_path: str = ""
    """Path to TensorRT engine file for the backbone. When set, replaces PyTorch backbone with TRT."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info("\n" + "=" * 80)
    logging.info("=" * 80)
    logging.info(f"Model Path: {args.model_path}")
    logging.info(f"Dataset Path: {args.dataset_path}")
    logging.info(f"Embodiment Tag: {args.embodiment_tag}")
    logging.info(f"Trajectories: {args.traj_ids}")
    logging.info(f"Steps per trajectory: {args.steps}")
    logging.info(f"Action Horizon: {args.action_horizon}")
    logging.info(f"Denoising Steps: {args.denoising_steps}")
    logging.info(f"Skip Timing Steps: {args.skip_timing_steps}")
    logging.info(f"Inference Mode: {args.inference_mode}")
    if args.inference_mode == "tensorrt":
        logging.info(f"TensorRT Engine: {args.trt_engine_path}")
        logging.info(f"TensorRT FP16 output: {args.trt_use_fp16}")
        if args.backbone_trt_engine_path:
            logging.info(f"Backbone TRT Engine: {args.backbone_trt_engine_path}")
    if args.attn_implementation:
        logging.info(f"Attention Implementation: {args.attn_implementation}")
    logging.info(f"Seed: {args.seed}")
    set_seed(args.seed)
    logging.info("=" * 80)

    # Download model checkpoint
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    assert local_model_path is not None, "Provide valid model_path for inference"
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    # Model loading
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 1: Loading Policy ===")
    logging.info("=" * 80)
    model_load_start = time.time()

    if local_model_path is not None:
        # For TensorRT mode on memory-constrained systems (e.g., Jetson with unified memory):
        # 1. Load PyTorch model to CPU (avoids GPU memory spike)
        # 2. Delete the PyTorch DiT (frees ~2GB of CPU memory)
        # 3. Load TRT engine to GPU
        # 4. Move remaining PyTorch components to GPU
        if args.inference_mode == "tensorrt" and torch.cuda.is_available():
            # TRT needs contiguous GPU memory - load ALL TRT engines FIRST while GPU is empty
            # This is critical on Jetson unified memory where CPU/GPU share 16GB:
            # Loading TRT engines before PyTorch avoids memory fragmentation and
            # ensures the large contiguous allocations succeed.
            logging.info("TensorRT mode: Loading ALL TRT engines first while GPU is empty...")

            # Load backbone TRT engine first (largest: ~3GB)
            backbone_trt = None
            if args.backbone_trt_engine_path:
                logging.info(f"Loading backbone TRT engine: {args.backbone_trt_engine_path}")
                backbone_trt = TensorRTBackboneWrapper(args.backbone_trt_engine_path, device=0)
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    logging.info(f"GPU memory after backbone TRT load: {mem_used:.2f} GB")

            # Load DiT TRT engine
            logging.info(f"Loading DiT TensorRT engine: {args.trt_engine_path}")
            trt_dit = TensorRTDiTWrapper(
                args.trt_engine_path,
                device=0,
                use_fp16_output=False,  # Let TRT output native dtype, convert in wrapper
            )
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"GPU memory after all TRT engines loaded: {mem_used:.2f} GB")

            # Load PyTorch model WITHOUT DiT (skip_dit=True saves ~2GB)
            # This is critical for Jetson unified memory where CPU and GPU share 16GB
            skip_backbone = bool(args.backbone_trt_engine_path)
            use_fp16 = args.attn_implementation == "sdpa"  # FP16 viable with SDPA (no flash_attn BF16 requirement)
            logging.info(f"Loading PyTorch model (skip_dit=True, skip_backbone={skip_backbone})...")
            policy = Gr00tPolicy(
                embodiment_tag=args.embodiment_tag,
                model_path=local_model_path,
                device="cuda",
                skip_dit=True,  # Don't load DiT weights - we'll use TRT instead
                skip_backbone=skip_backbone,
                use_fp16=use_fp16,
                attn_implementation=args.attn_implementation,
            )
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"GPU memory after PyTorch model load: {mem_used:.2f} GB")

            # Wire up TRT engines to replace the (empty) PyTorch components
            replace_dit_with_tensorrt(policy, args.trt_engine_path, preloaded_trt=trt_dit)

            if backbone_trt is not None:
                replace_backbone_with_tensorrt(policy, args.backbone_trt_engine_path, preloaded_trt=backbone_trt)

            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(f"GPU memory after all TRT replacements: allocated={mem_used:.2f} GB, reserved={mem_reserved:.2f} GB")

            logging.info(" TensorRT mode enabled")
        else:
            # PyTorch mode - load directly to GPU
            policy = Gr00tPolicy(
                embodiment_tag=args.embodiment_tag,
                model_path=local_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                attn_implementation=args.attn_implementation,
            )

            # PyTorch mode with torch.compile
            policy.model.action_head.model.forward = torch.compile(
                policy.model.action_head.model.forward, mode="max-autotune"
            )
            logging.info(" PyTorch mode enabled with torch.compile")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Cap PyTorch's CUDA cache at 60% of total memory to leave headroom for
            # TensorRT internal buffers, numpy, OS, and other allocations on unified memory
            torch.cuda.set_per_process_memory_fraction(0.6)
            logging.info("PyTorch CUDA memory capped at 60% of total unified memory")
    else:
        assert 0, "Please provide valid model_path argument for inference"
    # Override denoising steps if CLI arg differs from model config
    model_denoise = policy.model.action_head.num_inference_timesteps
    if args.denoising_steps != model_denoise:
        logging.info(f"Overriding num_inference_timesteps: {model_denoise} -> {args.denoising_steps}")
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

    model_load_time = time.time() - model_load_start
    logging.info(f"Model loading time: {model_load_time:.4f} seconds")

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()

    # Auto-fix action_horizon if model config doesn't match training delta_indices.
    # The finetuning pipeline doesn't update config.json, so the base model's
    # action_horizon (e.g. 50) may persist even when finetuned with fewer steps.
    model_ah = policy.model.action_head.action_horizon
    decode_ah = len(modality["action"].delta_indices)
    if model_ah != decode_ah:
        logging.info(f"Fixing action_horizon mismatch: model config={model_ah}, "
                     f"training delta_indices={decode_ah}. Setting to {decode_ah}.")
        policy.model.action_head.config.action_horizon = decode_ah
        policy.model.action_head.action_horizon = decode_ah
    logging.info(f"Current modality config: \n{modality}")

    # Dataset creation
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 2: Creating Dataset Loader ===")
    logging.info("=" * 80)
    dataset_load_start = time.time()

    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    dataset_load_time = time.time() - dataset_load_start
    logging.info(f"Dataset loader creation time: {dataset_load_time:.4f} seconds")

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    # Evaluation loop
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 3: Running Evaluation ===")
    logging.info("=" * 80)

    all_mse = []
    all_mae = []
    all_timings = []
    pred_actions = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        (
            state_keys,
            action_keys,
            pred_action_across_time,
            traj,
            actual_steps,
            timing_dict,
        ) = run_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            steps=args.steps,
            action_horizon=args.action_horizon,
            skip_timing_steps=args.skip_timing_steps,
            get_performance_stats=args.get_performance_stats,
        )
        pred_actions.append(pred_action_across_time)

        if args.get_performance_stats:
            mse, mae = evaluate_predictions(
                state_keys,
                action_keys,
                pred_action_across_time,
                traj,
                traj_id,
                actual_steps,
                args.action_horizon,
                save_plot_path=None,
            )

            logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
            all_mse.append(mse)
            all_mae.append(mae)
            all_timings.append(timing_dict)

        # Free trajectory data between runs
        del traj
        gc.collect()

    if args.get_performance_stats:
        # Final performance summary
        logging.info("\n" + "=" * 80)
        logging.info("=== EVALUATION SUMMARY ===")
        logging.info("=" * 80)

        if all_mse:
            avg_mse = np.mean(np.array(all_mse))
            avg_mae = np.mean(np.array(all_mae))
            logging.info("\nMetrics:")
            logging.info(f"  Average MSE across all trajs: {avg_mse:.6f}")
            logging.info(f"  Average MAE across all trajs: {avg_mae:.6f}")
        else:
            logging.info("No valid trajectories were evaluated.")

        # Detailed timing summary
        logging.info("\n" + "=" * 80)
        logging.info("=== DETAILED TIMING SUMMARY ===")
        logging.info("=" * 80)
        logging.info("\nInitialization:")
        logging.info(f"  Model loading time:          {model_load_time:.4f}s")
        logging.info(f"  Dataset loader creation:     {dataset_load_time:.4f}s")

        if all_timings:
            # Aggregate timing statistics
            total_episode_load = sum(t["episode_load_time"] for t in all_timings)
            total_data_prep = sum(sum(t["data_prep_times"]) for t in all_timings)
            total_inference = sum(sum(t["inference_times"]) for t in all_timings)

            # Count total inference steps
            total_inference_steps = sum(len(t["inference_times"]) for t in all_timings)

            logging.info(f"\nPer-Trajectory Timings ({len(all_timings)} trajectories):")
            logging.info(
                f"  Total episode loading:       {total_episode_load:.4f}s  (avg: {total_episode_load / len(all_timings):.4f}s)"
            )
            logging.info(
                f"  Total data preparation:      {total_data_prep:.4f}s  (avg: {total_data_prep / total_inference_steps:.4f}s per step)"
            )
            logging.info(
                f"  Total inference:             {total_inference:.4f}s  (avg: {total_inference / total_inference_steps:.4f}s per step)"
            )

            logging.info("\nInference Statistics:")
            logging.info(f"  Total inference steps:       {total_inference_steps}")
            logging.info(
                f"  Avg inference time per step: {total_inference / total_inference_steps:.4f}s"
            )

            # Collect all inference times for min/max/p90
            all_inf_times = [t for timing in all_timings for t in timing["inference_times"]]
            logging.info(f"  Min inference time:          {min(all_inf_times):.4f}s")
            logging.info(f"  Max inference time:          {max(all_inf_times):.4f}s")
            logging.info(f"  P90 inference time:          {np.percentile(all_inf_times, 90):.4f}s")

            # Episode memory summary
            mem_stats_list = [t.get("episode_memory") for t in all_timings if "episode_memory" in t]
            if mem_stats_list:
                logging.info("\nEpisode Memory Usage:")
                for i, mem in enumerate(mem_stats_list):
                    logging.info(f"  Trajectory {i}: {mem['total_mb']:.1f} MB "
                                 f"(video: {mem['video_mb']:.1f} MB, "
                                 f"state/action: {mem['state_action_mb']:.1f} MB)")
                avg_total = np.mean([m["total_mb"] for m in mem_stats_list])
                avg_video = np.mean([m["video_mb"] for m in mem_stats_list])
                logging.info(f"  Average: {avg_total:.1f} MB total (video: {avg_video:.1f} MB)")

    logging.info("=" * 80)
    logging.info("Done")
    return pred_actions


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
