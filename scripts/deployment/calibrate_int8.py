#!/usr/bin/env python3
"""
INT8 Calibration Data Collection and TensorRT Calibrator for GR00T.

Collects activation statistics from representative data for TensorRT INT8
quantization. Supports both DiT and backbone calibration.

Two-step workflow:
  1. Collect calibration data:
     python calibrate_int8.py collect \
         --model_path nvidia/GR00T-N1.6-3B \
         --dataset_path /path/to/dataset \
         --output_dir ./groot_n1d6_onnx/calib_data \
         --num_trajectories 125 \
         --component dit

  2. Build INT8 engine (uses the calibrator class from this file):
     python build_tensorrt_engine.py \
         --onnx ./groot_n1d6_onnx/dit_model.onnx \
         --engine ./groot_n1d6_onnx/dit_int8.trt \
         --precision int8 \
         --calib-data ./groot_n1d6_onnx/calib_data/dit
"""

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers (reused from existing scripts)
# ---------------------------------------------------------------------------


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


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


def prepare_model_inputs(policy, observation):
    """Prepare inputs for the model (from benchmark_inference.py pattern)."""
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
# DiT input capture (captures ALL denoising steps)
# ---------------------------------------------------------------------------


class DiTCalibrationCapture:
    """Capture DiT inputs at every denoising step for calibration."""

    def __init__(self):
        self.samples: list[dict] = []

    def hook_fn(self, module, args, kwargs):
        sample = {
            "sa_embs": kwargs["hidden_states"].detach().cpu().float().numpy(),
            "vl_embs": kwargs["encoder_hidden_states"].detach().cpu().float().numpy(),
            "timestep": kwargs["timestep"].detach().cpu().numpy(),
        }
        img_mask = kwargs.get("image_mask")
        if img_mask is not None:
            sample["image_mask"] = img_mask.detach().cpu().numpy()
        bb_mask = kwargs.get("backbone_attention_mask")
        if bb_mask is not None:
            sample["backbone_attention_mask"] = bb_mask.detach().cpu().numpy()
        self.samples.append(sample)

    def reset(self):
        self.samples = []


# ---------------------------------------------------------------------------
# Collect calibration data
# ---------------------------------------------------------------------------


def collect_dit_calibration_data(
    policy: Gr00tPolicy,
    dataset: LeRobotEpisodeLoader,
    output_dir: str,
    num_trajectories: int = 125,
    action_horizon: int = 16,
):
    """
    Run inference on trajectories and capture DiT inputs at every denoising step.
    Saves each sample as individual .npy files for the TRT calibrator.
    """
    os.makedirs(output_dir, exist_ok=True)

    modality_configs = deepcopy(policy.get_modality_config())
    modality_configs.pop("action", None)

    capture = DiTCalibrationCapture()
    dit_model = policy.model.action_head.model
    hook = dit_model.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)

    sample_count = 0
    num_trajs = min(num_trajectories, len(dataset))

    for traj_idx in range(num_trajs):
        traj = dataset[traj_idx]
        traj_len = len(traj)

        # Sample a few steps per trajectory
        step_indices = list(range(0, min(traj_len, 3 * action_horizon), action_horizon))[:3]

        for step_idx in step_indices:
            # Prepare observation
            data_point = extract_step_data(
                traj, step_idx, modality_configs, policy.embodiment_tag
            )
            obs = {}
            for k, v in data_point.states.items():
                obs[f"state.{k}"] = v
            for k, v in data_point.images.items():
                obs[f"video.{k}"] = np.array(v)
            for key in modality_configs["language"].modality_keys:
                obs[key] = data_point.text
            parsed_obs = parse_observation_gr00t(obs, modality_configs)

            capture.reset()

            with torch.inference_mode():
                _ = policy.get_action(parsed_obs)

            # Save each captured denoising step as a separate sample
            for sample in capture.samples:
                for key, arr in sample.items():
                    np.save(os.path.join(output_dir, f"sample_{sample_count:06d}_{key}.npy"), arr)
                sample_count += 1

        if (traj_idx + 1) % 10 == 0:
            logger.info(f"  Processed {traj_idx + 1}/{num_trajs} trajectories, {sample_count} samples")

    hook.remove()

    # Save metadata
    metadata = {
        "num_samples": sample_count,
        "num_trajectories": num_trajs,
        "input_names": list(capture.samples[0].keys()) if capture.samples else [],
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)

    logger.info(f"Saved {sample_count} calibration samples to {output_dir}")
    return sample_count


# ---------------------------------------------------------------------------
# TensorRT INT8 Calibrator
# ---------------------------------------------------------------------------


class DiTInt8Calibrator:
    """
    TensorRT IInt8EntropyCalibrator2 for DiT model.
    Feeds pre-collected calibration data to TensorRT during engine build.

    Usage:
        calibrator = DiTInt8Calibrator(calib_data_dir, cache_file)
        config.int8_calibrator = calibrator
    """

    def __init__(self, calib_data_dir: str, cache_file: str = "dit_int8_calib.cache"):
        import tensorrt as trt

        # Inherit from the correct base class at runtime
        self._base = trt.IInt8EntropyCalibrator2
        self.calib_data_dir = calib_data_dir
        self.cache_file = cache_file

        # Load metadata
        metadata = np.load(os.path.join(calib_data_dir, "metadata.npy"), allow_pickle=True).item()
        self.num_samples = metadata["num_samples"]
        self.input_names = metadata["input_names"]
        self.current_idx = 0

        # Pre-allocate GPU buffers (will be sized on first batch)
        self.device_buffers = {}

        logger.info(f"DiT INT8 Calibrator: {self.num_samples} samples from {calib_data_dir}")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Feed one calibration sample to TensorRT."""
        if self.current_idx >= self.num_samples:
            return None

        bindings = []
        for name in names:
            # Load numpy array for this input
            npy_path = os.path.join(
                self.calib_data_dir, f"sample_{self.current_idx:06d}_{name}.npy"
            )
            if not os.path.exists(npy_path):
                logger.warning(f"Missing calibration file: {npy_path}")
                return None

            data = np.load(npy_path)

            # Allocate or reuse GPU buffer
            if name not in self.device_buffers or self.device_buffers[name].nbytes < data.nbytes:
                import cuda.cuda as cuda_drv
                import cuda.cudart as cudart

                if name in self.device_buffers:
                    cudart.cudaFree(self.device_buffers[name])
                err, ptr = cudart.cudaMalloc(data.nbytes)
                self.device_buffers[name] = ptr
            else:
                ptr = self.device_buffers[name]

            # Copy data to GPU
            import cuda.cudart as cudart
            cudart.cudaMemcpy(
                ptr, data.ctypes.data, data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            bindings.append(int(ptr))

        self.current_idx += 1
        return bindings

    def read_calibration_cache(self):
        """Read calibration cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            logger.info(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache to disk."""
        logger.info(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def create_dit_calibrator(calib_data_dir: str, cache_file: str = None):
    """
    Factory function that creates a proper TensorRT calibrator class
    by dynamically inheriting from trt.IInt8EntropyCalibrator2.

    This is needed because trt.IInt8EntropyCalibrator2 must be a base class,
    not just delegated to.
    """
    import tensorrt as trt

    class _DiTCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data_dir, cache):
            super().__init__()
            self.data_dir = data_dir
            self._cache_file = cache or os.path.join(data_dir, "dit_int8_calib.cache")

            metadata = np.load(
                os.path.join(data_dir, "metadata.npy"), allow_pickle=True
            ).item()
            self.num_samples = metadata["num_samples"]
            self.input_names = metadata.get("input_names", [])
            self.current_idx = 0
            self._device_mem = {}

            # Use torch CUDA for GPU memory (more reliable than pycuda in TRT context)
            import torch
            self._torch = torch
            torch.cuda.init()

            logger.info(f"DiT INT8 Calibrator: {self.num_samples} samples, "
                        f"inputs: {self.input_names}")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.current_idx >= self.num_samples:
                return None

            bindings = []
            for name in names:
                npy_path = os.path.join(
                    self.data_dir, f"sample_{self.current_idx:06d}_{name}.npy"
                )
                if not os.path.exists(npy_path):
                    logger.warning(f"Missing {npy_path}, skipping sample {self.current_idx}")
                    self.current_idx += 1
                    if self.current_idx >= self.num_samples:
                        return None
                    return self.get_batch(names)

                data = np.load(npy_path)
                # Cast floating point data to FP32 for calibration
                if data.dtype in (np.float16, np.float64) or str(data.dtype).startswith("bfloat"):
                    data = data.astype(np.float32)
                data = np.ascontiguousarray(data)

                # Use torch CUDA tensors for GPU memory management
                t = self._torch.from_numpy(data).cuda()
                if name not in self._device_mem:
                    self._device_mem[name] = t
                else:
                    # Reuse if same size, otherwise reallocate
                    if self._device_mem[name].numel() >= t.numel():
                        self._device_mem[name][:t.numel()].copy_(t.flatten())
                        t = self._device_mem[name][:t.numel()].reshape(data.shape)
                    else:
                        self._device_mem[name] = t

                bindings.append(t.data_ptr())

            self.current_idx += 1
            if self.current_idx % 100 == 0:
                logger.info(f"  Calibration progress: {self.current_idx}/{self.num_samples}")
            return bindings

        def read_calibration_cache(self):
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self._cache_file, "wb") as f:
                f.write(cache)

    return _DiTCalibrator(calib_data_dir, cache_file)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="INT8 calibration data collection for GR00T")
    subparsers = parser.add_subparsers(dest="command")

    # Collect subcommand
    collect_parser = subparsers.add_parser("collect", help="Collect calibration data")
    collect_parser.add_argument("--model_path", type=str, required=True)
    collect_parser.add_argument("--dataset_path", type=str, required=True)
    collect_parser.add_argument("--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.GR1)
    collect_parser.add_argument("--output_dir", type=str, default="./groot_n1d6_onnx/calib_data/dit")
    collect_parser.add_argument("--num_trajectories", type=int, default=125)
    collect_parser.add_argument(
        "--component", type=str, choices=["dit", "backbone", "both"], default="dit"
    )
    collect_parser.add_argument("--video_backend", type=str, default="torchcodec")

    args = parser.parse_args()

    if args.command == "collect":
        logger.info("=" * 80)
        logger.info("INT8 Calibration Data Collection")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Dataset: {args.dataset_path}")
        logger.info(f"Component: {args.component}")
        logger.info(f"Trajectories: {args.num_trajectories}")
        logger.info(f"Output: {args.output_dir}")

        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            device="cuda",
        )

        dataset = LeRobotEpisodeLoader(
            dataset_path=args.dataset_path,
            modality_configs=policy.get_modality_config(),
            video_backend=args.video_backend,
        )

        if args.component in ("dit", "both"):
            dit_dir = args.output_dir if args.component == "dit" else os.path.join(args.output_dir, "dit")
            num = collect_dit_calibration_data(
                policy, dataset, dit_dir, args.num_trajectories
            )
            logger.info(f"DiT calibration: {num} samples saved to {dit_dir}")

        logger.info("Done!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
