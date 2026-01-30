#!/usr/bin/env python3
"""
Collect calibration data for TensorRT INT8 quantization.

This script runs inference on multiple samples and captures the DiT input tensors
which are then saved for use by the TensorRT INT8 calibrator.

Usage:
    python collect_calibration_data.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path /path/to/dataset \
        --embodiment_tag GR1 \
        --output_dir ./calibration_data \
        --num_samples 500
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


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CalibrationDataCollector:
    """
    Collects DiT input tensors for INT8 calibration.

    Captures:
    - sa_embs: State-action embeddings [B, seq_len, 1536]
    - vl_embs: Vision-language embeddings [B, seq_len, 2048]
    - timestep: Diffusion timestep [B]
    - image_mask: Boolean mask for image tokens [B, seq_len]
    - backbone_attention_mask: Attention mask [B, seq_len]
    """

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples = []
        self.sample_count = 0

    def hook_fn(self, module, args, kwargs):
        """Pre-forward hook to capture inputs."""
        if self.sample_count >= self.max_samples:
            return

        # Convert to float32 before numpy (bfloat16 is not supported by numpy)
        sample = {
            "sa_embs": kwargs["hidden_states"].detach().float().cpu().numpy(),
            "vl_embs": kwargs["encoder_hidden_states"].detach().float().cpu().numpy(),
            "timestep": kwargs["timestep"].detach().cpu().numpy(),
        }

        # Optional masks
        if kwargs.get("image_mask") is not None:
            sample["image_mask"] = kwargs["image_mask"].detach().cpu().numpy()
        if kwargs.get("backbone_attention_mask") is not None:
            sample["backbone_attention_mask"] = kwargs["backbone_attention_mask"].detach().cpu().numpy()

        self.samples.append(sample)
        self.sample_count += 1

        if self.sample_count % 100 == 0:
            logger.info(f"Collected {self.sample_count}/{self.max_samples} samples")

    def _pad_to_max_len(self, arrays: list[np.ndarray], axis: int = 1) -> np.ndarray:
        """Pad arrays to the maximum sequence length along the specified axis."""
        max_len = max(arr.shape[axis] for arr in arrays)
        padded = []
        for arr in arrays:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (0, max_len - arr.shape[axis])
            padded.append(np.pad(arr, pad_width, mode="constant", constant_values=0))
        return np.concatenate(padded, axis=0)

    def save(self, output_dir: str):
        """Save collected samples to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Check if sequences have variable lengths
        sa_lens = set(s["sa_embs"].shape[1] for s in self.samples)
        vl_lens = set(s["vl_embs"].shape[1] for s in self.samples)

        # Concatenate samples (pad if needed for variable-length sequences)
        if len(sa_lens) > 1:
            logger.info(f"Padding sa_embs (found {len(sa_lens)} different lengths: {min(sa_lens)}-{max(sa_lens)})")
            sa_embs = self._pad_to_max_len([s["sa_embs"] for s in self.samples], axis=1)
        else:
            sa_embs = np.concatenate([s["sa_embs"] for s in self.samples], axis=0)

        if len(vl_lens) > 1:
            logger.info(f"Padding vl_embs (found {len(vl_lens)} different lengths: {min(vl_lens)}-{max(vl_lens)})")
            vl_embs = self._pad_to_max_len([s["vl_embs"] for s in self.samples], axis=1)
        else:
            vl_embs = np.concatenate([s["vl_embs"] for s in self.samples], axis=0)

        timesteps = np.concatenate([s["timestep"] for s in self.samples], axis=0)

        # Save as .npy files
        np.save(os.path.join(output_dir, "sa_embs.npy"), sa_embs)
        np.save(os.path.join(output_dir, "vl_embs.npy"), vl_embs)
        np.save(os.path.join(output_dir, "timesteps.npy"), timesteps)

        logger.info(f"Saved sa_embs: {sa_embs.shape}")
        logger.info(f"Saved vl_embs: {vl_embs.shape}")
        logger.info(f"Saved timesteps: {timesteps.shape}")

        # Save masks if present (also may need padding)
        if "image_mask" in self.samples[0]:
            mask_lens = set(s["image_mask"].shape[1] for s in self.samples)
            if len(mask_lens) > 1:
                image_masks = self._pad_to_max_len([s["image_mask"] for s in self.samples], axis=1)
            else:
                image_masks = np.concatenate([s["image_mask"] for s in self.samples], axis=0)
            np.save(os.path.join(output_dir, "image_masks.npy"), image_masks)
            logger.info(f"Saved image_masks: {image_masks.shape}")

        if "backbone_attention_mask" in self.samples[0]:
            mask_lens = set(s["backbone_attention_mask"].shape[1] for s in self.samples)
            if len(mask_lens) > 1:
                backbone_masks = self._pad_to_max_len(
                    [s["backbone_attention_mask"] for s in self.samples], axis=1
                )
            else:
                backbone_masks = np.concatenate(
                    [s["backbone_attention_mask"] for s in self.samples], axis=0
                )
            np.save(os.path.join(output_dir, "backbone_attention_masks.npy"), backbone_masks)
            logger.info(f"Saved backbone_attention_masks: {backbone_masks.shape}")

        # Save metadata
        metadata = {
            "num_samples": len(self.samples),
            "sa_embs_shape": list(sa_embs.shape),
            "vl_embs_shape": list(vl_embs.shape),
            "has_image_mask": "image_mask" in self.samples[0],
            "has_backbone_mask": "backbone_attention_mask" in self.samples[0],
            "padded_sa_embs": len(sa_lens) > 1,
            "padded_vl_embs": len(vl_lens) > 1,
        }

        import json
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Calibration data saved to {output_dir}")
        logger.info(f"Total samples: {len(self.samples)}")


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    """Parse observation dictionary to expected format."""
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


def prepare_observation(policy, dataset, traj_idx, step_idx=0):
    """Prepare a single observation for inference."""
    traj = dataset[traj_idx]
    modality_configs = policy.get_modality_config()

    data_point = extract_step_data(
        traj,
        step_idx,
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


def main(args):
    logger.info("=" * 80)
    logger.info("INT8 Calibration Data Collection")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info("=" * 80)

    # Load the policy
    logger.info("\n[Step 1] Loading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    logger.info("Policy loaded")

    # Load dataset
    logger.info("\n[Step 2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    logger.info(f"Dataset loaded ({len(dataset)} trajectories)")

    # Calculate samples needed
    # Each inference step calls the DiT model num_inference_timesteps times (default 4)
    # We want diverse samples, so we'll use multiple trajectories and timesteps
    num_inference_timesteps = policy.model.config.num_inference_timesteps
    samples_per_inference = num_inference_timesteps
    num_inferences_needed = (args.num_samples + samples_per_inference - 1) // samples_per_inference

    logger.info(f"\nDiT is called {num_inference_timesteps} times per inference")
    logger.info(f"Will run {num_inferences_needed} inferences to collect {args.num_samples} samples")

    # Set up collector and hook
    collector = CalibrationDataCollector(max_samples=args.num_samples)
    hook = policy.model.action_head.model.register_forward_pre_hook(
        collector.hook_fn, with_kwargs=True
    )

    # Collect samples by running inference
    logger.info("\n[Step 3] Collecting calibration data...")
    num_trajs = len(dataset)
    inference_count = 0

    # Get the first video key from modality config (to determine trajectory length)
    modality_configs = policy.get_modality_config()
    video_key = modality_configs["video"].modality_keys[0]
    video_column = f"video.{video_key}"

    # Calculate action horizon to avoid out-of-bounds indexing
    action_delta_indices = modality_configs["action"].delta_indices
    action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

    try:
        for traj_idx in range(num_trajs):
            if collector.sample_count >= args.num_samples:
                break

            # Get trajectory length using the actual video key from config
            traj = dataset[traj_idx]
            traj_len = len(traj[video_column])

            # Compute effective length (can't sample steps too close to end due to action horizon)
            effective_len = max(0, traj_len - action_horizon + 1)
            if effective_len == 0:
                continue

            # Sample steps from this trajectory within the valid range
            step_indices = np.linspace(0, effective_len - 1, min(args.steps_per_traj, effective_len), dtype=int)

            for step_idx in step_indices:
                if collector.sample_count >= args.num_samples:
                    break

                try:
                    observation = prepare_observation(policy, dataset, traj_idx, step_idx)
                    with torch.inference_mode():
                        _ = policy.get_action(observation)
                    inference_count += 1
                except Exception as e:
                    logger.warning(f"Error on traj {traj_idx} step {step_idx}: {e}")
                    continue

    finally:
        hook.remove()

    logger.info(f"\nCompleted {inference_count} inferences")
    logger.info(f"Collected {collector.sample_count} calibration samples")

    # Save calibration data
    logger.info("\n[Step 4] Saving calibration data...")
    collector.save(args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION DATA COLLECTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total samples: {collector.sample_count}")
    logger.info("\nNext step: Build INT8 TensorRT engine with:")
    logger.info(f"  python build_tensorrt_engine.py \\")
    logger.info(f"    --onnx ./groot_n1d6_onnx/dit_model.onnx \\")
    logger.info(f"    --engine ./groot_n1d6_onnx/dit_model_int8.trt \\")
    logger.info(f"    --precision int8 \\")
    logger.info(f"    --calibration_data {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect INT8 calibration data")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
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
        default="./calibration_data",
        help="Output directory for calibration data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of calibration samples to collect (default: 500)",
    )
    parser.add_argument(
        "--steps_per_traj",
        type=int,
        default=10,
        help="Number of steps to sample per trajectory (default: 10)",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Options: ['decord', 'torchvision_av', 'torchcodec']",
    )

    args = parser.parse_args()
    main(args)
