#!/usr/bin/env python3
"""
End-to-end INT8 quantization pipeline for the GR00T N1.6 DiT model.

Orchestrates:
  1. ONNX export (skips if already present)
  2. Calibration data collection (skips if already present)
  3. INT8 TensorRT engine build
  4. Validation: compare INT8 vs BF16 action outputs + latency

Usage:
    python scripts/deployment/build_int8_pipeline.py \
        --model-path cando/checkpoint-2000 \
        --dataset-path alfiebot.CanDoChallenge \
        --embodiment-tag NEW_EMBODIMENT

    # Force rebuild everything:
    python scripts/deployment/build_int8_pipeline.py \
        --model-path cando/checkpoint-2000 \
        --dataset-path alfiebot.CanDoChallenge \
        --embodiment-tag NEW_EMBODIMENT \
        --force

    # Skip validation step:
    python scripts/deployment/build_int8_pipeline.py \
        --model-path cando/checkpoint-2000 \
        --dataset-path alfiebot.CanDoChallenge \
        --embodiment-tag NEW_EMBODIMENT \
        --skip-validation
"""

import argparse
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(cmd: list[str], step_name: str):
    """Run a subprocess and stream its output."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(SCRIPT_DIR + "/.."))
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end INT8 quantization pipeline for the DiT"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="NEW_EMBODIMENT",
        help="Embodiment tag (default: NEW_EMBODIMENT)",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="./groot_n1d6_onnx",
        help="Directory for ONNX model (default: ./groot_n1d6_onnx)",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default="./calibration_data",
        help="Directory for calibration data (default: ./calibration_data)",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=500,
        help="Number of calibration samples (default: 500)",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        help="Video backend (default: torchcodec)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild all steps even if outputs exist",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the validation step",
    )
    parser.add_argument(
        "--validation-traj-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Trajectory IDs for validation (default: 0 1 2)",
    )

    args = parser.parse_args()

    onnx_path = os.path.join(args.onnx_dir, "dit_model.onnx")
    int8_engine_path = os.path.join(args.onnx_dir, "dit_model_int8.trt")
    bf16_engine_path = os.path.join(args.onnx_dir, "dit_model_bf16.trt")
    calib_cache = os.path.join(args.calib_dir, "int8_calib.cache")

    logger.info("=" * 80)
    logger.info("INT8 Quantization Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model:        {args.model_path}")
    logger.info(f"Dataset:      {args.dataset_path}")
    logger.info(f"Embodiment:   {args.embodiment_tag}")
    logger.info(f"ONNX dir:     {args.onnx_dir}")
    logger.info(f"Calib dir:    {args.calib_dir}")
    logger.info(f"Calib samples:{args.num_calib_samples}")
    logger.info("=" * 80)

    # ── Step 1: Export ONNX ──────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("[Step 1/4] ONNX Export")
    logger.info("=" * 80)

    if os.path.exists(onnx_path) and not args.force:
        logger.info(f"ONNX model already exists at {onnx_path}, skipping export")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "export_onnx_n1d6.py"),
                "--model_path", args.model_path,
                "--dataset_path", args.dataset_path,
                "--embodiment_tag", args.embodiment_tag,
                "--output_dir", args.onnx_dir,
                "--video_backend", args.video_backend,
            ],
            "ONNX Export",
        )

    # ── Step 2: Collect calibration data ─────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("[Step 2/4] Calibration Data Collection")
    logger.info("=" * 80)

    calib_data_exists = os.path.exists(os.path.join(args.calib_dir, "sa_embs.npy"))
    if calib_data_exists and not args.force:
        logger.info(f"Calibration data already exists in {args.calib_dir}, skipping collection")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "collect_calibration_data.py"),
                "--model_path", args.model_path,
                "--dataset_path", args.dataset_path,
                "--embodiment_tag", args.embodiment_tag,
                "--output_dir", args.calib_dir,
                "--num_samples", str(args.num_calib_samples),
                "--video_backend", args.video_backend,
            ],
            "Calibration Data Collection",
        )

    # ── Step 3: Build INT8 TensorRT engine ───────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("[Step 3/4] INT8 TensorRT Engine Build")
    logger.info("=" * 80)

    if os.path.exists(int8_engine_path) and not args.force:
        logger.info(f"INT8 engine already exists at {int8_engine_path}, skipping build")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "build_tensorrt_engine.py"),
                "--onnx", onnx_path,
                "--engine", int8_engine_path,
                "--precision", "int8",
                "--calib-dir", args.calib_dir,
                "--calib-cache", calib_cache,
            ],
            "INT8 Engine Build",
        )

    # ── Step 4: Validation ───────────────────────────────────────────────
    if args.skip_validation:
        logger.info("\n[Step 4/4] Validation skipped (--skip-validation)")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("[Step 4/4] Validation")
        logger.info("=" * 80)

        _run_validation(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            embodiment_tag=args.embodiment_tag,
            bf16_engine_path=bf16_engine_path,
            int8_engine_path=int8_engine_path,
            traj_ids=args.validation_traj_ids,
            video_backend=args.video_backend,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("INT8 PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"INT8 engine: {int8_engine_path}")
    if os.path.exists(int8_engine_path):
        size_mb = os.path.getsize(int8_engine_path) / (1024**2)
        logger.info(f"Engine size:  {size_mb:.2f} MB")
    logger.info("\nRun inference with:")
    logger.info(f"  python scripts/deployment/standalone_inference_script.py \\")
    logger.info(f"    --model-path {args.model_path} \\")
    logger.info(f"    --dataset-path {args.dataset_path} \\")
    logger.info(f"    --embodiment-tag {args.embodiment_tag} \\")
    logger.info(f"    --inference-mode tensorrt \\")
    logger.info(f"    --trt-engine-path {int8_engine_path}")
    logger.info("=" * 80)


def _run_validation(
    model_path: str,
    dataset_path: str,
    embodiment_tag: str,
    bf16_engine_path: str,
    int8_engine_path: str,
    traj_ids: list[int],
    video_backend: str,
):
    """Compare INT8 vs BF16 inference: action MSE and latency."""
    import numpy as np
    import torch

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # Load policy
    logger.info("Loading policy for validation...")
    embodiment = EmbodimentTag(embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=model_path,
        device="cuda",
    )

    # Load dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=video_backend,
        video_backend_kwargs=None,
    )

    modality_configs = policy.get_modality_config()

    def prepare_observation(traj_idx, step_idx=0):
        traj = dataset[traj_idx]
        data_point = extract_step_data(
            traj, step_idx, modality_configs=modality_configs, embodiment_tag=embodiment,
        )
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for key in modality_configs["language"].modality_keys:
            obs[key] = data_point.text
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

    # Import replace_dit_with_tensorrt from the sibling module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "standalone_inference_script",
        os.path.join(os.path.dirname(__file__), "standalone_inference_script.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    replace_dit_with_tensorrt = mod.replace_dit_with_tensorrt

    def run_with_engine(engine_path, label):
        """Run inference with a specific TRT engine and measure latency."""
        # Reload fresh policy each time to avoid state contamination
        p = Gr00tPolicy(embodiment_tag=embodiment, model_path=model_path, device="cuda")
        replace_dit_with_tensorrt(p, engine_path)

        actions_list = []
        latencies = []

        for traj_id in traj_ids:
            obs = prepare_observation(traj_id)
            # Warmup
            with torch.inference_mode():
                _ = p.get_action(obs)

            # Timed run
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                result = p.get_action(obs)
            torch.cuda.synchronize()
            latencies.append(time.time() - t0)

            # get_action returns (action_dict, info) tuple
            action_dict = result[0] if isinstance(result, tuple) else result
            action_key = list(action_dict.keys())[0]
            actions_list.append(action_dict[action_key])

        avg_latency_ms = np.mean(latencies) * 1000
        logger.info(f"  {label}: avg latency = {avg_latency_ms:.2f} ms")
        return actions_list, avg_latency_ms

    # Run BF16 baseline (if engine exists)
    bf16_actions = None
    bf16_latency = None
    if os.path.exists(bf16_engine_path):
        logger.info(f"\nRunning BF16 baseline ({bf16_engine_path})...")
        bf16_actions, bf16_latency = run_with_engine(bf16_engine_path, "BF16")
    else:
        logger.info(f"\nBF16 engine not found at {bf16_engine_path}, skipping comparison")
        logger.info("  (Build one with: python build_tensorrt_engine.py --precision bf16 ...)")

    # Run INT8
    logger.info(f"\nRunning INT8 ({int8_engine_path})...")
    int8_actions, int8_latency = run_with_engine(int8_engine_path, "INT8")

    # Compare
    logger.info("\n--- Validation Results ---")
    if bf16_actions is not None:
        for i, traj_id in enumerate(traj_ids):
            mse = np.mean((bf16_actions[i] - int8_actions[i]) ** 2)
            max_err = np.max(np.abs(bf16_actions[i] - int8_actions[i]))
            logger.info(f"  Traj {traj_id}: MSE={mse:.6f}, MaxErr={max_err:.6f}")
        logger.info(f"\n  BF16 latency:  {bf16_latency:.2f} ms")
        logger.info(f"  INT8 latency:  {int8_latency:.2f} ms")
        logger.info(f"  Speedup:       {bf16_latency / int8_latency:.2f}x")
    else:
        logger.info(f"  INT8 latency: {int8_latency:.2f} ms")
        logger.info("  (No BF16 engine for comparison)")


if __name__ == "__main__":
    main()
