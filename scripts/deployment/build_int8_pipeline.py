#!/usr/bin/env python3
"""
End-to-end INT8 quantization pipeline for the GR00T N1.6 backbone and DiT.

Orchestrates:
  1. Export backbone to ONNX (Eagle vision encoder + LLM)
  2. Export DiT to ONNX
  3. Collect backbone calibration data
  4. Collect DiT calibration data
  5. Build INT8 backbone TensorRT engine
  6. Build INT8 DiT TensorRT engine

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
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end INT8 quantization pipeline for the GR00T N1.6 backbone and DiT"
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
        "--calib-dir-backbone",
        type=str,
        default="./calibration_data_backbone",
        help="Directory for backbone calibration data (default: ./calibration_data_backbone)",
    )
    parser.add_argument(
        "--calib-dir-dit",
        type=str,
        default="./calibration_data_dit",
        help="Directory for DiT calibration data (default: ./calibration_data_dit)",
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
    parser.add_argument(
        "--opt-sa-seq",
        type=int,
        default=None,
        help="Optimal sa_embs sequence length for DiT engine (default: 51 = 1 state + 50 action). "
             "Set to 1 + action_horizon from your finetuning delta_indices for best performance.",
    )

    args = parser.parse_args()

    # Normalize to lowercase — EmbodimentTag enum values are lowercase
    # (e.g. "new_embodiment"), but users often pass uppercase (e.g. "NEW_EMBODIMENT").
    args.embodiment_tag = args.embodiment_tag.lower()

    backbone_onnx_path = os.path.join(args.onnx_dir, "backbone_model.onnx")
    dit_onnx_path = os.path.join(args.onnx_dir, "dit_model.onnx")
    backbone_engine_path = os.path.join(args.onnx_dir, "backbone_int8_orin.trt")
    dit_engine_path = os.path.join(args.onnx_dir, "dit_model_int8.trt")
    bf16_engine_path = os.path.join(args.onnx_dir, "dit_model_bf16.trt")
    backbone_calib_data = os.path.join(args.calib_dir_backbone, "calib_data.npz")
    backbone_calib_cache = os.path.join(args.calib_dir_backbone, "calibration.cache")
    dit_calib_data = os.path.join(args.calib_dir_dit, "calib_data.npz")
    dit_calib_cache = os.path.join(args.calib_dir_dit, "calibration.cache")

    total_steps = 6
    logger.info("=" * 80)
    logger.info("INT8 Quantization Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model:             {args.model_path}")
    logger.info(f"Dataset:           {args.dataset_path}")
    logger.info(f"Embodiment:        {args.embodiment_tag}")
    logger.info(f"ONNX dir:          {args.onnx_dir}")
    logger.info(f"Backbone calib dir:{args.calib_dir_backbone}")
    logger.info(f"DiT calib dir:     {args.calib_dir_dit}")
    logger.info(f"Calib samples:     {args.num_calib_samples}")
    logger.info("=" * 80)

    # ── Step 1: Export Backbone to ONNX ──────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 1/{total_steps}] Export Backbone to ONNX")
    logger.info("=" * 80)

    if os.path.exists(backbone_onnx_path) and not args.force:
        logger.info(f"Backbone ONNX already exists at {backbone_onnx_path}, skipping export")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "export_backbone_onnx.py"),
                "--model_path", args.model_path,
                "--dataset_path", args.dataset_path,
                "--embodiment_tag", args.embodiment_tag,
                "--output_dir", args.onnx_dir,
                "--attn_implementation", "eager",
                "--export_dtype", "fp16",
                "--video_backend", args.video_backend,
            ],
            "Backbone ONNX Export",
        )

    # ── Step 2: Export DiT to ONNX ───────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 2/{total_steps}] Export DiT to ONNX")
    logger.info("=" * 80)

    if os.path.exists(dit_onnx_path) and not args.force:
        logger.info(f"DiT ONNX already exists at {dit_onnx_path}, skipping export")
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
            "DiT ONNX Export",
        )

    # ── Step 3: Collect Backbone Calibration Data ────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 3/{total_steps}] Collect Backbone Calibration Data")
    logger.info("=" * 80)

    if os.path.exists(backbone_calib_data) and not args.force:
        logger.info(f"Backbone calibration data already exists at {backbone_calib_data}, skipping")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "collect_calibration_data.py"),
                "--model_path", args.model_path,
                "--dataset_path", args.dataset_path,
                "--embodiment_tag", args.embodiment_tag,
                "--output_dir", args.calib_dir_backbone,
                "--capture_target", "backbone",
                "--attn_implementation", "eager",
                "--num_samples", str(args.num_calib_samples),
                "--video_backend", args.video_backend,
            ],
            "Backbone Calibration Data Collection",
        )

    # ── Step 4: Collect DiT Calibration Data ─────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 4/{total_steps}] Collect DiT Calibration Data")
    logger.info("=" * 80)

    if os.path.exists(dit_calib_data) and not args.force:
        logger.info(f"DiT calibration data already exists at {dit_calib_data}, skipping")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "collect_calibration_data.py"),
                "--model_path", args.model_path,
                "--dataset_path", args.dataset_path,
                "--embodiment_tag", args.embodiment_tag,
                "--output_dir", args.calib_dir_dit,
                "--capture_target", "dit",
                "--num_samples", str(args.num_calib_samples),
                "--video_backend", args.video_backend,
            ],
            "DiT Calibration Data Collection",
        )

    # ── Step 5: Build INT8 Backbone TensorRT Engine ──────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 5/{total_steps}] Build INT8 Backbone TensorRT Engine")
    logger.info("=" * 80)

    if os.path.exists(backbone_engine_path) and not args.force:
        logger.info(f"Backbone INT8 engine already exists at {backbone_engine_path}, skipping")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "build_tensorrt_engine.py"),
                "--onnx", backbone_onnx_path,
                "--engine", backbone_engine_path,
                "--precision", "int8",
                "--calib-data", backbone_calib_data,
                "--calib-cache", backbone_calib_cache,
                "--max-seq-len", "512",
            ],
            "Backbone INT8 Engine Build",
        )

    # ── Step 6: Build INT8 DiT TensorRT Engine ───────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 6/{total_steps}] Build INT8 DiT TensorRT Engine")
    logger.info("=" * 80)

    if os.path.exists(dit_engine_path) and not args.force:
        logger.info(f"DiT INT8 engine already exists at {dit_engine_path}, skipping")
    else:
        dit_build_cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "build_tensorrt_engine.py"),
                "--onnx", dit_onnx_path,
                "--engine", dit_engine_path,
                "--precision", "int8",
                "--calib-data", dit_calib_data,
                "--calib-cache", dit_calib_cache,
        ]
        if args.opt_sa_seq is not None:
            dit_build_cmd.extend(["--opt-sa-seq", str(args.opt_sa_seq)])
        run_step(dit_build_cmd, "DiT INT8 Engine Build")

    # ── Validation (optional) ────────────────────────────────────────────
    if args.skip_validation:
        logger.info("\nValidation skipped (--skip-validation)")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("Validation")
        logger.info("=" * 80)

        _run_validation(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            embodiment_tag=args.embodiment_tag,
            bf16_engine_path=bf16_engine_path,
            int8_engine_path=dit_engine_path,
            traj_ids=args.validation_traj_ids,
            video_backend=args.video_backend,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("INT8 PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Backbone INT8 engine: {backbone_engine_path}")
    logger.info(f"DiT INT8 engine:      {dit_engine_path}")
    for path in [backbone_engine_path, dit_engine_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            logger.info(f"  {os.path.basename(path)}: {size_mb:.2f} MB")
    logger.info("\nRun inference with:")
    logger.info(f"  python scripts/deployment/standalone_inference_script.py \\")
    logger.info(f"    --model-path {args.model_path} \\")
    logger.info(f"    --dataset-path {args.dataset_path} \\")
    logger.info(f"    --embodiment-tag {args.embodiment_tag} \\")
    logger.info(f"    --inference-mode tensorrt \\")
    logger.info(f"    --trt-engine-path {dit_engine_path} \\")
    logger.info(f"    --backbone-trt-engine-path {backbone_engine_path}")
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
