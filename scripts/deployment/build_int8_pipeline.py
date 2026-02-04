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
        --opt-sa-seq 17

    # Force rebuild everything:
    python scripts/deployment/build_int8_pipeline.py \
        --model-path cando/checkpoint-2000 \
        --dataset-path alfiebot.CanDoChallenge \
        --opt-sa-seq 17 \
        --refresh

    # With custom embodiment and paths:
    python scripts/deployment/build_int8_pipeline.py \
        --model-path cando/checkpoint-2000 \
        --dataset-path alfiebot.CanDoChallenge \
        --embodiment-tag new_embodiment \
        --opt-sa-seq 17 \
        --onnx-dir ./groot_n1d6_onnx
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
        default="new_embodiment",
        help="Embodiment tag (default: new_embodiment)",
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
        "--refresh",
        action="store_true",
        help="Force rebuild all steps even if outputs exist",
    )
    parser.add_argument(
        "--opt-sa-seq",
        type=int,
        default=51,
        help="Optimal sa_embs sequence length for DiT engine (default: 51 = 1 state + 50 action). "
             "Set to 1 + action_horizon from your finetuning delta_indices for best performance.",
    )

    args = parser.parse_args()

    backbone_onnx_path = os.path.join(args.onnx_dir, "backbone_model.onnx")
    dit_onnx_path = os.path.join(args.onnx_dir, "dit_model.onnx")
    backbone_engine_path = os.path.join(args.onnx_dir, "backbone_int8_orin.trt")
    dit_engine_path = os.path.join(args.onnx_dir, "dit_model_int8_orin.trt")
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
    logger.info(f"Opt sa seq:        {args.opt_sa_seq}")
    logger.info("=" * 80)

    # ── Step 1: Export Backbone to ONNX ──────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 1/{total_steps}] Export Backbone to ONNX")
    logger.info("=" * 80)

    if os.path.exists(backbone_onnx_path) and not args.refresh:
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
            ],
            "Backbone ONNX Export",
        )

    # ── Step 2: Export DiT to ONNX ───────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 2/{total_steps}] Export DiT to ONNX")
    logger.info("=" * 80)

    if os.path.exists(dit_onnx_path) and not args.refresh:
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

    if os.path.exists(backbone_calib_data) and not args.refresh:
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
            ],
            "Backbone Calibration Data Collection",
        )

    # ── Step 4: Collect DiT Calibration Data ─────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"[Step 4/{total_steps}] Collect DiT Calibration Data")
    logger.info("=" * 80)

    if os.path.exists(dit_calib_data) and not args.refresh:
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

    if os.path.exists(backbone_engine_path) and not args.refresh:
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

    if os.path.exists(dit_engine_path) and not args.refresh:
        logger.info(f"DiT INT8 engine already exists at {dit_engine_path}, skipping")
    else:
        run_step(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "build_tensorrt_engine.py"),
                "--onnx", dit_onnx_path,
                "--engine", dit_engine_path,
                "--precision", "int8",
                "--calib-data", dit_calib_data,
                "--calib-cache", dit_calib_cache,
                "--max-seq-len", "512",
                "--opt-sa-seq", str(args.opt_sa_seq),
            ],
            "DiT INT8 Engine Build",
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
    logger.info(f"    --inference-mode tensorrt \\")
    logger.info(f"    --trt-engine-path {dit_engine_path} \\")
    logger.info(f"    --backbone-trt-engine-path {backbone_engine_path} \\")
    logger.info(f"    --attn-implementation eager")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
