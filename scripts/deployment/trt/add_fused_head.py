# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add a fused action-head engine for another denoising-step count to an engine bundle.

``build_trt_pipeline.py --fused-denoising-steps`` builds fused heads only as part of a full
export/quantize/build run (about 30 minutes on AGX Orin). This adds one step count to a
bundle that pipeline already produced, in about 10 minutes: it exports only the fused
graph, INT8-calibrates it the way the pipeline does when the bundle has ``onnx_int8/``,
and builds it with the bundle's shape hints. The runtime picks the new engine up for that
``--denoising-steps`` value; nothing else in the bundle changes.

Example:
    python -m scripts.deployment.trt.add_fused_head \\
        --model-path checkpoints/alfie/checkpoint-20000 --dataset-path <lerobot> \\
        --embodiment-tag new_embodiment --bundle-dir gr00t_trt_deployment/alfie_int8_rtc \\
        --steps 8
"""

from dataclasses import dataclass
import gc
import logging
import os
import shutil
import tempfile

import torch
import tyro

from .build_tensorrt_engine import build_engine, derive_shapes_with_hint
from .export_onnx_n1d7 import FUSED_ACTION_HEAD_ONNX, export_fused_action_head_to_onnx
from .int8_calibration import quantize_pipeline


logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_path: str
    """The checkpoint the bundle was built from."""
    dataset_path: str
    """Calibration data (use the one the bundle was calibrated on)."""
    bundle_dir: str
    """build_trt_pipeline.py --output-dir: holds onnx/, engines/ and, for INT8, onnx_int8/."""
    steps: int
    """Denoising-step count of the new fused engine."""
    embodiment_tag: str = "new_embodiment"
    int8_calibration_frames: int = 64
    """As build_trt_pipeline.py's default."""
    workspace: int = 8192


def main(cfg: Config) -> None:
    import json

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logging.basicConfig(level=logging.INFO)
    onnx_dir = os.path.join(cfg.bundle_dir, "onnx")
    int8_dir = os.path.join(cfg.bundle_dir, "onnx_int8")
    engine_dir = os.path.join(cfg.bundle_dir, "engines")
    with open(os.path.join(onnx_dir, "export_metadata.json")) as f:
        metadata = json.load(f)
    tag = EmbodimentTag.resolve(cfg.embodiment_tag)
    if str(tag) != metadata["embodiment_tag"]:
        raise ValueError(f"Bundle was exported for {metadata['embodiment_tag']}, not {tag}")
    int8 = os.path.isdir(int8_dir)
    name = FUSED_ACTION_HEAD_ONNX.format(steps=cfg.steps)
    vl_seq_len = int(metadata["vl_seq_len"])
    num_image_tokens = int(metadata["num_image_tokens"])

    # 1. Export the fused graph next to the bundle's other graphs.
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device="cuda")
    export_fused_action_head_to_onnx(
        policy,
        onnx_dir,
        vl_seq_len=vl_seq_len,
        num_steps=cfg.steps,
        embodiment_id=int(metadata["embodiment_id"]),
        num_image_tokens=num_image_tokens,
        use_bf16=True,
        batch_size=int(metadata.get("batch_size", 1)),
    )
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    # 2. INT8: calibrate only the new graph (a scratch directory holds just it, so the
    #    bundle's other INT8 graphs are not re-quantized), then move it into onnx_int8/.
    source_dir = onnx_dir
    if int8:
        with tempfile.TemporaryDirectory(dir=cfg.bundle_dir) as scratch:
            src, dst = os.path.join(scratch, "onnx"), os.path.join(scratch, "onnx_int8")
            os.makedirs(src)
            for file in (name, name + ".data", "export_metadata.json"):
                if os.path.exists(os.path.join(onnx_dir, file)):
                    os.symlink(
                        os.path.abspath(os.path.join(onnx_dir, file)), os.path.join(src, file)
                    )
            policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device="cuda")
            dataset = LeRobotEpisodeLoader(
                dataset_path=cfg.dataset_path, modality_configs=policy.get_modality_config()
            )
            quantize_pipeline(
                policy, dataset, src, dst, ("action_head",), num_frames=cfg.int8_calibration_frames
            )
            del policy
            gc.collect()
            torch.cuda.empty_cache()
            for file in (name, name + ".data"):
                target = os.path.join(int8_dir, file)
                if os.path.lexists(target):
                    os.remove(target)
                if os.path.exists(os.path.join(dst, file)):
                    shutil.move(os.path.join(dst, file), target)
        source_dir = int8_dir

    # 3. Build with the bundle's shape hints (as build_full_pipeline derives them).
    hints = {
        "vl_seq_len": vl_seq_len,
        "num_image_tokens": num_image_tokens,
        "num_text_tokens": int(metadata.get("num_text_tokens", vl_seq_len - num_image_tokens)),
    }
    max_hints = {"num_text_tokens": max(vl_seq_len * 2, vl_seq_len + 64)}
    onnx_path = os.path.join(source_dir, name)
    min_shapes, opt_shapes, max_shapes = derive_shapes_with_hint(
        onnx_path, opt_seq_lens=hints, max_seq_lens=max_hints
    )
    engine_path = os.path.join(engine_dir, name.replace(".onnx", ".engine"))
    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision="bf16",
        workspace_mb=cfg.workspace,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )
    logger.info("Added %s (%s)", engine_path, "INT8" if int8 else "BF16")


if __name__ == "__main__":
    main(tyro.cli(Config))
