# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""INT8 quantization step of the TensorRT pipeline (``build_trt_pipeline.py --int8 ...``).

1. Run the PyTorch policy on frames spread over the dataset and record the real inputs of
   the graphs to quantize: the ViT's ``pixel_values``, the fused action head's inputs, and
   the per-step DiT inputs.
2. Quantize those graphs with ``quantize_onnx_int8`` into a mirror of the ONNX directory
   (every other file is symlinked), which the build step then compiles.

Measured on GR00T-N1.7-3B / DROID (4 images, 2 denoising steps) on a Jetson AGX Orin: see
the "Round 3" section of ``INFERENCE_OPTIMIZATIONS_N1D7.md``.
"""

from __future__ import annotations

import gc
import logging
import os
import re

import numpy as np
import torch


logger = logging.getLogger(__name__)

INT8_COMPONENTS = ("vit", "llm", "action_head")

# SmoothQuant strength per component, from sweeps on GR00T-N1.7-3B / DROID.
SMOOTH_ALPHA = {"vit": 0.7, "llm": 0.8, "action_head": 0.8}

_FUSED_RE = re.compile(r"action_head_fused_\d+step\.onnx")
# LLM layers that stay BF16 under --int8 llm (set from a per-layer sensitivity study on
# GR00T-N1.7-3B; see INFERENCE_OPTIMIZATIONS_N1D7.md, round 4).
LLM_KEEP_BF16: str | None = r"mlp/down_proj"
# The ViT patch embedding (a Gemm over raw normalized pixels since round 5) costs about
# 0.4 points of mean action error in INT8 for ~0.1 ms; as a BF16 Gemm it is still far
# cheaper than the Conv3d it replaced.
VIT_KEEP_BF16: str | None = r"patch_embed"

# The DiT inside a fused action head graph; the other parts of the head stay BF16 unless
# ``quantize_action_head_all`` is set.
_FUSED_DIT_FILTER = "/model/"


def _calibration_steps(dataset, modality_configs, num_frames: int) -> list[tuple[int, int]]:
    """``num_frames`` (trajectory, step) pairs spread evenly over the whole dataset."""
    max_delta = max(
        max(cfg.delta_indices) for cfg in modality_configs.values() if cfg.delta_indices
    )
    min_delta = min(
        min(cfg.delta_indices) for cfg in modality_configs.values() if cfg.delta_indices
    )
    pairs = []
    for traj_idx in range(len(dataset)):
        length = len(dataset[traj_idx])
        first, last = max(0, -min_delta), length - max_delta - 1
        pairs += [(traj_idx, step) for step in range(first, last + 1)]
    if not pairs:
        raise ValueError("Dataset has no step with a full observation and action window")
    picks = np.linspace(0, len(pairs) - 1, min(num_frames, len(pairs))).round().astype(int)
    return [pairs[i] for i in sorted(set(picks.tolist()))]


def capture_calibration_data(
    policy, dataset, num_frames: int = 64, seed: int = 0
) -> dict[str, dict[str, list[torch.Tensor]]]:
    """Run ``policy`` (PyTorch) on dataset frames and record the inputs of each graph.

    Returns ``{"vit": {...}, "llm": {...}, "action_head_fused": {...}, "dit": {...}}``, each
    mapping the
    ONNX input names of that graph to one CPU tensor per sample.
    """
    from export_onnx_n1d7 import extract_step_data, parse_observation_gr00t

    model = policy.model
    action_head = model.action_head
    vision = model.backbone.model.model.visual
    records: dict[str, dict[str, list[torch.Tensor]]] = {
        "vit": {},
        "llm": {},
        "action_head_fused": {},
        "dit": {},
    }
    language_model = model.backbone.model.model.language_model

    def add(group, name, tensor):
        records[group].setdefault(name, []).append(tensor.detach().cpu().clone())

    def vit_hook(module, args, kwargs):
        pixel_values = args[0] if args else kwargs["hidden_states"]
        add("vit", "pixel_values", pixel_values)

    def llm_hook(module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return
        add("llm", "inputs_embeds", embeds.to(torch.bfloat16))
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 2:
            mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device=embeds.device)
        add("llm", "attention_mask", mask.to(torch.int64))
        add("llm", "position_ids", kwargs["position_ids"].to(torch.int64))
        visual = kwargs.get("visual_pos_masks")
        if visual is not None:
            add("llm", "visual_pos_masks", visual.to(torch.bool))
        for index, feature in enumerate(kwargs.get("deepstack_visual_embeds") or []):
            add("llm", f"deepstack_{index}", feature.to(torch.bfloat16))

    def dit_hook(module, args, kwargs):
        add("dit", "sa_embs", kwargs["hidden_states"])
        add("dit", "vl_embs", kwargs["encoder_hidden_states"])
        add("dit", "timestep", kwargs["timestep"].to(torch.int64))
        for name in ("image_mask", "backbone_attention_mask"):
            if kwargs.get(name) is not None:
                add("dit", name, kwargs[name].to(torch.bool))

    original_get_action = action_head.get_action
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def recording_get_action(backbone_output, action_input, *args, **kwargs):
        features = backbone_output.backbone_features
        state = action_input.state
        if state.ndim == 3 and state.shape[1] > 1:
            state = state.reshape(state.shape[0], 1, -1)
        noise = torch.randn(
            (features.shape[0], action_head.config.action_horizon, action_head.action_dim),
            generator=generator,
        )
        add("action_head_fused", "backbone_features", features.to(torch.bfloat16))
        add("action_head_fused", "state", state.to(torch.bfloat16))
        add("action_head_fused", "embodiment_id", action_input.embodiment_id.to(torch.int64))
        add("action_head_fused", "init_actions", noise.to(torch.bfloat16))
        image_mask = backbone_output.image_mask.to(torch.bool)
        attended = backbone_output.backbone_attention_mask.to(torch.bool)
        add("action_head_fused", "image_mask", image_mask)
        add("action_head_fused", "backbone_attention_mask", attended)
        # The current fused graph takes the token layout as index tensors (the graph
        # inputs decide which of these the calibration run feeds).
        add("action_head_fused", "image_index", (image_mask[0] & attended[0]).nonzero().squeeze(1))
        add("action_head_fused", "text_index", (~image_mask[0] & attended[0]).nonzero().squeeze(1))
        return original_get_action(backbone_output, action_input, *args, **kwargs)

    modality_configs = policy.get_modality_config()
    steps = _calibration_steps(dataset, modality_configs, num_frames)
    logger.info("Capturing INT8 calibration inputs on %d dataset frames...", len(steps))
    hooks = [
        vision.register_forward_pre_hook(vit_hook, with_kwargs=True),
        language_model.register_forward_pre_hook(llm_hook, with_kwargs=True),
        action_head.model.register_forward_pre_hook(dit_hook, with_kwargs=True),
    ]
    action_head.get_action = recording_get_action
    try:
        cached_traj, cached = None, None
        for traj_idx, step in steps:
            if traj_idx != cached_traj:
                cached_traj, cached = traj_idx, dataset[traj_idx]
            data = extract_step_data(
                cached,
                step,
                modality_configs=modality_configs,
                embodiment_tag=policy.embodiment_tag,
            )
            obs = {f"state.{k}": v for k, v in data.states.items()}
            obs.update({f"video.{k}": np.array(v) for k, v in data.images.items()})
            for key in modality_configs["language"].modality_keys:
                obs[key] = data.text
            with torch.inference_mode():
                policy.get_action(parse_observation_gr00t(obs, modality_configs))
    finally:
        for hook in hooks:
            hook.remove()
        del action_head.get_action  # restore the class method
    return records


def _mirror(src: str, dst: str, skip: set[str]) -> None:
    """Symlink every file of ``src`` into ``dst`` except the names in ``skip``.

    Anything already in ``dst`` under a skipped name (or its ``.data``) is removed, so a
    link left by an earlier run with a different component set can never be written
    through: that would overwrite the exported graph in ``src``.
    """
    os.makedirs(dst, exist_ok=True)
    for name in skip:
        for stale in (name, name + ".data"):
            path = os.path.join(dst, stale)
            if os.path.lexists(path):
                os.remove(path)
    for name in sorted(os.listdir(src)):
        out = os.path.join(dst, name)
        if name in skip or name[: -len(".data")] in skip:
            continue
        if os.path.lexists(out):
            os.remove(out)
        os.symlink(os.path.join(os.path.abspath(src), name), out)


def quantize_pipeline(
    policy,
    dataset,
    onnx_dir: str,
    int8_dir: str,
    components: tuple[str, ...],
    num_frames: int = 64,
    skip: str | None = None,
    quantize_action_head_all: bool = False,
    llm_keep_bf16: str | None = LLM_KEEP_BF16,
) -> dict[str, dict]:
    """Calibrate on ``dataset`` and write INT8 versions of ``components`` to ``int8_dir``.

    ``policy`` is released (moved off the GPU) after the capture so the calibration
    engines have the memory. ``llm_keep_bf16`` is the LLM's own keep-BF16 pattern
    (``None`` quantizes every LLM linear). Returns the per-file quantization summaries.
    """
    from quantize_onnx_int8 import quantize_file

    unknown = set(components) - set(INT8_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown --int8 component(s) {sorted(unknown)}; use {INT8_COMPONENTS}")
    if os.path.abspath(onnx_dir) == os.path.abspath(int8_dir):
        raise ValueError("The INT8 ONNX directory must differ from the export directory")

    names = sorted(os.listdir(onnx_dir))
    targets: dict[str, tuple[str, str | None, str | None, float]] = {}
    # file -> (calibration group, only-regex, component's own keep-BF16 regex, alpha)
    if "vit" in components:
        vit = next((n for n in ("vit_bf16.onnx", "vit_fp32.onnx") if n in names), None)
        if vit is None:
            raise FileNotFoundError(f"--int8 vit: no ViT graph in {onnx_dir}")
        targets[vit] = ("vit", None, VIT_KEEP_BF16, SMOOTH_ALPHA["vit"])
    if "llm" in components:
        if "llm_bf16.onnx" not in names:
            raise FileNotFoundError(f"--int8 llm: no llm_bf16.onnx in {onnx_dir}")
        targets["llm_bf16.onnx"] = ("llm", None, llm_keep_bf16, SMOOTH_ALPHA["llm"])
    if "action_head" in components:
        only = None if quantize_action_head_all else _FUSED_DIT_FILTER
        found = False
        for name in names:
            if _FUSED_RE.fullmatch(name):
                targets[name] = ("action_head_fused", only, None, SMOOTH_ALPHA["action_head"])
                found = True
        if "dit_bf16.onnx" in names:
            targets["dit_bf16.onnx"] = ("dit", None, None, SMOOTH_ALPHA["action_head"])
            found = True
        if not found:
            raise FileNotFoundError(f"--int8 action_head: no DiT or fused head graph in {onnx_dir}")

    records = capture_calibration_data(policy, dataset, num_frames)
    policy.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    _mirror(onnx_dir, int8_dir, skip=set(targets))
    summaries = {}
    for name, (group, only, keep_bf16, alpha) in targets.items():
        feeds = records[group]
        if not feeds:
            raise RuntimeError(f"No calibration inputs were captured for {name}")
        patterns = [p for p in (skip, keep_bf16) if p]
        summaries[name] = quantize_file(
            os.path.join(onnx_dir, name),
            os.path.join(int8_dir, name),
            feeds,
            only=only,
            skip="|".join(f"(?:{p})" for p in patterns) if patterns else None,
            smooth_alpha=alpha,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return summaries
