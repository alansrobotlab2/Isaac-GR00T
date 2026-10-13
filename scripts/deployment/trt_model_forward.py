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

"""TensorRT forward functions for GR00T N1.7 inference.

This module provides TRT-accelerated forward functions that replace the
PyTorch backbone and action head during inference.

Architecture (n17_full_pipeline mode):
  Backbone: ViT (TRT) → embed_tokens + masked_scatter + get_rope_index (PyTorch)
            → LLM (TRT, with deepstack injection)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]

Architecture (vit_llm_only mode):
  Backbone: ViT (TRT) → embed_tokens + masked_scatter + get_rope_index (PyTorch)
            → LLM (TRT, with deepstack injection)
  Action Head: stays in PyTorch
  Use when DiT cannot be exported with dynamic vl_seq_len (e.g. torch 2.10 / sm121).

Architecture (action_head mode):
  Backbone: stays in PyTorch (Qwen3-VL)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import logging
import os
import re
import sys

from gr00t.deployment.modes import InferenceMode
import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


# Ensure sibling modules are importable (scripts/deployment is not a package)
_deploy_dir = os.path.dirname(os.path.abspath(__file__))
if _deploy_dir not in sys.path:
    sys.path.insert(0, _deploy_dir)
from _trt_contract import (  # noqa: E402
    assert_engine_bundle_present,
    assert_engine_matches_policy,
    assert_grid_thw_matches,
)
from trt_torch import Engine  # noqa: E402


def _assert_supports_trt_padding_strip(attention_mask: torch.Tensor) -> None:
    """Hard-fail if the TRT padding-strip path would silently corrupt B>1 batches.

    transformers 4.57+ strips padding tokens before calling ``language_model``.
    The TRT forward path replicates that with ``valid_mask = attention_mask[0] == 1``
    so engine inputs match export-time captured shapes. With B=1 (or B>1 where
    every sample has the same mask, e.g. tiled benchmarking via
    ``verify_n1d7_trt.py --batch_size N``) this is correct. With B>1 *and*
    heterogeneous masks (different valid lengths per sample), samples 1..B-1
    would be silently mis-stripped using sample 0's padding template, producing
    wrong inputs to the LLM TRT engine — no error, no warning, just degraded
    output that's caught only via downstream eval drift.

    Raise rather than warn so the caller can't miss it in the multi-rank log
    torrent. To support heterogeneous B>1, switch to the all-batch
    common-padding mask ``(attention_mask == 1).all(dim=0)`` (only strips
    columns padded across *all* samples) — that is a semantic change requiring
    modeling sign-off, so we hard-fail here for now.

    Note on the build-time batch coupling: the ONNX export pipeline
    (``scripts/deployment/export_onnx_n1d7.py``) bakes the batch dim as a
    static shape — only ``seq_len`` is registered in ``dynamic_axes`` —
    so a TRT engine built at ``--batch-size N`` only accepts B=N at
    runtime. Lifting heterogeneous B>1 inference therefore requires both
    rebuilding the engine at that batch size *and* relaxing this
    assertion. Production today is B=1; this check covers the runtime
    half of that contract.
    """
    if attention_mask.shape[0] <= 1:
        return
    if not (attention_mask == attention_mask[0:1]).all():
        raise ValueError(
            "TRT backbone padding-strip requires B=1 or homogeneous attention "
            f"masks across the batch (got B={attention_mask.shape[0]} with "
            "heterogeneous masks). attention_mask[0] would silently mis-strip "
            "samples 1..B-1 using sample 0's padding template, producing wrong "
            "inputs to the LLM TRT engine. Switch to "
            "(attention_mask == 1).all(dim=0) to only strip columns padded "
            "across all samples — but that is a semantic change requiring "
            "modeling sign-off."
        )


def _resolve_vit_engine_path(trt_engine_path: str) -> str:
    """Locate the ViT engine, tolerating the legacy filename.

    Older builds emitted ``vit_bf16.engine`` even when the source ONNX
    was FP32 — a name that obscured the actual engine precision. New
    builds emit ``vit.engine`` (precision-neutral; the real dtype lives
    in ``export_metadata.json`` and the ONNX tensor types). Existing
    engine directories with the legacy name are still accepted with a
    warning emitted on each call (no per-process suppression: the
    warning content is path-dependent and call sites are few), to
    encourage migration to the precision-neutral name. When no engine
    is present we return the canonical name so the caller can put it
    in any "not found" error.
    """
    new = os.path.join(trt_engine_path, "vit.engine")
    legacy = os.path.join(trt_engine_path, "vit_bf16.engine")
    if os.path.exists(new):
        return new
    if os.path.exists(legacy):
        logger.warning(
            f"Loading ViT engine from legacy path {legacy}; new builds "
            f"emit vit.engine. Rebuild the pipeline to drop the misleading "
            f"precision tag from the filename."
        )
        return legacy
    return new


# ============================================================
# N1.7 Backbone TRT Forward (ViT TRT + LLM TRT)
# ============================================================


# Token layouts (prompt + image grid) to keep a static plan for, per backbone.
_STATIC_PLAN_CACHE_SIZE = 8


@dataclass
class _StaticPlan:
    """Backbone glue that depends only on the token layout, not on pixel or state values.

    For a fixed prompt and camera setup every field is identical from step to step, but
    the transformers helpers that produce them (``get_placeholder_mask``,
    ``get_rope_index``) run host-side loops and read GPU tensors back, which stalls the
    engine pipeline. They are computed once per layout and replayed afterwards.
    """

    visual_index: torch.Tensor  # [B * num_vis_tokens] flat positions of image tokens
    visual_pos_masks: torch.Tensor  # [B, seq_len] bool, padding stripped
    position_ids: torch.Tensor  # [3, B, seq_len] int64, padding stripped
    rope_deltas: torch.Tensor
    attention_mask: torch.Tensor  # [B, seq_len] int64, padding stripped
    valid_mask: torch.Tensor | None  # [full_seq_len] bool, None when there is no padding
    image_mask_out: torch.Tensor  # [B, seq_len] bool
    backbone_attention_mask: torch.Tensor  # [B, seq_len] bool


def _static_plan_key(input_ids, attention_mask, grid_thw):
    """Hashable identity of the token layout, from a single GPU-to-host read."""
    flat = torch.cat(
        [
            input_ids.reshape(-1),
            attention_mask.reshape(-1).to(input_ids.dtype),
            grid_thw.reshape(-1).to(input_ids.dtype),
        ]
    )
    return (tuple(input_ids.shape), tuple(grid_thw.shape), flat.cpu().numpy().tobytes())


def _build_static_plan(self, input_ids, attention_mask, grid_thw, inputs_embeds, image_embeds):
    """Run the transformers glue once for this token layout and record its results."""
    inner_model = self.model.model  # Qwen3VLModel

    # The ViT engine baked pos/rotary buffers for the export-time grid; a
    # different runtime grid would silently corrupt vision features.
    assert_grid_thw_matches(
        getattr(self, "_vit_baked_grid_thw", None), grid_thw, source="ViT TRT forward"
    )

    # Validates that the image token count matches the ViT output.
    image_mask, _ = inner_model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    visual_pos_masks = image_mask[..., 0]

    # Compute 3D position IDs (complex Python logic in transformers)
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, grid_thw, video_grid_thw=None, attention_mask=attention_mask
    )

    image_mask_out = input_ids == self._image_token_id
    backbone_attention_mask = attention_mask == 1

    # transformers 4.57+ strips padding tokens before calling language_model internally.
    # Apply the same stripping so TRT engine inputs match export-time captured shapes.
    _assert_supports_trt_padding_strip(attention_mask)
    valid_mask = attention_mask[0] == 1  # [full_seq_len]
    stripped_visual_pos_masks = visual_pos_masks
    if valid_mask.all():
        valid_mask = None
    else:
        attention_mask = attention_mask[:, valid_mask]
        position_ids = position_ids[:, :, valid_mask]
        stripped_visual_pos_masks = visual_pos_masks[:, valid_mask]
        image_mask_out = image_mask_out[:, valid_mask]
        backbone_attention_mask = backbone_attention_mask[:, valid_mask]

    return _StaticPlan(
        # masked_scatter fills masked positions in row-major order, i.e. these indices.
        visual_index=visual_pos_masks.reshape(-1).nonzero().squeeze(1),
        visual_pos_masks=stripped_visual_pos_masks.contiguous(),
        position_ids=position_ids.to(torch.int64).contiguous(),
        rope_deltas=rope_deltas,
        attention_mask=attention_mask.to(torch.int64).contiguous(),
        valid_mask=valid_mask,
        image_mask_out=image_mask_out.contiguous(),
        backbone_attention_mask=backbone_attention_mask.contiguous(),
    )


def _qwen3_vit_and_scatter(self, vl_input):
    """Shared logic: ViT TRT + embed_tokens + scatter + get_rope_index.

    Returns all inputs needed by either PyTorch LLM or LLM TRT engine.
    These ops stay in PyTorch because they involve dynamic Python logic
    (get_rope_index, masked_scatter, get_placeholder_mask); everything that depends
    only on the token layout is cached in a :class:`_StaticPlan`.
    """
    inner_model = self.model.model  # Qwen3VLModel

    pixel_values = vl_input["pixel_values"]
    grid_thw = vl_input["image_grid_thw"]
    input_ids = vl_input["input_ids"]
    attention_mask = vl_input["attention_mask"]
    engine_dtype = torch.bfloat16

    # Look the layout up before any engine is enqueued: the key is a host read of GPU
    # data, which would otherwise wait for the ViT to finish.
    plans = self.__dict__.setdefault("_trt_static_plans", OrderedDict())
    key = _static_plan_key(input_ids, attention_mask, grid_thw)
    plan = plans.get(key)

    # --- ViT TRT Engine ---
    # Detect ViT engine dtype (FP32 for accuracy or BF16 for speed)
    vit_dtype = self.vit_engine.dtype_of("pixel_values")
    if isinstance(pixel_values, (list, tuple)):
        pv = torch.cat(pixel_values, dim=0)
    else:
        pv = pixel_values
    if pv.dtype != vit_dtype:
        pv = pv.to(vit_dtype)

    self.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
    vit_result = self.vit_engine(pv)
    image_embeds = vit_result["image_embeds"]
    deepstack_features = vit_result.get("deepstack_features")

    # Unpack deepstack: [num_layers, N, D] → list of [N, D]
    deepstack_list = []
    if deepstack_features is not None and deepstack_features.numel() > 1:
        deepstack_list = list(deepstack_features.unbind(0))

    # --- PyTorch: embed_tokens + scatter ---
    inputs_embeds = self._embedding_layer(input_ids)

    if inputs_embeds.dtype != engine_dtype:
        inputs_embeds = inputs_embeds.to(engine_dtype)
    if image_embeds.dtype != engine_dtype:
        image_embeds = image_embeds.to(engine_dtype)

    if plan is None:
        plan = _build_static_plan(
            self, input_ids, attention_mask, grid_thw, inputs_embeds, image_embeds
        )
        if len(plans) >= _STATIC_PLAN_CACHE_SIZE:
            plans.popitem(last=False)
        plans[key] = plan
    else:
        plans.move_to_end(key)

    # Equivalent to inputs_embeds.masked_scatter(image_mask, image_embeds).
    hidden = inputs_embeds.shape[-1]
    inputs_embeds = (
        inputs_embeds.reshape(-1, hidden)
        .index_copy(0, plan.visual_index, image_embeds)
        .reshape(inputs_embeds.shape)
    )
    inner_model.rope_deltas = plan.rope_deltas

    if plan.valid_mask is not None:
        inputs_embeds = inputs_embeds[:, plan.valid_mask, :]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": plan.attention_mask,
        "position_ids": plan.position_ids,
        "visual_pos_masks": plan.visual_pos_masks,
        "deepstack_list": deepstack_list,
        "image_mask_out": plan.image_mask_out,
        "backbone_attention_mask": plan.backbone_attention_mask,
    }


def qwen3_backbone_tensorrt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + PyTorch LLM.

    ViT is replaced with a TRT engine. The LLM stays in PyTorch.
    Used when LLM TRT engine is not available.

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    qwen_model = self.model
    inner_model = qwen_model.model

    # LLM forward (PyTorch)
    outputs = inner_model.language_model(
        input_ids=None,
        position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"],
        inputs_embeds=prepared["inputs_embeds"],
        visual_pos_masks=prepared["visual_pos_masks"],
        deepstack_visual_embeds=prepared["deepstack_list"] or None,
        output_hidden_states=True,
    )

    return BatchFeature(
        data={
            "backbone_features": outputs.last_hidden_state,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


def qwen3_backbone_llm_trt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with PyTorch ViT + LLM TRT.

    ViT stays in PyTorch. LLM is replaced with a TRT engine.
    Used when ViT TRT has accuracy issues but LLM TRT is accurate.
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    # Run PyTorch ViT + scatter + rope (original backbone logic up to LLM)
    qwen_model = self.model
    inner_model = qwen_model.model

    # ViT forward (PyTorch — kept for accuracy)
    pixel_values = vl_input["pixel_values"]
    grid_thw = vl_input["image_grid_thw"]
    image_embeds_split, deepstack_image_embeds = inner_model.get_image_features(
        pixel_values, grid_thw
    )
    # get_image_features returns a tuple of per-image tensors; concat for scatter
    image_embeds = torch.cat(list(image_embeds_split), dim=0)

    # Scatter image embeddings into text embeddings
    input_ids = vl_input["input_ids"]
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_mask, _ = inner_model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
    deepstack_list = list(deepstack_image_embeds) if deepstack_image_embeds else []

    # Compute position IDs
    attention_mask = vl_input["attention_mask"]
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, grid_thw, video_grid_thw=None, attention_mask=attention_mask
    )
    inner_model.rope_deltas = rope_deltas

    image_mask_out = input_ids == qwen_model.config.image_token_id
    backbone_attention_mask = attention_mask == 1

    # Strip padding tokens (transformers 4.57+)
    _assert_supports_trt_padding_strip(attention_mask)
    valid_mask = attention_mask[0] == 1
    if not valid_mask.all():
        inputs_embeds = inputs_embeds[:, valid_mask, :]
        attention_mask = attention_mask[:, valid_mask]
        position_ids = position_ids[:, :, valid_mask]
        if visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks[:, valid_mask]
        image_mask_out = image_mask_out[:, valid_mask]
        backbone_attention_mask = backbone_attention_mask[:, valid_mask]

    # LLM forward (TRT)
    llm_float_dtype = self.llm_engine.dtype_of("inputs_embeds")

    if inputs_embeds.dtype != llm_float_dtype:
        inputs_embeds = inputs_embeds.to(llm_float_dtype)
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)

    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)

    llm_kwargs = {}
    if visual_pos_masks is not None and deepstack_list:
        self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
        llm_kwargs["visual_pos_masks"] = visual_pos_masks
        for i, ds in enumerate(deepstack_list):
            name = f"deepstack_{i}"
            if ds.dtype != llm_float_dtype:
                ds = ds.to(llm_float_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, ds.shape)
            llm_kwargs[name] = ds

    backbone_features = self.llm_engine(inputs_embeds, attention_mask, position_ids, **llm_kwargs)[
        "embeddings"
    ]

    if backbone_features.dtype != torch.bfloat16:
        backbone_features = backbone_features.to(torch.bfloat16)

    return BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask_out,
        }
    )


def qwen3_backbone_full_trt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + LLM TRT.

    Both ViT and LLM are replaced with TRT engines.
    PyTorch ops kept: embed_tokens, masked_scatter, get_rope_index (lightweight).

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    inputs_embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    position_ids = prepared["position_ids"]

    # Detect LLM engine's expected float dtype from its first input binding.
    # Handles both BF16 engines (default) and FP32 engines gracefully.
    llm_float_dtype = self.llm_engine.dtype_of("inputs_embeds")

    if inputs_embeds.dtype != llm_float_dtype:
        inputs_embeds = inputs_embeds.to(llm_float_dtype)
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)

    # Set LLM engine input shapes
    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)

    llm_kwargs = {}

    # Visual pos masks and deepstack features
    visual_pos_masks = prepared["visual_pos_masks"]
    deepstack_list = prepared["deepstack_list"]

    if visual_pos_masks is not None and deepstack_list:
        self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
        llm_kwargs["visual_pos_masks"] = visual_pos_masks

        for i, ds in enumerate(deepstack_list):
            name = f"deepstack_{i}"
            if ds.dtype != llm_float_dtype:
                ds = ds.to(llm_float_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, ds.shape)
            llm_kwargs[name] = ds

    backbone_features = self.llm_engine(inputs_embeds, attention_mask, position_ids, **llm_kwargs)[
        "embeddings"
    ]

    # Cast LLM output back to BF16 — downstream (vl_self_attention, DiT) expect BF16.
    if backbone_features.dtype != torch.bfloat16:
        backbone_features = backbone_features.to(torch.bfloat16)

    return BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


# ============================================================
# Action Head TRT Forward
# ============================================================


_FUSED_ENGINE_RE = re.compile(r"action_head_fused_(\d+)step\.engine")


def _find_fused_engines(trt_engine_path) -> dict[int, str]:
    """Map denoising-step count -> fused action-head engine found in the engine directory."""
    found = {}
    for name in os.listdir(trt_engine_path):
        match = _FUSED_ENGINE_RE.fullmatch(name)
        if match:
            found[int(match.group(1))] = os.path.join(trt_engine_path, name)
    return found


def _load_step_engines(action_head):
    """Load the per-component action-head engines (the path used without a fused engine)."""
    trt_engine_path, engine_kwargs = action_head._trt_engine_source
    print(f"Loading action head engines from: {trt_engine_path}")
    vl_sa_engine_path = os.path.join(trt_engine_path, "vl_self_attention.engine")
    action_head.vl_sa_engine = (
        Engine(vl_sa_engine_path, **engine_kwargs) if os.path.exists(vl_sa_engine_path) else None
    )
    action_head.state_encoder_engine = Engine(
        os.path.join(trt_engine_path, "state_encoder.engine"), **engine_kwargs
    )
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine"), **engine_kwargs
    )
    action_head.dit_engine = Engine(
        os.path.join(trt_engine_path, "dit_bf16.engine"), **engine_kwargs
    )
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine"), **engine_kwargs
    )
    action_head._trt_step_engines_loaded = True


def _fused_engine_for(action_head, num_steps):
    """Fused engine for this step count, loaded on first use; None if there is none."""
    engines = action_head.__dict__.setdefault("_trt_fused_engines", {})
    if num_steps not in engines:
        path = action_head.__dict__.get("_trt_fused_engine_paths", {}).get(num_steps)
        if path is None:
            return None
        print(f"Loading fused action head engine: {path}")
        engines[num_steps] = Engine(path, **action_head._trt_engine_source[1])
    return engines[num_steps]


def _initial_actions(self, batch_size, device, dtype):
    if hasattr(self, "init_actions"):
        return self.init_actions.expand((batch_size, -1, -1))
    return torch.randn(
        size=(batch_size, self.config.action_horizon, self.action_dim), dtype=dtype, device=device
    )


def _flatten_state_history(state):
    """[B, state_history_length, max_state_dim] -> [B, 1, state_history_length * max_state_dim]."""
    if state.ndim == 3 and state.shape[1] > 1:
        return state.view(state.shape[0], 1, -1)
    if not (state.ndim == 3 and state.shape[1] == 1):
        logger.warning(f"Unexpected state shape: {state.shape}")
    return state


def _fused_action_head_forward(self, engine, backbone_output, action_input):
    """Run the whole action head (VLLN through the last Euler step) as one engine."""
    engine_dtype = torch.bfloat16
    backbone_features = backbone_output.backbone_features
    if backbone_features.dtype != engine_dtype:
        backbone_features = backbone_features.to(engine_dtype)
    state = action_input.state
    if state.dtype != engine_dtype:
        state = state.to(engine_dtype)
    state = _flatten_state_history(state)
    embodiment_id = action_input.embodiment_id
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)
    init_actions = _initial_actions(
        self, backbone_features.shape[0], backbone_features.device, engine_dtype
    )
    image_mask = backbone_output.image_mask
    backbone_attention_mask = backbone_output.backbone_attention_mask

    engine.set_runtime_tensor_shape("backbone_features", backbone_features.shape)
    engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
    engine.set_runtime_tensor_shape("backbone_attention_mask", backbone_attention_mask.shape)
    actions = engine(
        backbone_features=backbone_features,
        state=state,
        embodiment_id=embodiment_id,
        init_actions=init_actions,
        image_mask=image_mask,
        backbone_attention_mask=backbone_attention_mask,
    )["actions"]
    return BatchFeature(data={"action_pred": actions})


def action_head_tensorrt_forward(self, backbone_output, action_input, options=None):
    """Replace ActionHead.get_action() with TRT-accelerated inference.
    VLLN (LayerNorm) stays in PyTorch. State Encoder, Action Encoder,
    DiT, and Action Decoder are replaced with TRT engines.

    N1.7 change: state is reshaped from [B, state_history_length, max_state_dim]
    to [B, 1, state_history_length * max_state_dim] before the state encoder.

    Args:
        self: ActionHead instance (monkey-patched)
        backbone_output: BatchFeature with backbone_features, backbone_attention_mask, image_mask
        action_input: BatchFeature with state, embodiment_id

    When the engine directory holds a fused engine for the current number of denoising
    steps (``action_head_fused_<N>step.engine``), the whole head runs as that one engine.
    """
    if "_trt_engine_source" in self.__dict__:
        fused_engine = _fused_engine_for(self, self.num_inference_timesteps)
        if fused_engine is not None:
            return _fused_action_head_forward(self, fused_engine, backbone_output, action_input)
        if not self.__dict__.get("_trt_step_engines_loaded", False):
            _load_step_engines(self)

    # --- VLLN (PyTorch) + vl_self_attention (TRT if available, else PyTorch) ---
    backbone_features = backbone_output.backbone_features
    backbone_features = self.vlln(backbone_features)
    if hasattr(self, "vl_sa_engine") and self.vl_sa_engine is not None:
        engine_dtype = torch.bfloat16
        if backbone_features.dtype != engine_dtype:
            backbone_features = backbone_features.to(engine_dtype)
        self.vl_sa_engine.set_runtime_tensor_shape("hidden_states", backbone_features.shape)
        backbone_features = self.vl_sa_engine(backbone_features)["output"]
    else:
        backbone_features = self.vl_self_attention(backbone_features)
    vl_embs = backbone_features

    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]
    device = vl_embs.device

    engine_dtype = torch.bfloat16

    # Ensure consistent dtypes
    if vl_embs.dtype != engine_dtype:
        vl_embs = vl_embs.to(engine_dtype)
    if action_input.state.dtype != engine_dtype:
        action_input.state = action_input.state.to(engine_dtype)
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    # --- State history reshape (N1.7) ---
    state = _flatten_state_history(action_input.state)

    # --- State Encoder TRT ---
    self.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(state, embodiment_id)["output"]

    # --- Initialize actions as random noise ---
    actions = _initial_actions(self, batch_size, device, engine_dtype)

    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    # Timestep and position-embedding tensors are the same on every call; build them once.
    loop_key = (num_steps, batch_size, str(device))
    loop_cache = self.__dict__.get("_trt_loop_cache")
    if loop_cache is None or loop_cache[0] != loop_key:
        timestep_tensors = [
            torch.full(
                size=(batch_size,),
                fill_value=int(t / float(num_steps) * self.num_timestep_buckets),
                device=device,
                dtype=torch.int64,
            )
            for t in range(num_steps)
        ]
        pos_embs = None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(self.config.action_horizon, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(engine_dtype)
        loop_cache = (loop_key, timestep_tensors, pos_embs)
        self.__dict__["_trt_loop_cache"] = loop_cache
    _, timestep_tensors, pos_embs = loop_cache

    # --- Denoising loop ---
    for timesteps_tensor in timestep_tensors:
        # Action Encoder TRT
        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("timesteps", timesteps_tensor.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(
            actions.to(engine_dtype), timesteps_tensor, embodiment_id
        )["output"]

        # Maybe add position embedding (stays in PyTorch)
        if pos_embs is not None:
            action_features = action_features + pos_embs

        # Concatenate state + action embeddings
        sa_embs = torch.cat((state_features, action_features), dim=1).to(engine_dtype)

        # DiT TRT
        self.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("timestep", timesteps_tensor.shape)

        dit_kwargs = {}
        if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
            image_mask = backbone_output.image_mask
            self.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
            dit_kwargs["image_mask"] = image_mask

        if (
            hasattr(backbone_output, "backbone_attention_mask")
            and backbone_output.backbone_attention_mask is not None
        ):
            bb_mask = backbone_output.backbone_attention_mask
            self.dit_engine.set_runtime_tensor_shape("backbone_attention_mask", bb_mask.shape)
            dit_kwargs["backbone_attention_mask"] = bb_mask

        model_output = self.dit_engine(sa_embs, vl_embs, timesteps_tensor, **dit_kwargs)["output"]

        # Action Decoder TRT
        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        pred_velocity = pred[:, -self.action_horizon :]

        # Euler integration
        actions = actions + dt * pred_velocity

    return BatchFeature(data={"action_pred": actions})


# ============================================================
# Engine Setup
# ============================================================


# Engines each mode swaps in with no PyTorch fallback: their absence must fail
# fast with a build hint rather than a bare FileNotFoundError from Engine.load.
_ACTION_HEAD_ENGINES = (
    "state_encoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "action_decoder.engine",
)
_MANDATORY_ENGINES = {
    InferenceMode.n17_full_pipeline: _ACTION_HEAD_ENGINES,
    InferenceMode.action_head: _ACTION_HEAD_ENGINES,
}


def setup_tensorrt_engines(
    policy,
    trt_engine_path,
    mode=InferenceMode.n17_full_pipeline,
    use_vit_engine=True,
    sync=False,
    cuda_graph=False,
):
    """Load TRT engines, delete PyTorch modules, and monkey-patch forward methods.

    Args:
        policy: Gr00tPolicy instance
        trt_engine_path: Path to directory containing TRT engine files
        mode: an :class:`~gr00t.deployment.modes.InferenceMode` (or its string
            value) selecting which engine subset to swap in. Invalid values raise.
        use_vit_engine: ``n17_full_pipeline`` only. False keeps the ViT in PyTorch
            even when a ViT engine is present (LLM + action head still run in TRT).
            The ViT engine is FP32, which on Jetson Orin is slower than the BF16
            PyTorch ViT with flash attention.
        sync: Block on the CUDA stream after every engine call. The default ``False``
            lets the engines queue back to back like ordinary PyTorch modules; results
            are identical, and reading the actions back still waits for them.
        cuda_graph: Replay each engine through a captured CUDA graph (see
            :class:`trt_torch.Engine`). Lower host overhead and identical results;
            the first inference is slower while the graphs are captured.
    """
    mode = InferenceMode(mode)
    engine_kwargs = {"sync": sync, "cuda_graph": cuda_graph}
    required = _MANDATORY_ENGINES.get(mode)
    if required is not None:
        assert_engine_bundle_present(
            trt_engine_path,
            required,
            mode=mode.value,
            source=f"setup_tensorrt_engines({mode})",
        )
    _meta = assert_engine_matches_policy(
        policy, trt_engine_path, source=f"setup_tensorrt_engines({mode})"
    )

    # Stash the ViT engine's baked grid_thw so the runtime ViT path can reject a
    # mismatched image config. None for older bundles (no recorded grid).
    policy.model.backbone._vit_baked_grid_thw = _meta.get("vit_grid_thw") if _meta else None

    if mode == InferenceMode.n17_full_pipeline:
        _setup_n17_full_pipeline(
            policy, trt_engine_path, use_vit_engine=use_vit_engine, engine_kwargs=engine_kwargs
        )
    else:
        _INFERENCE_MODE_DISPATCH[mode](policy, trt_engine_path, engine_kwargs=engine_kwargs)


def close_tensorrt_engines(policy):
    """Release every TRT ``Engine`` attached to ``policy`` (best-effort).

    setup_tensorrt_engines stashes engines as plain attributes on the backbone
    and action head (which subset depends on the mode), so walk the model's
    submodules and close each. Needed because the sim-eval entrypoint hard-exits
    via os._exit, which skips ``Engine.__del__`` and would otherwise leak GPU
    memory into the next eval shard.
    """
    # The sim wrapper exposes the inner Gr00tPolicy as ``.policy``; a remote
    # PolicyClient has no model and no engines to close.
    model = getattr(getattr(policy, "policy", policy), "model", None)
    if model is None:
        return
    closed = set()
    for module in model.modules():
        # Snapshot the attribute values: close() runs C++ destructors, so avoid
        # iterating a live __dict__ view.
        values = []
        for value in list(vars(module).values()):
            # Fused action-head engines are kept in a dict keyed by step count.
            values.extend(value.values() if isinstance(value, dict) else [value])
        for value in values:
            if isinstance(value, Engine) and id(value) not in closed:
                closed.add(id(value))
                try:
                    value.close()
                except Exception as e:
                    print(f"Failed to close TRT engine: {e}")


@contextmanager
def closing_tensorrt_engines(policy):
    """``with`` form of :func:`close_tensorrt_engines`, scoped to ``policy``.

    Closes only the engines attached to ``policy`` on block exit (including the
    exception path), so it never touches engines owned by another live policy.
    """
    try:
        yield policy
    finally:
        close_tensorrt_engines(policy)


def _setup_n17_full_pipeline(policy, trt_engine_path, use_vit_engine=True, engine_kwargs=None):
    """Set up TRT engines for N1.7: ViT TRT + LLM TRT + Action Head TRT.

    The Qwen3-VL backbone's vision encoder and text model are both replaced
    with TRT engines. PyTorch ops kept: embed_tokens, masked_scatter,
    get_rope_index (lightweight, <1ms).

    Falls back to PyTorch LLM if llm_bf16.engine is not found.
    """
    backbone = policy.model.backbone
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration
    action_head = policy.model.action_head

    # --- Backbone setup ---
    # Save references needed by the TRT forward
    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id

    # Load ViT TRT engine (optional — PyTorch ViT used as fallback for accuracy)
    vit_engine_path = _resolve_vit_engine_path(trt_engine_path)
    use_vit_trt = use_vit_engine and os.path.exists(vit_engine_path)
    if use_vit_trt:
        print(f"Loading ViT engine: {vit_engine_path}")
        backbone.vit_engine = Engine(vit_engine_path, **(engine_kwargs or {}))
        del qwen_model.model.visual
        torch.cuda.empty_cache()
        print("  Deleted PyTorch ViT (replaced by TRT engine)")
    elif not use_vit_engine:
        backbone.vit_engine = None
        print("  ViT engine disabled (use_vit_engine=False), keeping PyTorch ViT")
    else:
        backbone.vit_engine = None
        print(f"  ViT engine not found at {vit_engine_path}, keeping PyTorch ViT")

    # Load LLM TRT engine (if available)
    llm_engine_path = os.path.join(trt_engine_path, "llm_bf16.engine")
    use_llm_trt = os.path.exists(llm_engine_path)

    if use_llm_trt:
        print(f"Loading LLM engine: {llm_engine_path}")
        backbone.llm_engine = Engine(llm_engine_path, **(engine_kwargs or {}))

        # Delete PyTorch LLM layers to free GPU memory
        # Keep embed_tokens (needed for token embedding before TRT)
        # Keep get_rope_index via inner_model (needed for position IDs)
        del qwen_model.model.language_model.layers
        del qwen_model.model.language_model.norm
        torch.cuda.empty_cache()
        print("  Deleted PyTorch LLM layers (replaced by TRT engine)")
    else:
        backbone.llm_engine = None
        print(f"  LLM engine not found at {llm_engine_path}, using PyTorch LLM")

    # Monkey-patch backbone forward
    if use_vit_trt and use_llm_trt:
        backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)
    elif use_vit_trt and not use_llm_trt:
        backbone.forward = partial(qwen3_backbone_tensorrt_forward, backbone)
    elif not use_vit_trt and use_llm_trt:
        # PyTorch ViT + LLM TRT (best accuracy when ViT TRT has issues)
        backbone.forward = partial(qwen3_backbone_llm_trt_forward, backbone)
    else:
        print("  No backbone TRT engines loaded, backbone remains in PyTorch")

    # --- Action head setup ---
    # The PyTorch modules are replaced either by a fused engine (one per denoising-step
    # count, optional) or by the per-component engines.
    vl_sa_engine_path = os.path.join(trt_engine_path, "vl_self_attention.engine")
    if os.path.exists(vl_sa_engine_path) and hasattr(action_head, "vl_self_attention"):
        del action_head.vl_self_attention
        print("  Deleted PyTorch vl_self_attention (replaced by TRT engine)")
    for module_name in ("model", "state_encoder", "action_encoder", "action_decoder"):
        if hasattr(action_head, module_name):
            delattr(action_head, module_name)
    torch.cuda.empty_cache()

    assert action_head.action_dim == action_head.config.max_action_dim

    action_head._trt_engine_source = (trt_engine_path, dict(engine_kwargs or {}))
    action_head._trt_fused_engine_paths = _find_fused_engines(trt_engine_path)
    action_head._trt_fused_engines = {}
    action_head._trt_step_engines_loaded = False
    num_steps = action_head.num_inference_timesteps
    if _fused_engine_for(action_head, num_steps) is None:
        _load_step_engines(action_head)
        action_head_status = "TRT"
    else:
        # The per-component engines (about 3 GB) are only loaded if the step count
        # is later changed to one without a fused engine.
        action_head_status = f"TRT (fused, {num_steps} denoising steps)"

    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    llm_status = "TRT" if use_llm_trt else "PyTorch"
    vit_status = "TRT" if backbone.vit_engine else "PyTorch"
    print("N1.7 full-pipeline TRT engines loaded.")
    print(f"  ViT: {vit_status} | LLM: {llm_status} | Action Head: {action_head_status}")


def _setup_vit_llm_only(policy, trt_engine_path, engine_kwargs=None):
    """Set up TRT engines for ViT + LLM only; action head stays in PyTorch.

    Use this on platforms where DiT cannot be exported with dynamic vl_seq_len
    (e.g. DGX Spark / torch 2.10 dynamo exporter bakes seq_len as static).
    The backbone (ViT TRT + LLM TRT) still gets TRT acceleration; the PyTorch
    action head receives the LLM embeddings at the actual runtime seq_len
    without any shape constraint.
    """
    backbone = policy.model.backbone
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration

    # Save references needed by the TRT forward
    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id

    # Load ViT TRT engine
    vit_engine_path = _resolve_vit_engine_path(trt_engine_path)
    if not os.path.exists(vit_engine_path):
        raise FileNotFoundError(
            f"ViT TRT engine not found: {vit_engine_path}\n"
            f"Run export_onnx_n1d7.py + build_tensorrt_engine.py first."
        )
    print(f"Loading ViT engine: {vit_engine_path}")
    backbone.vit_engine = Engine(vit_engine_path, **(engine_kwargs or {}))
    del qwen_model.model.visual
    torch.cuda.empty_cache()
    print("  Deleted PyTorch ViT (replaced by TRT engine)")

    # Load LLM TRT engine
    llm_engine_path = os.path.join(trt_engine_path, "llm_bf16.engine")
    if not os.path.exists(llm_engine_path):
        raise FileNotFoundError(
            f"LLM TRT engine not found: {llm_engine_path}\n"
            f"Run export_onnx_n1d7.py + build_tensorrt_engine.py first."
        )
    print(f"Loading LLM engine: {llm_engine_path}")
    backbone.llm_engine = Engine(llm_engine_path, **(engine_kwargs or {}))
    del qwen_model.model.language_model.layers
    del qwen_model.model.language_model.norm
    torch.cuda.empty_cache()
    print("  Deleted PyTorch LLM layers (replaced by TRT engine)")

    # Patch backbone forward to use ViT TRT + LLM TRT
    backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)

    print("vit_llm_only TRT engines loaded.")
    print("  ViT: TRT | LLM: TRT | Action Head: PyTorch")


def _setup_action_head(policy, trt_engine_path, engine_kwargs=None):
    """Set up TRT engines for action head only (N1.7 mode).

    Backbone (Qwen3-VL) stays in PyTorch. Only the 4 action head components
    (State Encoder, Action Encoder, DiT, Action Decoder) are replaced with
    TRT engines.
    """
    action_head = policy.model.action_head

    # Delete PyTorch modules that are replaced by TRT
    if hasattr(action_head, "model"):
        del action_head.model
    if hasattr(action_head, "state_encoder"):
        del action_head.state_encoder
    if hasattr(action_head, "action_encoder"):
        del action_head.action_encoder
    if hasattr(action_head, "action_decoder"):
        del action_head.action_decoder
    torch.cuda.empty_cache()

    # Verify action_dim consistency
    assert action_head.action_dim == action_head.config.max_action_dim, (
        f"action_dim mismatch: action_head.action_dim={action_head.action_dim} "
        f"!= config.max_action_dim={action_head.config.max_action_dim}"
    )

    # Load action head TRT engines
    print(f"Loading action head engines from: {trt_engine_path}")
    action_head.state_encoder_engine = Engine(
        os.path.join(trt_engine_path, "state_encoder.engine"), **(engine_kwargs or {})
    )
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine"), **(engine_kwargs or {})
    )
    action_head.dit_engine = Engine(
        os.path.join(trt_engine_path, "dit_bf16.engine"), **(engine_kwargs or {})
    )
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine"), **(engine_kwargs or {})
    )

    # Monkey-patch: backbone.forward stays original, only action head is replaced
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    print("Action head TRT engines loaded and forward method patched.")
    print("  Backbone remains in PyTorch (Qwen3-VL).")


def _setup_dit_only(policy, trt_engine_path, engine_kwargs=None):
    """Set up TRT engine for DiT-only acceleration (backward compatible).

    Only replaces the DiT model in the action head. The backbone and other
    action head components remain in PyTorch.
    """
    action_head = policy.model.action_head

    # Delete the PyTorch DiT model
    if hasattr(action_head, "model"):
        del action_head.model
    torch.cuda.empty_cache()

    # Load DiT TRT engine
    # Support both naming conventions
    dit_path = os.path.join(trt_engine_path, "dit_bf16.engine")
    if not os.path.exists(dit_path):
        dit_path = os.path.join(trt_engine_path, "dit_model_bf16.engine")
    if not os.path.exists(dit_path):
        # Try the old naming convention
        dit_path = os.path.join(trt_engine_path, "dit_model_bf16.trt")

    print(f"Loading DiT engine: {dit_path}")
    action_head.dit_engine = Engine(dit_path, **(engine_kwargs or {}))

    # Monkey-patch only the get_action method
    # We need a simpler forward that only replaces the DiT call
    @torch.no_grad()
    def dit_only_get_action_with_features(
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input=None,
        options=None,
    ):
        """get_action_with_features with DiT replaced by TRT.

        ``action_input``/``options`` are accepted (and unused) so the patched
        method matches the ``Gr00tN1d7.get_action_with_features`` contract that
        ``get_action`` calls with these keywords; the DiT-only path needs neither.
        """
        vl_embs = backbone_features
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        engine_dtype = torch.bfloat16

        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        dt = 1.0 / action_head.num_inference_timesteps

        for t in range(action_head.num_inference_timesteps):
            t_cont = t / float(action_head.num_inference_timesteps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1).to(engine_dtype)

            # Use TRT for DiT
            vl_embs_trt = vl_embs.to(engine_dtype)
            timesteps_trt = timesteps_tensor.to(torch.int64)

            action_head.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            action_head.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs_trt.shape)
            action_head.dit_engine.set_runtime_tensor_shape("timestep", timesteps_trt.shape)

            dit_kwargs = {}
            if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
                image_mask = backbone_output.image_mask
                action_head.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
                dit_kwargs["image_mask"] = image_mask

            if (
                hasattr(backbone_output, "backbone_attention_mask")
                and backbone_output.backbone_attention_mask is not None
            ):
                bb_mask = backbone_output.backbone_attention_mask
                action_head.dit_engine.set_runtime_tensor_shape(
                    "backbone_attention_mask", bb_mask.shape
                )
                dit_kwargs["backbone_attention_mask"] = bb_mask

            model_output = action_head.dit_engine(
                sa_embs, vl_embs_trt, timesteps_trt, **dit_kwargs
            )["output"]

            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embs,
                "state_features": state_features,
            }
        )

    action_head.get_action_with_features = dit_only_get_action_with_features
    print("DiT-only TRT engine loaded and forward method patched.")


# Maps each InferenceMode to its engine-swap routine. setup_tensorrt_engines
# dispatches through this; a deployment test asserts the keys equal InferenceMode
# exactly, so a new mode can't be added without a matching setup branch.
_INFERENCE_MODE_DISPATCH = {
    InferenceMode.n17_full_pipeline: _setup_n17_full_pipeline,
    InferenceMode.vit_llm_only: _setup_vit_llm_only,
    InferenceMode.action_head: _setup_action_head,
    InferenceMode.dit_only: _setup_dit_only,
}
