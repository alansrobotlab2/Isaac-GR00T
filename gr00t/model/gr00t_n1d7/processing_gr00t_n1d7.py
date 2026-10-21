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

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import random
import re
import threading
from typing import Any, Dict
import warnings

import albumentations as A
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import cached_file

from gr00t.configs.data.embodiment_configs import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.utils import parse_modality_configs, to_json_serializable

from .image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)


try:
    from transformers import Qwen3VLProcessor
except ImportError:
    Qwen3VLProcessor = None

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)

### Projector-index assignments, declared as ``{projector_index: {tags}}``.
#
# This grouped form is the source of truth: a tag only shares a projector with
# another tag if it is deliberately placed inside that index's set, so an
# accidental collision can't slip in unnoticed. Multiple tags share an index
# only when they describe the *same physical embodiment*. To add a brand-new
# embodiment, give it an unused index; to add a data-source/subtask variant of
# an existing one, add its tag to that group.
#
# ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX`` below is derived from this and is the
# public, tag-keyed lookup used everywhere else.
_PROJECTOR_INDEX_GROUPS: dict[int, set[str]] = {
    0: {"simpler_env_google"},
    1: {"simpler_env_widowx"},
    2: {"libero_sim"},
    # Finetune placeholder projector; sim-eval robocasa tags piggyback on
    # `new_embodiment`.
    10: {"new_embodiment", "robocasa_panda_omron", "robocasa_gr1_tabletop"},
    11: {"unitree_g1_sonic"},
    24: {"oxe_droid_relative_eef_relative_joint"},
    # Same G1 embodiment either side of the pretrain/posttrain boundary
    # (`real_g1_*` is pretrain, `unitree_g1_full_body_*` is posttrain).
    25: {
        "real_g1_relative_eef_relative_joints",
        "unitree_g1_full_body_with_waist_height_nav_cmd",
    },
    # One R1 Pro Sharpa robot, four data-source variants.
    26: {
        "real_r1_pro_sharpa_relative_eef",
        "real_r1_pro_sharpa_relative_eef_human",
        "real_r1_pro_sharpa_relative_eef_maxinsights",
        "real_r1_pro_sharpa_relative_eef_mecka",
    },
    # xdof base + subtask refinement.
    27: {
        "xdof_relative_eef_relative_joint",
        "xdof_relative_eef_relative_joint_subtask",
    },
}


def _build_tag_to_projector_index(groups: dict[int, set[str]]) -> dict[str, int]:
    """Flatten ``{index: {tags}}`` into ``{tag: index}``.

    Guards against a tag accidentally appearing in two groups, which would
    otherwise be silently resolved by insertion order into a single mapping.
    """
    mapping: dict[str, int] = {}
    for index, tags in groups.items():
        for tag in tags:
            if tag in mapping:
                raise ValueError(
                    f"Embodiment tag {tag!r} is assigned to multiple projector "
                    f"indices ({mapping[tag]} and {index}) in "
                    "_PROJECTOR_INDEX_GROUPS; each tag must map to exactly one index."
                )
            mapping[tag] = index
    return mapping


EMBODIMENT_TAG_TO_PROJECTOR_INDEX: dict[str, int] = _build_tag_to_projector_index(
    _PROJECTOR_INDEX_GROUPS
)


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> Qwen3VLProcessor:
    if Qwen3VLProcessor is None:
        raise ImportError(
            "Qwen3VLProcessor is not available. "
            "Please upgrade transformers: pip install transformers>=4.52.0"
        )
    return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


def validate_action_horizons(modality_configs, max_action_horizon: int) -> None:
    """Fail at processor construction if any configured embodiment's action horizon
    (the number of action ``delta_indices``) exceeds ``max_action_horizon``.

    ``max_action_horizon`` is set from the model's ``action_horizon``; without this
    check a horizon/model mismatch (e.g. a 50-step embodiment on a 40-step model)
    surfaces only deep in the first forward, after the model and dataset are built.
    """
    offenders: dict[str, int] = {}
    for tag, config in modality_configs.items():
        action = config.get("action") if isinstance(config, dict) else None
        delta_indices = getattr(action, "delta_indices", None)
        if delta_indices is None:
            continue
        horizon = len(delta_indices)
        if horizon > max_action_horizon:
            offenders[tag] = horizon
    if offenders:
        required = max(offenders.values())
        details = ", ".join(f"{tag}={horizon}" for tag, horizon in sorted(offenders.items()))
        raise ValueError(
            f"Embodiment action horizon exceeds max_action_horizon ({max_action_horizon}): "
            f"{details}. Increase model config action_horizon to >= {required} (or reduce the "
            "embodiment action delta_indices)."
        )


_EVAL_TRANSFORM_POOL = None
_PROMPT_CACHE_SIZE = 32


def _eval_transform_pool():
    """Shared thread pool for the per-frame eval image transforms (created on first use)."""
    global _EVAL_TRANSFORM_POOL
    if _EVAL_TRANSFORM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _EVAL_TRANSFORM_POOL = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="gr00t-image"
        )
    return _EVAL_TRANSFORM_POOL


class Gr00tN1d7DataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict = {},
    ):
        ### We need to use the same processor for padding input ids and concat
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        # Device for the VLM image processor (resize, normalize, patchify). None keeps
        # it on the CPU (training, data loaders). Inference sets it to the model's GPU:
        # the uint8 frames are copied instead of float32 patches, and the work runs on a
        # side stream that the calling thread's stream then waits for (no CPU sync).
        self.image_device: str | torch.device | None = None
        self._image_stream = None
        # Skip the VLM image processor's resize. Only for frames already at a size it
        # would keep (see Gr00tN1d7Processor.set_images_pretransformed); its resize
        # otherwise resamples every frame, even one already at the target size.
        self.skip_image_resize = False

    def set_image_device(self, device: str | torch.device | None) -> None:
        """Run the VLM image processor on ``device`` (e.g. ``"cuda"``), or the CPU if None."""
        device = torch.device(device) if device is not None else None
        self.image_device = device
        self._image_stream = (
            torch.cuda.Stream(device=device)
            if device is not None and device.type == "cuda"
            else None
        )

    def __getstate__(self):
        # CUDA streams cannot be pickled (data loader workers); workers use the CPU path.
        state = self.__dict__.copy()
        state["image_device"] = None
        state["_image_stream"] = None
        state["_prompt_cache"] = None
        return state

    def enable_prompt_cache(self, enabled: bool = True) -> None:
        """Reuse the tokenized prompt while the text and image grid stay the same.

        At inference the prompt is identical from step to step; only the pixels change.
        With the cache on, the image processor still runs every call, but the tokenizer
        (and the image-token expansion before it) only runs for a new (text, image grid)
        pair. The ids are identical to an uncached call.
        """
        self._prompt_cache = {} if enabled else None

    def _run_processor(self, text_list: list[str], image_inputs: list) -> BatchFeature:
        kwargs = {"return_tensors": "pt", "padding": True}
        if self.image_device is not None:
            kwargs["device"] = self.image_device
        if self.skip_image_resize:
            kwargs["do_resize"] = False
        cache = getattr(self, "_prompt_cache", None)
        if cache is None:
            return self.processor(text=text_list, images=image_inputs, **kwargs)
        image_kwargs = {"return_tensors": "pt"}
        if self.image_device is not None:
            image_kwargs["device"] = self.image_device
        if self.skip_image_resize:
            image_kwargs["do_resize"] = False
        image_out = self.processor.image_processor(images=image_inputs, **image_kwargs)
        grid = image_out["image_grid_thw"]
        key = (tuple(text_list), tuple(tuple(row) for row in grid.tolist()))
        text_out = cache.get(key)
        if text_out is None:
            full = self.processor(text=text_list, images=image_inputs, **kwargs)
            text_out = {k: v for k, v in full.items() if k not in image_out}
            if len(cache) >= _PROMPT_CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[key] = text_out
        out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in text_out.items()}
        out.update(image_out)
        return BatchFeature(data=out)

    def _process_vlm(self, text_list: list[str], image_inputs: list) -> BatchFeature:
        stream = self._image_stream
        if stream is None:
            return self._run_processor(text_list, image_inputs)
        consumer = torch.cuda.current_stream(stream.device)
        if any(isinstance(img, torch.Tensor) and img.is_cuda for img in image_inputs):
            # Frames already on the GPU may still be in flight on the caller's stream.
            # (CPU frames need no wait, and waiting would serialize this thread behind
            # whatever model work is queued, e.g. in a pipelined server.)
            stream.wait_stream(consumer)
        with torch.cuda.stream(stream):
            vlm_inputs = self._run_processor(text_list, image_inputs)
        if threading.current_thread() is not threading.main_thread():
            # A prefetch thread: the main thread may be queueing the previous frame on
            # the same default stream, so a GPU-side wait would stall that frame. Block
            # only this thread instead.
            stream.synchronize()
            return vlm_inputs
        # Main thread (e.g. the inference server): do not block the CPU. Work queued
        # later on this stream (the model) waits for the image kernels on the GPU, so
        # the remaining CPU preprocessing overlaps them. The allocator must not recycle
        # these tensors until that stream is done with them.
        consumer.wait_stream(stream)
        for value in vlm_inputs.values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(consumer)
        return vlm_inputs

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # Handle vlm_content specially - extract text and images
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                vlm_inputs = self._process_vlm(text_list, image_inputs)
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in (
                "pixel_values",
                "image_grid_thw",
                "attention_mask",
                "input_ids",
            ):
                raise Exception("Not implemented")
            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d7DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d7Processor(BaseProcessor):
    data_collator_class = Gr00tN1d7DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        model_type: str = "qwen",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        max_action_horizon: int = 50,
        apply_sincos_state_encoding: bool = False,
        use_albumentations: bool = False,
        extra_augmentation_config: dict | None = None,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        # State augmentation
        exclude_state: bool = False,
        state_dropout_prob: float = 0.0,
        # Normalization
        use_mean_std: bool = False,
        letter_box_transform: bool = False,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.use_mean_std = use_mean_std
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.extra_augmentation_config = extra_augmentation_config

        # State augmentation settings
        self.exclude_state = exclude_state
        self.state_dropout_prob = state_dropout_prob

        self.letter_box_transform = letter_box_transform

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        validate_action_horizons(self.modality_configs, self.max_action_horizon)

        # Save image processing settings
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # Merge any missing pre-trained embodiment tags into the custom mapping
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        # Runtime-only inference option, not saved: see set_images_pretransformed.
        self.images_pretransformed = False

        # Statistics cache (mirrors state_action_processor.statistics for serialization)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

        # Choose between torchvision and albumentations transforms
        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                    extra_augmentation_config=self.extra_augmentation_config,
                    letter_box_transform=self.letter_box_transform,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size,
                image_crop_size,
                random_rotation_angle,
                color_jitter_params,
                letter_box_transform=self.letter_box_transform,
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.train()

    @property
    def collator(self):
        return self._collator

    def set_images_pretransformed(self, enabled: bool = True) -> None:
        """Take inference camera frames as final model input: no crop, no resize.

        For a camera pipeline that already produces what the eval transform and the VLM
        image processor would: a centre crop of ``crop_fraction`` of the frame, resized to
        a short edge of ``shortest_image_edge`` with both sides multiples of 32 (patch 16,
        merge 2), e.g. 352x256 for 4:3 or 448x256 for 16:9 cameras. Frames of any other
        size raise instead of being silently fed at the wrong framing. Training ignores it.
        """
        self.images_pretransformed = enabled
        self._collator.skip_image_resize = enabled

    def _use_pretransformed_images(self, image_transform) -> bool:
        return getattr(self, "images_pretransformed", False) and (
            image_transform is self.eval_image_transform
        )

    def _pretransformed_frame(self, image) -> torch.Tensor:
        """An (H, W, 3) uint8 frame at a final model size, as a (3, H, W) uint8 tensor."""
        frame = np.asarray(image)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Pre-transformed images must be (H, W, 3) uint8; got {frame.shape} {frame.dtype}"
            )
        height, width = frame.shape[:2]
        image_processor = self.processor.image_processor
        factor = image_processor.patch_size * image_processor.merge_size
        short_edge = (
            self.shortest_image_edge
            if self.shortest_image_edge is not None
            else self.image_target_size[0]
        )
        kept = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=image_processor.size["shortest_edge"],
            max_pixels=image_processor.size["longest_edge"],
        ) == (height, width)
        if min(height, width) != short_edge or not kept:
            raise ValueError(
                f"Pre-transformed images must have a short edge of {short_edge} and both "
                f"sides a multiple of {factor} (e.g. 352x256 for 4:3, 448x256 for 16:9 "
                f"cameras); got {width}x{height}"
            )
        return torch.from_numpy(frame).permute(2, 0, 1)

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        for key in statistics:
            if key not in self.statistics or override:
                if override:
                    logger.info("Overriding statistics for embodiment %r", key)
                self.statistics[key] = deepcopy(statistics[key])
            else:
                # Surfaced as a warning (not print) because callers running with
                # override_pretraining_statistics=False on a mixture dataset will
                # otherwise silently keep the pre-existing pretraining stats and
                # discard newly-merged per-dataset stats — training proceeds with
                # the wrong mean/std and only an easy-to-miss stdout line records
                # the drop.
                logger.warning(
                    "Statistics for embodiment %r already present; new stats "
                    "DISCARDED (override=False). If the new data differs from "
                    "the existing distribution this will cause silent "
                    "normalization mismatch — pass override=True (or "
                    "override_pretraining_statistics=True at the dataset level) "
                    "to use the merged stats instead.",
                    key,
                )

        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(
                embodiment_tag
            )

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def unapply(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
        prev_action: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Undo action normalization and convert relative→absolute.

        Args:
            action: Normalized action array of shape (..., action_horizon, action_dim)
            embodiment_tag: Embodiment tag
            state: State observations with "state." prefixed keys (for relative actions)
            prev_action: Unused (kept for API compatibility)

        Returns:
            Dict mapping "action.<key>" to unnormalized (absolute) action arrays.
        """
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Strip "state." prefix for StateActionProcessor
        stripped_state = None
        if state is not None:
            stripped_state = {k.replace("state.", ""): v for k, v in state.items()}

        result = self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=stripped_state
        )
        return {f"action.{key}": value for key, value in result.items()}

    def process_observation(self, observation: dict[str, Any], embodiment_tag: EmbodimentTag):
        """Process batched observation tensors for inference.

        Args:
            observation: Dict with keys like "video.<view>", "state.<key>", "<language_key>"
                Video values expected as numpy arrays of shape (B, T, H, W, C).
            embodiment_tag: Embodiment tag identifying the robot configuration.

        Returns:
            BatchFeature with tokenized VLM inputs, state, embodiment_id, and action_mask.
        """
        modality_config = self.modality_configs[embodiment_tag.value]
        transformed_observation = {}

        # Normalize states
        state_keys = modality_config["state"].modality_keys
        state_data = {key: observation[f"state.{key}"] for key in state_keys}
        exclude_state = self.exclude_state or getattr(
            modality_config["state"], "exclude_state", False
        )
        if exclude_state:
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            norm_state_dict = self.state_action_processor.apply_state(
                state=state_data, embodiment_tag=embodiment_tag.value
            )
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
            )

        assert normalized_states.shape[1] <= self.max_state_dim, (
            f"State dimension {normalized_states.shape[1]} exceeds max_state_dim {self.max_state_dim}"
        )
        padding_shape = (
            *normalized_states.shape[:-1],
            self.max_state_dim - normalized_states.shape[-1],
        )
        normalized_states = torch.cat([normalized_states, torch.zeros(padding_shape)], dim=-1)
        transformed_observation["state"] = normalized_states

        # Process images: observation values are (B, T, H, W, C) numpy arrays
        image_keys = modality_config["video"].modality_keys
        images_dict = {view: torch.from_numpy(observation[f"video.{view}"]) for view in image_keys}
        images = torch.stack(
            [images_dict[view] for view in image_keys], dim=2
        )  # (B, T, V, H, W, C)
        assert images.ndim == 6
        B, T, V, img_H, img_W, img_C = images.shape

        if self._use_pretransformed_images(self.eval_image_transform):
            images_flat = images.reshape(B * T * V, img_H, img_W, img_C).numpy()
            transformed_images = (
                torch.stack([self._pretransformed_frame(img) for img in images_flat])
                .reshape(B, T * V, img_C, img_H, img_W)
                .numpy()
            )
        elif self.use_albumentations:
            images_flat = images.reshape(B * T * V, img_H, img_W, img_C)
            pil_images = [Image.fromarray(img.numpy()) for img in images_flat]
            transformed_pil, _ = apply_with_replay(self.eval_image_transform, pil_images)
            transformed_stacked = torch.stack(transformed_pil)  # (B*T*V, C, H_new, W_new)
            _, img_C_new, img_H_new, img_W_new = transformed_stacked.shape
            transformed_images = transformed_stacked.reshape(
                B, T * V, img_C_new, img_H_new, img_W_new
            ).numpy()
        else:
            # Rearrange (B, T, V, H, W, C) → (B, T*V, C, H, W) for torchvision
            images_perm = images.permute(0, 1, 2, 5, 3, 4).reshape(B, T * V, img_C, img_H, img_W)
            transformed_images = self.eval_image_transform(images_perm).numpy()

        language_key = modality_config["language"].modality_keys[0]
        language = [
            re.sub(r"[^\w\s]", "", lang.lower()) if self.formalize_language else lang
            for lang in observation[language_key]
        ]

        texts, all_images = [], []
        for i in range(B):
            vlm_inputs = self._apply_vlm_processing(transformed_images[i], language[i])
            vc = vlm_inputs["vlm_content"]
            texts.append(vc["text"])
            all_images.extend(vc["images"])
        tokenized = self.processor(text=texts, images=all_images, return_tensors="pt", padding=True)
        for k, v in tokenized.items():
            transformed_observation[k] = v

        embodiment_id = (
            torch.ones(B, dtype=torch.int32) * self.embodiment_id_mapping[embodiment_tag.value]
        )
        transformed_observation["embodiment_id"] = embodiment_id

        # Action mask: shape (B, max_action_horizon), 1 in the valid horizon window
        action_config = modality_config["action"]
        action_horizon = len(action_config.delta_indices)
        assert action_horizon <= self.max_action_horizon, (
            f"Action horizon {action_horizon} (from delta_indices) exceeds"
            f" max_action_horizon {self.max_action_horizon}. Increase model config"
            f" action_horizon to >= {action_horizon}."
        )
        action_mask = torch.zeros((B, self.max_action_horizon), dtype=torch.float32)
        if action_horizon > 0:
            action_mask[:, :action_horizon] = 1.0
        transformed_observation["action_mask"] = action_mask

        return BatchFeature(transformed_observation)

    def _chat_template_text(self, language: str, num_images: int, conversation) -> str:
        cache = self.__dict__.setdefault("_chat_template_cache", {})
        key = (language, num_images)
        text = cache.get(key)
        if text is None:
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            if len(cache) >= 256:
                cache.pop(next(iter(cache)))
            cache[key] = text
        return text

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        frames = [torch.as_tensor(v) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in frames],
                    {"type": "text", "text": language},
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it. The text
        # depends only on the instruction and the number of images, so it is cached.
        text = self._chat_template_text(language, len(frames), conversation)

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": frames,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        # Use StateActionProcessor to handle relative conversion and normalization
        norm_state_dict, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            # Concatenate actions
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys],
                dim=-1,
            )  # (t, d)
            action_dim = normalized_actions.shape[1]
            # Pad action to max_action_dim
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            # Pad action to max_action_horizon
            action_horizon = normalized_actions.shape[0]
            assert action_horizon <= self.max_action_horizon, (
                f"Action sequence length {action_horizon} exceeds max_action_horizon"
                f" {self.max_action_horizon}. Increase model config action_horizon to"
                f" >= {action_horizon}."
            )
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            # Create action mask
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        # Concatenate states with optional dropout/noise augmentation
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        exclude_state = self.exclude_state or getattr(
            self.modality_configs[embodiment_tag.value]["state"], "exclude_state", False
        )
        if exclude_state or (
            self.state_dropout_prob > 0
            and random.random() < self.state_dropout_prob
            and self.training
        ):
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
            )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0],
                    self.max_state_dim - normalized_states.shape[1],
                ),
            ],
            dim=-1,
        )

        # Crop and resize images.
        if self.training:
            image_transform = self.train_image_transform
        else:
            image_transform = self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            masks=content.masks,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        # Add VLM inputs
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform: transforms.Compose | A.Compose,
        language: str,
    ):
        temporal_stacked_images = {}

        if self._use_pretransformed_images(image_transform):
            # The camera pipeline already cropped and resized (set_images_pretransformed).
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [self._pretransformed_frame(img) for img in images[view]]
                )
        elif (
            self.use_albumentations
            and masks is None
            and image_transform is self.eval_image_transform
            and not hasattr(image_transform, "replay")
        ):
            # Inference: the eval transform is deterministic, so every frame of every
            # camera is independent. OpenCV releases the GIL, so a thread pool runs them
            # in parallel (identical output; on a Jetson AGX Orin 4 frames take 2.5 ms
            # instead of 5.7 ms, 12 frames 4.8 ms instead of 17 ms).
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
            jobs = [(view, img) for view in image_keys for img in images[view]]
            results = list(
                _eval_transform_pool().map(
                    lambda job: apply_with_replay(image_transform, [job[1]])[0][0], jobs
                )
            )
            start = 0
            for view in image_keys:
                count = len(images[view])
                temporal_stacked_images[view] = torch.stack(results[start : start + count])
                start += count
        elif self.use_albumentations:
            # Use albumentations transforms
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                if masks is not None:
                    assert view in masks, f"{view} not in masks"
                view_masks = masks.get(view) if masks else None
                view_images = images[view]

                # Apply transforms with replay for consistency
                transformed_images, replay = apply_with_replay(
                    image_transform, view_images, view_masks, replay
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)
        else:
            if masks is not None:
                raise ValueError(
                    "Mask transforms require albumentations. Set use_albumentations_transforms=True."
                )
            # Use torchvision transforms
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = torch.stack(
            [temporal_stacked_images[view] for view in image_keys], dim=1
        ).flatten(0, 1)  # (T*V, C, H, W)

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = save_directory / "processor_config.json"
        statistics_file = save_directory / "statistics.json"
        embodiment_id_file = save_directory / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                # Image processing settings
                "image_crop_size": self.image_crop_size,
                "image_target_size": self.image_target_size,
                "use_albumentations": self.use_albumentations,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                "shortest_image_edge": self.shortest_image_edge,
                "crop_fraction": self.crop_fraction,
                "letter_box_transform": self.letter_box_transform,
                # VLM settings
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                # State action dimensions
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                # StateActionProcessor settings
                "use_percentiles": self.use_percentiles,
                "use_mean_std": self.use_mean_std,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
                # State augmentation
                "exclude_state": self.exclude_state,
                "state_dropout_prob": self.state_dropout_prob,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        # Save statistics
        with open(statistics_file, "w") as f:
            json.dump(
                to_json_serializable(self.state_action_processor.statistics),
                f,
                indent=2,
            )
        # Save embodiment id mapping
        with open(embodiment_id_file, "w") as f:
            json.dump(self.embodiment_id_mapping, f, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        hub_keys = (
            "_commit_hash",
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        )
        hub_kwargs = {key: kwargs.pop(key) for key in hub_keys if key in kwargs}
        use_auth_token = kwargs.pop("use_auth_token", None)
        if "token" not in hub_kwargs and use_auth_token is not None:
            hub_kwargs["token"] = use_auth_token
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(
                cached_file(pretrained_model_name_or_path, "processor_config.json", **hub_kwargs)
            )
            statistics_file = Path(
                cached_file(pretrained_model_name_or_path, "statistics.json", **hub_kwargs)
            )
            embodiment_id_file = Path(
                cached_file(pretrained_model_name_or_path, "embodiment_id.json", **hub_kwargs)
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file, "r") as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

        # Backfill fields that older checkpoints may not have serialized.
        # Without these, __init__ defaults silently apply — correct today but
        # fragile if defaults ever change.
        processor_kwargs.setdefault("model_name", "nvidia/Cosmos-Reason2-2B")
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs while keeping pretrained embodiment configs
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
                "exclude_state",
                "state_dropout_prob",
                "use_mean_std",
                "model_name",
                "model_type",
                "max_action_horizon",
                "max_state_dim",
                "max_action_dim",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


AutoProcessor.register("Gr00tN1d7", Gr00tN1d7Processor)
