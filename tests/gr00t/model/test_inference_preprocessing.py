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

"""Inference-time preprocessing shortcuts must not change the model inputs.

- the eval image transforms run per frame in a thread pool,
- the collator caches the tokenized prompt,
- the collator can run the VLM image processor on the GPU,
- frames already at the final model size can skip the crop and resizes.
"""

import copy
import threading
from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.image_augmentations import build_image_transformations_albumentations
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
import numpy as np
import pytest
import torch


INSTRUCTION = "pick up the black bowl and place it on the plate"


def _frames(num_views=3, num_frames=2, h=180, w=320, seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"view_{v}": [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(num_frames)]
        for v in range(num_views)
    }


def _stub_processor():
    """A processor with only what ``_get_vlm_inputs`` needs (no tokenizer download)."""
    processor = object.__new__(Gr00tN1d7Processor)
    processor.use_albumentations = True
    processor.train_image_transform, processor.eval_image_transform = (
        build_image_transformations_albumentations(
            image_target_size=[256, 256],
            image_crop_size=[230, 230],
            random_rotation_angle=None,
            color_jitter_params=None,
            shortest_image_edge=256,
            crop_fraction=0.95,
        )
    )
    processor._apply_vlm_processing = lambda images, language: {"images": images}
    processor.shortest_image_edge = 256
    processor.image_target_size = [256, 256]
    # The Qwen3-VL image processor settings _pretransformed_frame reads.
    processor.processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
    )
    return processor


def test_threaded_eval_transforms_match_sequential():
    processor = _stub_processor()
    images = _frames()
    keys = sorted(images)
    threaded = processor._get_vlm_inputs(
        keys, images, None, processor.eval_image_transform, INSTRUCTION
    )["images"]
    # A copy of the transform is not "the" eval transform, so it takes the sequential path.
    sequential = processor._get_vlm_inputs(
        keys, images, None, copy.copy(processor.eval_image_transform), INSTRUCTION
    )["images"]
    assert threaded.dtype == torch.uint8
    assert threaded.shape[0] == len(keys) * 2
    assert torch.equal(threaded, sequential)


def test_pretransformed_images_skip_the_eval_transform():
    processor = _stub_processor()
    processor.images_pretransformed = True
    images = _frames(h=256, w=352)
    keys = sorted(images)
    out = processor._get_vlm_inputs(
        keys, images, None, processor.eval_image_transform, INSTRUCTION
    )["images"]
    # (T*V, C, H, W): time-major, views in key order.
    expected = torch.stack(
        [torch.from_numpy(images[k][t]).permute(2, 0, 1) for t in range(2) for k in keys]
    )
    assert out.dtype == torch.uint8
    assert torch.equal(out, expected)
    # 16:9 at the model size is accepted too.
    processor._get_vlm_inputs(
        keys, _frames(h=256, w=448), None, processor.eval_image_transform, INSTRUCTION
    )


@pytest.mark.parametrize(
    "h, w",
    [
        (240, 320),  # raw camera size: still needs the crop and resize
        (256, 340),  # eval-transform output before the multiple-of-32 snap
        (288, 384),  # multiples of 32, but the short edge is not 256
    ],
)
def test_pretransformed_images_reject_other_sizes(h, w):
    processor = _stub_processor()
    processor.images_pretransformed = True
    images = _frames(h=h, w=w)
    with pytest.raises(ValueError, match="short edge of 256"):
        processor._get_vlm_inputs(
            sorted(images), images, None, processor.eval_image_transform, INSTRUCTION
        )


def test_pretransformed_images_do_not_affect_training():
    processor = _stub_processor()
    processor.images_pretransformed = True
    images = _frames(h=240, w=320)  # not a model size: only the eval path would reject it
    out = processor._get_vlm_inputs(
        sorted(images), images, None, processor.train_image_transform, INSTRUCTION
    )["images"]
    assert out.shape[-2:] == (256, 340)


# ---------------------------------------------------------------------------
# Collator: prompt cache and GPU image processor (needs the real Qwen3-VL processor)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def collator(load_hf_model_weights):
    try:
        with load_hf_model_weights():
            return Gr00tN1d7DataCollator(
                "nvidia/Cosmos-Reason2-2B", "qwen", {"trust_remote_code": True}
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Cosmos-Reason2-2B processor unavailable (gated/offline?): {exc}")


def _vlm_features(collator, seed=0, language=INSTRUCTION, num_images=4, h=256, w=448):
    rng = np.random.default_rng(seed)
    frames = [
        torch.from_numpy(rng.integers(0, 256, (3, h, w), dtype=np.uint8)) for _ in range(num_images)
    ]
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": f} for f in frames],
                {"type": "text", "text": language},
            ],
        }
    ]
    text = collator.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    return [
        {"vlm_content": {"text": text, "images": frames}, "state": np.zeros((1, 4), np.float32)}
    ]


def _inputs(collator, features):
    return collator(features)["inputs"]


def _assert_same(a, b, pixel_atol=0.0):
    assert sorted(a) == sorted(b)
    for key in a:
        if key == "pixel_values":
            torch.testing.assert_close(
                a[key].float().cpu(), b[key].float().cpu(), atol=pixel_atol, rtol=0
            )
        else:
            assert torch.equal(a[key].cpu(), b[key].cpu()), key


def test_prompt_cache_is_exact(collator):
    collator.set_image_device(None)
    collator.enable_prompt_cache(False)
    reference = [_inputs(collator, _vlm_features(collator, seed)) for seed in range(2)]
    other_prompt = _inputs(collator, _vlm_features(collator, 0, language="open the drawer"))
    fewer_images = _inputs(collator, _vlm_features(collator, 0, num_images=2))

    collator.enable_prompt_cache(True)
    try:
        for seed in range(2):  # first call fills the cache, second one hits it
            _assert_same(_inputs(collator, _vlm_features(collator, seed)), reference[seed])
        assert len(collator._prompt_cache) == 1
        # A different prompt or image layout is a different entry, never a stale hit.
        _assert_same(
            _inputs(collator, _vlm_features(collator, 0, language="open the drawer")), other_prompt
        )
        _assert_same(_inputs(collator, _vlm_features(collator, 0, num_images=2)), fewer_images)
        assert len(collator._prompt_cache) == 3
        # Cached ids are copies: editing a result must not corrupt the next one.
        out = _inputs(collator, _vlm_features(collator, 0))
        out["input_ids"].zero_()
        _assert_same(_inputs(collator, _vlm_features(collator, 0)), reference[0])
    finally:
        collator.enable_prompt_cache(False)


@pytest.mark.gpu
def test_gpu_image_processor_matches_cpu(collator):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    collator.set_image_device(None)
    reference = _inputs(collator, _vlm_features(collator))
    collator.set_image_device("cuda")
    try:
        # Main thread: the result is ready for work queued on the current stream.
        out = _inputs(collator, _vlm_features(collator))
        assert out["pixel_values"].is_cuda
        _assert_same(out, reference, pixel_atol=0.011)  # within one 8-bit step

        # Background thread (prefetch): the result is complete when the call returns.
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(_inputs(collator, _vlm_features(collator)))
        )
        worker.start()
        worker.join()
        _assert_same(result, reference, pixel_atol=0.011)
    finally:
        collator.set_image_device(None)


@pytest.mark.parametrize("h, w", [(256, 352), (256, 448)])
def test_skip_image_resize_matches_resize_at_model_size(collator, h, w):
    """At a size the image processor keeps, its resize is a no-op worth skipping."""
    collator.set_image_device(None)
    reference = _inputs(collator, _vlm_features(collator, h=h, w=w))
    collator.skip_image_resize = True
    try:
        out = _inputs(collator, _vlm_features(collator, h=h, w=w))
    finally:
        collator.skip_image_resize = False
    assert out["image_grid_thw"][0].tolist() == [1, h // 16, w // 16]
    _assert_same(out, reference, pixel_atol=0.011)  # within one 8-bit step
