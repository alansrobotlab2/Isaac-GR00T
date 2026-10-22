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

"""CPU checks for the fused action-head engine: K/V sharing math and engine discovery."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")


def test_shared_kv_processor_matches_default_attention(monkeypatch):
    """Reusing K/V across denoising steps must not change the cross-attention output."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.attention import Attention

    monkeypatch.syspath_prepend(DEPLOY_DIR)
    from export_onnx_n1d7 import _SharedKVCrossAttnProcessor, _supports_shared_kv

    torch.manual_seed(0)
    attn = Attention(query_dim=48, heads=4, dim_head=12, cross_attention_dim=32, bias=True).eval()
    assert _supports_shared_kv(attn)
    self_attn = Attention(query_dim=48, heads=4, dim_head=12).eval()
    assert not _supports_shared_kv(self_attn)

    conditioning = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[:, 5:] = False
    queries = [torch.randn(2, 7, 48) for _ in range(3)]

    with torch.no_grad():
        expected = [
            attn(q, encoder_hidden_states=conditioning, attention_mask=mask) for q in queries
        ]
        shared = _SharedKVCrossAttnProcessor()
        attn.set_processor(shared)
        actual = [attn(q, encoder_hidden_states=conditioning, attention_mask=mask) for q in queries]

    assert shared.cache is not None
    for want, got in zip(expected, actual):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.fixture
def trt_model_forward(monkeypatch):
    """Import ``trt_model_forward`` without TensorRT (same approach as the ViT-path tests)."""
    if "tensorrt" not in sys.modules:
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            trt_torch_stub = types.ModuleType("trt_torch")
            trt_torch_stub.Engine = object
            monkeypatch.setitem(sys.modules, "trt_torch", trt_torch_stub)
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    monkeypatch.delitem(sys.modules, "trt_model_forward", raising=False)
    import trt_model_forward as module

    return module


def test_find_fused_engines_maps_step_count_to_path(tmp_path, trt_model_forward):
    for name in (
        "action_head_fused_2step.engine",
        "action_head_fused_4step.engine",
        "action_head_fused_4step.onnx",  # not an engine
        "action_head_fused_xstep.engine",  # not a step count
        "dit_bf16.engine",
    ):
        (tmp_path / name).write_bytes(b"x")

    found = trt_model_forward._find_fused_engines(str(tmp_path))

    assert found == {
        2: str(tmp_path / "action_head_fused_2step.engine"),
        4: str(tmp_path / "action_head_fused_4step.engine"),
    }


class _FakeEngine:
    """Records what the fused forward binds; declares the given input names."""

    def __init__(self, input_names):
        self.in_meta = [[name, (-1,), None] for name in input_names]
        self.shapes = {}
        self.calls = []

    def set_runtime_tensor_shape(self, name, shape):
        self.shapes[name] = tuple(shape)

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return {"actions": inputs["init_actions"].float()}


def _fused_inputs(torch):
    from transformers.feature_extraction_utils import BatchFeature

    image_mask = torch.zeros(1, 9, dtype=torch.bool)
    image_mask[:, 1:6] = True
    attended = torch.ones(1, 9, dtype=torch.bool)
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.zeros(1, 9, 8, dtype=torch.bfloat16),
            "image_mask": image_mask,
            "backbone_attention_mask": attended,
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.zeros(1, 1, 4, dtype=torch.bfloat16),
            "embodiment_id": torch.tensor([3]),
        }
    )
    return backbone_output, action_input


def test_fused_forward_feeds_token_indices_to_current_engines(trt_model_forward):
    """A graph with index inputs gets the two token subsets (computed when the backbone
    output lacks them); a mask graph gets the masks; the embodiment is checked once."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    backbone_output, action_input = _fused_inputs(torch)
    head = types.SimpleNamespace(
        init_actions=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        _trt_fused_embodiment_id=3,
        _trt_fused_embodiment_ok=False,
    )

    engine = _FakeEngine(
        ["backbone_features", "state", "init_actions", "image_index", "text_index"]
    )
    trt_model_forward._fused_action_head_forward(head, engine, backbone_output, action_input)
    bound = engine.calls[0]
    assert bound["image_index"].tolist() == [1, 2, 3, 4, 5]
    assert bound["text_index"].tolist() == [0, 6, 7, 8]
    assert engine.shapes["image_index"] == (5,) and engine.shapes["text_index"] == (4,)
    assert "image_mask" not in bound and head._trt_fused_embodiment_ok

    # Indices supplied by the backbone forward (cached per layout) are used as given.
    backbone_output["image_index"] = torch.tensor([2, 3])
    backbone_output["text_index"] = torch.tensor([0, 1])
    trt_model_forward._fused_action_head_forward(head, engine, backbone_output, action_input)
    assert engine.calls[1]["image_index"].tolist() == [2, 3]

    # Older graphs: boolean masks over the whole sequence.
    old = _FakeEngine(
        [
            "backbone_features",
            "state",
            "embodiment_id",
            "init_actions",
            "image_mask",
            "backbone_attention_mask",
        ]
    )
    trt_model_forward._fused_action_head_forward(head, old, backbone_output, action_input)
    assert "image_mask" in old.calls[0] and "image_index" not in old.calls[0]
    assert old.shapes["image_mask"] == (1, 9)


def test_fused_forward_rejects_other_embodiment(trt_model_forward):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    backbone_output, action_input = _fused_inputs(torch)
    head = types.SimpleNamespace(
        init_actions=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        _trt_fused_embodiment_id=7,
        _trt_fused_embodiment_ok=False,
    )
    engine = _FakeEngine(
        ["backbone_features", "state", "init_actions", "image_index", "text_index"]
    )
    with pytest.raises(ValueError, match="embodiment id 7"):
        trt_model_forward._fused_action_head_forward(head, engine, backbone_output, action_input)


def test_fused_onnx_and_engine_names_agree(monkeypatch, trt_model_forward):
    """The exporter's ONNX name, the builder's pattern and the runtime's pattern must line up."""
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    pytest.importorskip("tensorrt")
    from build_tensorrt_engine import FUSED_ACTION_HEAD_ONNX_RE
    from export_onnx_n1d7 import FUSED_ACTION_HEAD_ONNX

    onnx_name = FUSED_ACTION_HEAD_ONNX.format(steps=2)
    match = FUSED_ACTION_HEAD_ONNX_RE.fullmatch(onnx_name)
    assert match and match.group(1) == "2"
    engine_name = onnx_name.replace(".onnx", ".engine")
    assert trt_model_forward._FUSED_ENGINE_RE.fullmatch(engine_name).group(1) == "2"
