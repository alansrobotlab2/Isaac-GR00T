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

"""CPU checks for the opt-in INT4 weight-only quantizer (``quantize_onnx_int4.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")
BLOCK = 4


@pytest.fixture
def quantizer(monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    import quantize_onnx_int4

    return quantize_onnx_int4


def _unpack_int4(tensor) -> np.ndarray:
    raw = np.frombuffer(tensor.raw_data, dtype=np.uint8)
    nibbles = np.stack([raw & 0x0F, raw >> 4], axis=1).reshape(-1)[: int(np.prod(tensor.dims))]
    signed = nibbles.astype(np.int8)
    signed[signed > 7] -= 16
    return signed.reshape(tuple(tensor.dims))


def _model(quantizer, weight):
    """x -> MatMul(w) -> MatMul(Identity(w)): one weight shared by two MatMuls."""
    from onnx import TensorProto, helper

    k, n = weight.shape
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="/model/first/MatMul"),
        helper.make_node("Identity", ["w"], ["w_alias"], name="Identity_0"),
        helper.make_node("MatMul", ["h", "w_alias"], ["y"], name="/model/second/MatMul"),
    ]
    graph = helper.make_graph(
        nodes,
        "shared_weight",
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1, 3, k])],
        [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [1, 3, n])],
        [quantizer._bf16_tensor("w", weight)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def test_quantize_weight_is_close_and_in_range(quantizer):
    import torch

    torch.manual_seed(0)
    weight = torch.randn(16, 8) * 0.02
    quantized, scale, relative_error = quantizer.quantize_weight(weight, BLOCK)

    assert quantized.dtype == torch.int8
    assert quantized.min() >= -8 and quantized.max() <= 7
    assert scale.shape == (16 // BLOCK, 8)
    restored = (quantized.reshape(16 // BLOCK, BLOCK, 8) * scale[:, None, :]).reshape(16, 8)
    assert torch.allclose(restored, weight, atol=float(scale.max()) * 0.51 + 1e-6)
    assert 0 < relative_error < 0.2


def test_shared_weight_gets_one_int4_tensor_per_matmul(quantizer):
    import torch

    torch.manual_seed(1)
    weight = (torch.randn(8, 8) * 0.02).to(torch.bfloat16).float()
    model = _model(quantizer, weight)

    summary = quantizer.quantize_model(model, block_size=BLOCK)

    assert summary["layers"] == 1 and summary["params"] == 64
    graph = model.graph
    dequantize = [n for n in graph.node if n.op_type == "DequantizeLinear"]
    matmuls = [n for n in graph.node if n.op_type == "MatMul"]
    assert len(dequantize) == len(matmuls) == 2
    # Each MatMul reads its own DequantizeLinear, which reads its own INT4 tensor.
    assert {m.input[1] for m in matmuls} == {d.output[0] for d in dequantize}
    assert len({d.input[0] for d in dequantize}) == 2
    assert not [n for n in graph.node if n.op_type == "Identity"]
    initializers = {i.name: i for i in graph.initializer}
    assert "w" not in initializers
    assert next(o.version for o in model.opset_import if o.domain == "") >= 21
    for node in dequantize:
        attrs = {a.name: a.i for a in node.attribute}
        assert attrs == {"axis": 0, "block_size": BLOCK}

    expected, _, _ = quantizer.quantize_weight(weight, BLOCK)
    for node in dequantize:
        np.testing.assert_array_equal(_unpack_int4(initializers[node.input[0]]), expected.numpy())


def test_only_and_skip_filters(quantizer):
    import torch

    weight = torch.randn(8, 8) * 0.02
    assert quantizer.quantize_model(_model(quantizer, weight), BLOCK, only="/other/")["layers"] == 0
    assert quantizer.quantize_model(_model(quantizer, weight), BLOCK, skip="/model/")["layers"] == 0
    assert quantizer.quantize_model(_model(quantizer, weight), BLOCK, only="/model/")["layers"] == 1
