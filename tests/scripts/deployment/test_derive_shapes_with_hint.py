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

"""TensorRT shape profiles derived from an ONNX graph's named dynamic dims."""

from __future__ import annotations

from pathlib import Path

import pytest


onnx = pytest.importorskip("onnx")
pytest.importorskip("tensorrt")

DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")


def _graph_with_named_dims(path):
    from onnx import TensorProto, helper

    features = helper.make_tensor_value_info(
        "backbone_features", TensorProto.FLOAT, [1, "vl_seq_len", 8]
    )
    text_index = helper.make_tensor_value_info("text_index", TensorProto.INT64, ["num_text_tokens"])
    other = helper.make_tensor_value_info("other", TensorProto.INT64, ["unnamed"])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, "vl_seq_len", 8])
    node = helper.make_node("Identity", ["backbone_features"], ["out"])
    graph = helper.make_graph([node], "g", [features, text_index, other], [out])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), path)


def test_max_seq_lens_overrides_the_derived_maximum(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    from build_tensorrt_engine import derive_shapes_with_hint

    path = str(tmp_path / "g.onnx")
    _graph_with_named_dims(path)
    mins, opts, maxs = derive_shapes_with_hint(
        path,
        opt_seq_lens={"vl_seq_len": 461, "num_text_tokens": 13},
        max_seq_lens={"num_text_tokens": 922},
    )
    assert (mins["backbone_features"], opts["backbone_features"], maxs["backbone_features"]) == (
        (1, 1, 8),
        (1, 461, 8),
        (1, 922, 8),
    )
    # The text count keeps the sequence's range instead of max(2 * 13, 13 + 64).
    assert (mins["text_index"], opts["text_index"], maxs["text_index"]) == ((1,), (13,), (922,))
    # A dim without a hint keeps the wide default range.
    assert (mins["other"], opts["other"], maxs["other"]) == ((1,), (256,), (512,))
