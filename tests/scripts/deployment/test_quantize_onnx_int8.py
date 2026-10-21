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

"""Checks for the INT8 post-training quantizer (``quantize_onnx_int8.py``)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest


DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")
DIM, HIDDEN, TOKENS = 16, 24, 5


@pytest.fixture
def quantizer(monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    import quantize_onnx_int8

    return quantize_onnx_int8


def _model(quantizer, dtype_name="FLOAT"):
    """x -> Gemm(transB) -> Relu -> MatMul(w2) -> MatMul(Identity(w2)), plus a residual Add.

    Covers both weight layouts, a weight shared through an Identity alias, and an
    activation (``x``) read by a linear layer and by a non-linear consumer.
    """
    from onnx import TensorProto, helper
    import torch

    torch.manual_seed(0)
    dtype = getattr(TensorProto, dtype_name)
    w1 = torch.randn(HIDDEN, DIM) * 0.2  # Gemm transB=1: [out, in]
    b1 = torch.randn(HIDDEN) * 0.1
    w2 = torch.randn(HIDDEN, HIDDEN) * 0.2  # MatMul: [in, out]
    w3 = torch.randn(HIDDEN, DIM) * 0.2
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h1"], name="/blocks.0/fc1/Gemm", transB=1),
        helper.make_node("Relu", ["h1"], ["r1"], name="relu"),
        helper.make_node("MatMul", ["r1", "w2"], ["h2"], name="/blocks.0/fc2/MatMul"),
        helper.make_node("Identity", ["w2"], ["w2_alias"], name="Identity_0"),
        helper.make_node("MatMul", ["h2", "w2_alias"], ["h3"], name="/blocks.1/fc2/MatMul"),
        helper.make_node("MatMul", ["h3", "w3"], ["h4"], name="/blocks.1/out/MatMul"),
        helper.make_node("ReduceSum", ["x"], ["xs"], name="sum", keepdims=1),
        helper.make_node("Add", ["h4", "xs"], ["y"], name="residual"),
    ]
    inits = [
        quantizer._float_tensor("w1", w1, dtype),
        quantizer._float_tensor("b1", b1, dtype),
        quantizer._float_tensor("w2", w2, dtype),
        quantizer._float_tensor("w3", w3, dtype),
    ]
    graph = helper.make_graph(
        nodes,
        "int8_test",
        [helper.make_tensor_value_info("x", dtype, [TOKENS, DIM])],
        [helper.make_tensor_value_info("y", dtype, [TOKENS, DIM])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def _run(model, x):
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(model).run(None, {"x": x})[0]


@pytest.mark.parametrize("channel_axis", [0, 1])
def test_quantize_weight_per_output_channel(quantizer, channel_axis):
    import torch

    torch.manual_seed(0)
    weight = torch.randn(12, 10)
    weight[3, 4] = 40.0  # one outlier must only coarsen its own channel
    quantized, scale, relative_error = quantizer.quantize_weight(weight, channel_axis)

    assert quantized.dtype == torch.int8
    assert int(quantized.abs().max()) <= 127
    assert scale.shape == (weight.shape[channel_axis],)
    restored = quantized.float() * scale.unsqueeze(1 - channel_axis)
    step = scale.unsqueeze(1 - channel_axis)
    assert torch.all((restored - weight).abs() <= step * 0.51 + 1e-6)
    assert 0 < relative_error < 0.02


def test_mse_optimal_clip(quantizer):
    import torch

    torch.manual_seed(0)
    # Uniform data: nothing to gain from clipping.
    uniform = torch.rand(100_000)
    hist = torch.histc(uniform, 4096, 0, float(uniform.max()))
    assert quantizer._mse_optimal_clip(hist, float(uniform.max())) > 0.95 * float(uniform.max())
    # Gaussian bulk with one huge outlier: clipping it beats a coarse step for everything.
    data = torch.cat([torch.randn(1_000_000).abs(), torch.full((1,), 200.0)])
    hist = torch.histc(data, 4096, 0, float(data.max()))
    assert quantizer._mse_optimal_clip(hist, float(data.max())) < 50.0


def test_find_linear_layers_filters(quantizer):
    model = _model(quantizer)
    names = [layer.node.name for layer in quantizer.find_linear_layers(model)]
    assert names == [
        "/blocks.0/fc1/Gemm",
        "/blocks.0/fc2/MatMul",
        "/blocks.1/fc2/MatMul",
        "/blocks.1/out/MatMul",
    ]
    axes = [layer.channel_axis for layer in quantizer.find_linear_layers(model)]
    assert axes == [0, 1, 1, 1]
    only = quantizer.find_linear_layers(model, only=r"blocks\.1")
    assert [layer.node.name for layer in only] == ["/blocks.1/fc2/MatMul", "/blocks.1/out/MatMul"]
    skip = quantizer.find_linear_layers(model, skip=r"fc2")
    assert [layer.node.name for layer in skip] == ["/blocks.0/fc1/Gemm", "/blocks.1/out/MatMul"]


def _clips(model, x):
    """Exact per-tensor ranges of every linear input, from a float reference run."""
    from onnx import helper

    probe = copy.deepcopy(model)
    taps = ["x", "r1", "h2", "h3"]
    for name in taps[1:]:
        probe.graph.output.append(helper.make_tensor_value_info(name, 0, None))
    from onnx.reference import ReferenceEvaluator

    values = ReferenceEvaluator(probe).run(None, {"x": x})
    found = dict(zip(["y", *taps[1:]], values))
    found["x"] = x
    return {name: float(np.abs(found[name]).max()) for name in taps}, found


@pytest.mark.parametrize("smooth", [False, True])
def test_quantize_model_rewrites_graph_and_keeps_outputs(quantizer, smooth):
    import torch

    model = _model(quantizer)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((TOKENS, DIM)).astype(np.float32)
    x[:, 3] *= 20.0  # an outlier channel, the case smoothing is for
    clips, values = _clips(model, x)
    divisors = None
    if smooth:
        divisors = {
            name: torch.from_numpy(np.abs(values[name]).max(axis=0)).float().clamp_min(1e-3) ** 0.5
            for name in clips
        }
        clips = {name: clips[name] / float(divisors[name].min()) for name in clips}

    want = _run(model, x)
    quantized = copy.deepcopy(model)
    layers = quantizer.find_linear_layers(quantized)
    summary = quantizer.quantize_model(quantized, layers, clips, divisors)

    ops = [node.op_type for node in quantized.graph.node]
    assert summary["layers"] == 4
    assert summary["activations"] == 4  # x, r1, h2, h3
    assert ops.count("QuantizeLinear") == 4
    assert ops.count("DequantizeLinear") == 4 + 4  # 4 activations + 4 weight uses
    assert ops.count("Mul") == (4 if smooth else 0)
    assert "Identity" not in ops  # the alias lost its consumer
    # The residual branch still reads the unquantized input.
    residual = next(node for node in quantized.graph.node if node.name == "sum")
    assert residual.input[0] == "x"
    assert {opset.version for opset in quantized.opset_import if opset.domain == ""} == {21}

    got = _run(quantized, x)
    relative = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert relative < 0.05, relative


@pytest.mark.gpu
def test_calibrate_with_tensorrt(quantizer, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    import onnx

    model = _model(quantizer, "BFLOAT16")
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)
    torch.manual_seed(0)
    # Samples of two different shapes exercise the dynamic profile.
    samples = [torch.randn(TOKENS, DIM) for _ in range(3)] + [torch.randn(TOKENS, DIM) * 2]
    taps = sorted({layer.node.input[0] for layer in quantizer.find_linear_layers(model)})
    clips, divisors = quantizer.calibrate(path, taps, {"x": samples})
    assert set(clips) == set(taps)
    assert divisors == {}
    max_x = max(float(s.abs().max()) for s in samples)
    assert 0 < clips["x"] <= max_x * 1.001

    out = tmp_path / "model_int8.onnx"
    out.symlink_to(path)  # a stale link must be replaced, not written through
    summary = quantizer.quantize_file(path, str(out), {"x": samples}, smooth_alpha=0.5)
    assert summary["layers"] == 4
    assert (tmp_path / "model_int8.onnx.data").exists()
    assert not out.is_symlink()
    assert onnx.load(path, load_external_data=False).graph.node[0].op_type == "Gemm"


# ---------------------------------------------------------------------------
# Pipeline helpers (int8_calibration.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    import int8_calibration

    return int8_calibration


class _Cfg:
    def __init__(self, delta_indices):
        self.delta_indices = delta_indices


def test_calibration_steps_cover_dataset_within_valid_windows(calibration):
    dataset = [list(range(50)), list(range(40)), list(range(100))]
    configs = {"video": _Cfg([-15, 0]), "state": _Cfg([0]), "action": _Cfg(list(range(16)))}
    steps = calibration._calibration_steps(dataset, configs, num_frames=12)
    assert len(steps) == 12
    assert {traj for traj, _ in steps} == {0, 1, 2}
    for traj, step in steps:
        assert 15 <= step <= len(dataset[traj]) - 16  # history and action window fit
    assert steps == sorted(steps)
    # Asking for more frames than exist returns every valid frame once.
    every = calibration._calibration_steps(dataset, configs, num_frames=10_000)
    assert len(every) == len(set(every)) == (50 - 30) + (40 - 30) + (100 - 30)


def test_mirror_links_everything_but_quantized_graphs(calibration, tmp_path):
    src, dst = tmp_path / "onnx", tmp_path / "onnx_int8"
    src.mkdir()
    for name in ["vit_bf16.onnx", "llm_bf16.onnx", "llm_bf16.onnx.data", "export_metadata.json"]:
        (src / name).write_text(name)
    (src / "dit_bf16.onnx").write_text("dit")
    (src / "dit_bf16.onnx.data").write_text("dit weights")
    calibration._mirror(str(src), str(dst), skip={"vit_bf16.onnx", "dit_bf16.onnx"})
    assert sorted(p.name for p in dst.iterdir()) == [
        "export_metadata.json",
        "llm_bf16.onnx",
        "llm_bf16.onnx.data",
    ]
    assert all((dst / p).is_symlink() for p in ["llm_bf16.onnx", "export_metadata.json"])
    # Re-running replaces the links instead of failing.
    calibration._mirror(str(src), str(dst), skip={"vit_bf16.onnx", "dit_bf16.onnx"})
    # A later run that also quantizes the LLM must drop its old link, or writing the
    # quantized graph would go through it into the export directory.
    calibration._mirror(str(src), str(dst), skip={"vit_bf16.onnx", "llm_bf16.onnx"})
    assert not (dst / "llm_bf16.onnx").exists()
    assert not (dst / "llm_bf16.onnx.data").exists()
    assert (src / "llm_bf16.onnx").read_text() == "llm_bf16.onnx"
