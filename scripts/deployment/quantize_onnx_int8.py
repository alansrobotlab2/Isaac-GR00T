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

"""Post-training INT8 quantization of an exported ONNX graph's linear layers for TensorRT.

Every ``Gemm`` / ``MatMul`` with a constant weight is rewritten as::

    activation -> QuantizeLinear -> DequantizeLinear --\\
                                                        Gemm / MatMul -> (unchanged, BF16)
    INT8 weight (per output channel) -> DequantizeLinear --/

TensorRT fuses that into an integer GEMM. Both halves are needed: with only the weights
quantized TensorRT folds them back to BF16 and nothing changes. On a Jetson AGX Orin the
integer kernel is about 2x faster than the BF16 one at every size that matters here
(1,792 ViT tokens, 461 LLM tokens, 41 DiT tokens), because it does half the tensor-core
work and reads half the weight bytes.

The activation scales are calibrated with TensorRT itself: the unmodified graph is built
once with every linear-layer input marked as an extra output, the calibration samples are
run through it, and each scale is the clipping range that minimizes the squared
quantization error of that tensor. No PyTorch model and no onnxruntime are needed, only
the graph and a file of representative inputs.

Calibration samples are a ``torch.save``-d dict ``{input name: tensor [num_samples, ...]}``
holding real inputs of the graph; ``build_trt_pipeline.py --int8 ...`` captures them from
the dataset and calls this module. It changes the numerics, so check the accuracy the
pipeline's verify step reports.
"""

from dataclasses import dataclass
import logging
import os
import re

import numpy as np
import onnx
from onnx import TensorProto, helper
import torch
import tyro


logger = logging.getLogger(__name__)

# Q/DQ on BF16 tensors needs opset 21.
_MIN_OPSET = 21
_HIST_BINS = 4096
_ONNX_FLOAT_TYPES = {TensorProto.BFLOAT16: torch.bfloat16, TensorProto.FLOAT: torch.float32}


# ---------------------------------------------------------------------------
# ONNX tensor helpers
# ---------------------------------------------------------------------------


def _read_float_tensor(tensor: onnx.TensorProto) -> torch.Tensor:
    dims = tuple(tensor.dims)
    if tensor.data_type == TensorProto.BFLOAT16:
        raw = np.frombuffer(tensor.raw_data, dtype=np.int16).copy()
        return torch.from_numpy(raw).view(torch.bfloat16).reshape(dims).float()
    return torch.from_numpy(onnx.numpy_helper.to_array(tensor).copy()).reshape(dims).float()


def _float_tensor(name: str, values: torch.Tensor, data_type: int) -> onnx.TensorProto:
    values = values.detach().cpu()
    if data_type == TensorProto.BFLOAT16:
        raw = values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    else:
        raw = values.to(torch.float32).contiguous().numpy().tobytes()
    return helper.make_tensor(name, data_type, list(values.shape), raw, raw=True)


def _int8_tensor(name: str, values: torch.Tensor) -> onnx.TensorProto:
    raw = values.detach().cpu().to(torch.int8).contiguous().numpy().tobytes()
    return helper.make_tensor(name, TensorProto.INT8, list(values.shape), raw, raw=True)


# ---------------------------------------------------------------------------
# Finding the layers
# ---------------------------------------------------------------------------


@dataclass
class _Linear:
    node: onnx.NodeProto
    weight_name: str  # initializer holding the weight (after resolving Identity aliases)
    channel_axis: int  # axis of the weight that indexes output channels


def find_linear_layers(
    model: onnx.ModelProto, only: str | None = None, skip: str | None = None
) -> list[_Linear]:
    """Return the Gemm / MatMul nodes of ``model`` that have a constant 2-D float weight.

    ``only`` / ``skip`` are regular expressions matched against the node names.
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    # A weight used by several nodes (the unrolled steps of a fused action head) reaches
    # the later ones through an Identity node.
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity" and node.input[0] in initializers
    }
    layers = []
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            channel_axis = 1  # weight is [in, out]
        elif node.op_type == "Gemm":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("transA", 0) or attrs.get("alpha", 1.0) != 1.0:
                continue
            channel_axis = 0 if attrs.get("transB", 0) else 1
        else:
            continue
        weight_name = aliases.get(node.input[1], node.input[1])
        tensor = initializers.get(weight_name)
        if tensor is None or len(tensor.dims) != 2 or tensor.data_type not in _ONNX_FLOAT_TYPES:
            continue
        if node.input[0] in initializers or node.input[0] in aliases:
            continue
        if only is not None and not re.search(only, node.name):
            continue
        if skip is not None and re.search(skip, node.name):
            continue
        layers.append(_Linear(node, weight_name, channel_axis))
    return layers


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _sample_lists(feeds) -> dict[str, list[torch.Tensor]]:
    """Normalize calibration data to ``{input: [one tensor per sample]}``.

    Accepts a stacked tensor ``[num_samples, ...]`` or a list of per-sample tensors per
    input (needed when samples differ in shape, e.g. prompts of different lengths).
    """
    lists = {name: list(value) for name, value in feeds.items()}
    counts = {len(v) for v in lists.values()}
    if len(counts) != 1:
        raise ValueError(f"Calibration inputs have different sample counts: {counts}")
    return lists


def _build_tap_engine(onnx_path: str, taps: list[str], feeds: dict[str, list[torch.Tensor]]):
    """Build ``onnx_path`` unmodified, with the tensors in ``taps`` as additional outputs."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(onnx_path):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Could not parse {onnx_path}: {errors}")

    tensors = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor is not None:
                tensors[tensor.name] = tensor
    existing = {network.get_output(i).name for i in range(network.num_outputs)}
    existing |= {network.get_input(i).name for i in range(network.num_inputs)}
    marked = []
    for name in taps:
        if name in existing:
            marked.append(name)
        elif name in tensors:
            network.mark_output(tensors[name])
            marked.append(name)

    config = builder.create_builder_config()
    # Calibration only needs correct values; do not spend build time on tactics.
    config.builder_optimization_level = 1
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        if name not in feeds:
            raise ValueError(
                f"Calibration data has no entry for graph input '{name}' of {onnx_path} "
                f"(has: {sorted(feeds)})"
            )
        shapes = [tuple(x.shape) for x in feeds[name]]
        low = tuple(min(dims) for dims in zip(*shapes))
        high = tuple(max(dims) for dims in zip(*shapes))
        profile.set_shape(name, low, max(set(shapes), key=shapes.count), high)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT could not build the calibration engine for {onnx_path}")
    engine = trt.Runtime(trt_logger).deserialize_cuda_engine(serialized)
    return engine, marked


_TRT_TO_TORCH = None


def _torch_dtype(trt_dtype):
    global _TRT_TO_TORCH
    if _TRT_TO_TORCH is None:
        import tensorrt as trt

        _TRT_TO_TORCH = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
            trt.DataType.INT8: torch.int8,
        }
    return _TRT_TO_TORCH[trt_dtype]


def _mse_optimal_clip(hist: torch.Tensor, max_value: float, levels: int = 127) -> float:
    """Clipping range for symmetric quantization that minimizes squared error.

    ``hist`` is a histogram of ``|x|`` over ``[0, max_value]``.
    """
    if max_value <= 0:
        return 1e-6
    hist = hist.double()
    bins = hist.numel()
    centers = (torch.arange(bins, dtype=torch.float64, device=hist.device) + 0.5) * (
        max_value / bins
    )
    # Candidate clips from 2% to 100% of the observed maximum.
    clips = torch.linspace(0.02, 1.0, 197, dtype=torch.float64, device=hist.device) * max_value
    step = clips / levels
    rounding = step**2 / 12.0  # uniform rounding noise inside the range
    over = (centers[None, :] - clips[:, None]).clamp_min(0.0) ** 2  # clipped values
    inside = (centers[None, :] <= clips[:, None]).double()
    error = (hist[None, :] * (inside * rounding[:, None] + over)).sum(dim=1)
    return float(clips[error.argmin()].item())


def calibrate(
    onnx_path: str,
    taps: list[str],
    feeds: dict[str, torch.Tensor | list[torch.Tensor]],
    weight_channel_max: dict[str, torch.Tensor] | None = None,
    smooth_alpha: float | None = None,
    device: str = "cuda",
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Run the calibration samples through ``onnx_path``.

    Returns ``({tensor: clip range}, {tensor: smoothing divisor})``. With ``smooth_alpha``
    each activation is first divided per channel by ``s = amax_x**a / amax_w**(1 - a)``
    (SmoothQuant), which moves outlier channels into the weights; ``weight_channel_max``
    holds, per tapped tensor, the largest weight magnitude per input channel over all the
    layers that read it. The clip range is computed on the divided tensor.
    """
    feeds = _sample_lists(feeds)
    engine, marked = _build_tap_engine(onnx_path, taps, feeds)
    missing = sorted(set(taps) - set(marked))
    if missing:
        logger.warning(
            "  %d linear inputs are not visible to TensorRT and stay BF16 (e.g. %s)",
            len(missing),
            missing[0],
        )
    context = engine.create_execution_context()
    import tensorrt as trt

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    num_samples = len(feeds[inputs[0]])
    stream = torch.cuda.Stream()
    buffers: dict[str, torch.Tensor] = {}
    bound_shapes: dict[str, tuple] = {}

    def run(index):
        sample = {name: feeds[name][index] for name in inputs}
        shapes = {name: tuple(x.shape) for name, x in sample.items()}
        if shapes != bound_shapes:  # (re)allocate for a new input shape
            for name, shape in shapes.items():
                context.set_input_shape(name, shape)
            for name in names:
                buffers[name] = torch.zeros(
                    tuple(context.get_tensor_shape(name)),
                    dtype=_torch_dtype(engine.get_tensor_dtype(name)),
                    device=device,
                )
                context.set_tensor_address(name, buffers[name].data_ptr())
            bound_shapes.clear()
            bound_shapes.update(shapes)
        for name, x in sample.items():
            buffers[name].copy_(x.to(device=device, dtype=buffers[name].dtype))
        torch.cuda.synchronize()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    def channels_last(name):
        # Graph inputs are read back from their (already copied) input buffers too.
        x = buffers[name].float()
        return x.reshape(-1, x.shape[-1])

    # Pass 1: per-channel range of every tensor.
    channel_max = {name: None for name in marked}
    for index in range(num_samples):
        run(index)
        for name in marked:
            m = channels_last(name).abs().amax(dim=0)
            prev = channel_max[name]
            channel_max[name] = m if prev is None else torch.maximum(prev, m)

    divisors = {}
    if smooth_alpha is not None:
        for name in marked:
            w = weight_channel_max[name].to(device).clamp_min(1e-5)
            x = channel_max[name].clamp_min(1e-5)
            divisors[name] = (x**smooth_alpha / w ** (1.0 - smooth_alpha)).clamp(1e-2, 1e2)
    maxima = {
        name: (channel_max[name] / divisors[name] if name in divisors else channel_max[name])
        .max()
        .item()
        for name in marked
    }

    # Pass 2: histogram of |x / s| over that range.
    hists = {name: torch.zeros(_HIST_BINS, device=device) for name in marked}
    for index in range(num_samples):
        run(index)
        for name in marked:
            if maxima[name] > 0:
                x = channels_last(name)
                if name in divisors:
                    x = x / divisors[name]
                hists[name] += torch.histc(x.abs().flatten(), _HIST_BINS, 0.0, maxima[name])
    clips = {name: _mse_optimal_clip(hists[name], maxima[name]) for name in marked}
    return clips, {name: d.cpu() for name, d in divisors.items()}


# ---------------------------------------------------------------------------
# Graph rewrite
# ---------------------------------------------------------------------------


def quantize_weight(weight: torch.Tensor, channel_axis: int):
    """Symmetric INT8, one scale per output channel. Returns ``(int8, scales, relative error)``."""
    reduce_axis = 1 - channel_axis
    scale = (weight.abs().amax(dim=reduce_axis) / 127.0).clamp_min(1e-8)
    scale = scale.to(torch.bfloat16).float()  # the value TensorRT dequantizes with
    expanded = scale.unsqueeze(reduce_axis)
    quantized = torch.clamp(torch.round(weight / expanded), -127, 127)
    error = ((quantized * expanded - weight) ** 2).sum() / (weight**2).sum().clamp_min(1e-30)
    return quantized.to(torch.int8), scale, error.sqrt().item()


def quantize_model(
    model: onnx.ModelProto,
    layers: list[_Linear],
    clips: dict[str, float],
    divisors: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Rewrite ``layers`` of ``model`` in place as INT8 Q/DQ.

    ``clips``: activation clip ranges. ``divisors``: optional per-channel SmoothQuant
    divisors; the activation is multiplied by ``1 / s`` before quantization and the
    weights' input channels by ``s``, which leaves the layer's output unchanged.
    """
    divisors = divisors or {}
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    by_node = {id(layer.node): layer for layer in layers}
    weight_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    weight_uses: dict[str, int] = {}
    activation_dq: dict[str, str] = {}
    new_nodes, errors, params, quantized_layers = [], [], 0, 0

    for node in graph.node:
        layer = by_node.get(id(node))
        activation = node.input[0] if layer is not None else None
        if layer is None or activation not in clips:
            new_nodes.append(node)
            continue
        tensor = initializers[layer.weight_name]
        data_type = tensor.data_type

        # Activation: one Q/DQ pair per tensor, shared by all linear layers reading it.
        if activation not in activation_dq:
            prefix = f"{activation}_int8"
            source = activation
            if activation in divisors:
                graph.initializer.append(
                    _float_tensor(prefix + "_smooth", 1.0 / divisors[activation], data_type)
                )
                new_nodes.append(
                    helper.make_node(
                        "Mul",
                        [activation, prefix + "_smooth"],
                        [prefix + "_smoothed"],
                        name=prefix + "_smooth_mul",
                    )
                )
                source = prefix + "_smoothed"
            scale = torch.tensor(clips[activation] / 127.0)
            graph.initializer.append(_float_tensor(prefix + "_scale", scale, data_type))
            graph.initializer.append(
                _int8_tensor(prefix + "_zero", torch.zeros((), dtype=torch.int8))
            )
            new_nodes.append(
                helper.make_node(
                    "QuantizeLinear",
                    [source, prefix + "_scale", prefix + "_zero"],
                    [prefix + "_q"],
                    name=prefix + "_quantize",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [prefix + "_q", prefix + "_scale", prefix + "_zero"],
                    [prefix + "_dq"],
                    name=prefix + "_dequantize",
                )
            )
            activation_dq[activation] = prefix + "_dq"

        # Weight: every consumer gets its own INT8 copy (TensorRT does not fuse a
        # dequantized constant that is shared between layers).
        cache_key = (layer.weight_name, activation if activation in divisors else None)
        if cache_key not in weight_cache:
            weight = _read_float_tensor(tensor)
            if activation in divisors:
                # Input channels are the non-output axis of the weight.
                input_axis = 1 - layer.channel_axis
                shape = [1, 1]
                shape[input_axis] = -1
                weight = weight * divisors[activation].reshape(shape)
            quantized, scale, error = quantize_weight(weight, layer.channel_axis)
            weight_cache[cache_key] = (quantized, scale)
            errors.append(error)
            params += quantized.numel()
        quantized, scale = weight_cache[cache_key]
        index = weight_uses.get(layer.weight_name, 0)
        weight_uses[layer.weight_name] = index + 1
        prefix = f"{layer.weight_name}_use{index}"
        graph.initializer.append(_int8_tensor(prefix + "_int8", quantized))
        graph.initializer.append(_float_tensor(prefix + "_scale", scale, data_type))
        graph.initializer.append(_int8_tensor(prefix + "_zero", torch.zeros_like(scale)))
        new_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [prefix + "_int8", prefix + "_scale", prefix + "_zero"],
                [prefix + "_dequantized"],
                name=prefix + "_dequantize",
                axis=layer.channel_axis,
            )
        )
        node.input[0] = activation_dq[activation]
        node.input[1] = prefix + "_dequantized"
        new_nodes.append(node)
        quantized_layers += 1

    # Drop Identity aliases and initializers that lost their last consumer.
    outputs = {out.name for out in graph.output}
    consumed = {name for node in new_nodes for name in node.input}
    new_nodes = [
        node
        for node in new_nodes
        if node.op_type != "Identity" or node.output[0] in consumed or node.output[0] in outputs
    ]
    used = {name for node in new_nodes for name in node.input}
    kept = [init for init in graph.initializer if init.name in used or init.name in outputs]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.node[:]
    graph.node.extend(new_nodes)
    for opset in model.opset_import:
        if opset.domain == "" and opset.version < _MIN_OPSET:
            opset.version = _MIN_OPSET
    model.ir_version = max(model.ir_version, 10)

    return {
        "layers": quantized_layers,
        "params": params,
        "activations": len(activation_dq),
        "mean_weight_error": float(np.mean(errors)) if errors else 0.0,
        "max_weight_error": float(np.max(errors)) if errors else 0.0,
    }


def quantize_file(
    input_path: str,
    output_path: str,
    feeds: dict[str, torch.Tensor | list[torch.Tensor]],
    only: str | None = None,
    skip: str | None = None,
    smooth_alpha: float | None = 0.7,
) -> dict:
    """Calibrate and quantize one ONNX file. Weights go to ``<output_path>.data``."""
    model = onnx.load(input_path, load_external_data=True)
    layers = find_linear_layers(model, only, skip)
    if not layers:
        raise ValueError(f"No Gemm/MatMul with a constant weight matched in {input_path}")
    taps = sorted({layer.node.input[0] for layer in layers})
    logger.info(
        "%s: calibrating %d activation tensors of %d linear layers on %d samples...",
        os.path.basename(input_path),
        len(taps),
        len(layers),
        min(len(v) for v in feeds.values()),
    )
    weight_channel_max = None
    if smooth_alpha is not None:
        initializers = {init.name: init for init in model.graph.initializer}
        weight_channel_max = {}
        for layer in layers:
            weight = _read_float_tensor(initializers[layer.weight_name])
            per_input = weight.abs().amax(dim=layer.channel_axis)
            name = layer.node.input[0]
            prev = weight_channel_max.get(name)
            weight_channel_max[name] = per_input if prev is None else torch.maximum(prev, per_input)
    clips, divisors = calibrate(input_path, taps, feeds, weight_channel_max, smooth_alpha)
    summary = quantize_model(model, layers, clips, divisors)

    data_name = os.path.basename(output_path) + ".data"
    data_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), data_name)
    for path in (output_path, data_path):
        # Never write through a symlink (e.g. into the exported graph it points at), and
        # never append to an old external-data file.
        if os.path.lexists(path):
            os.remove(path)
    onnx.save(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )
    logger.info(
        "%s: %d linear layers (%.0fM parameters) -> INT8, %d quantized activations. "
        "Relative weight error: mean %.4f, max %.4f. %.0f MB of weights.",
        os.path.basename(output_path),
        summary["layers"],
        summary["params"] / 1e6,
        summary["activations"],
        summary["mean_weight_error"],
        summary["max_weight_error"],
        os.path.getsize(data_path) / 1e6,
    )
    return summary


@dataclass
class QuantizeConfig:
    """Quantize the linear layers of one exported ONNX graph to INT8."""

    onnx: str
    """Graph to quantize (e.g. gr00t_trt_deployment/onnx/vit_bf16.onnx)."""

    output: str
    """Where to write the quantized graph (its weights go to <output>.data)."""

    calibration_data: str
    """torch.save-d dict {graph input name: tensor [num_samples, ...] or list of per-sample
    tensors} of real inputs."""

    only: str | None = None
    """Only quantize layers whose node name matches this regular expression."""

    skip: str | None = None
    """Leave layers whose node name matches this regular expression in BF16."""

    smooth_alpha: float | None = 0.7
    """SmoothQuant migration strength (0-1); None disables smoothing. 0.7 was best for the
    N1.7 ViT in a sweep of 0.3-0.85."""


def main(cfg: QuantizeConfig) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.path.abspath(cfg.onnx) == os.path.abspath(cfg.output):
        raise ValueError("--output must differ from --onnx")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
    feeds = torch.load(cfg.calibration_data, map_location="cpu")
    return quantize_file(cfg.onnx, cfg.output, feeds, cfg.only, cfg.skip, cfg.smooth_alpha)


if __name__ == "__main__":
    main(tyro.cli(QuantizeConfig))
