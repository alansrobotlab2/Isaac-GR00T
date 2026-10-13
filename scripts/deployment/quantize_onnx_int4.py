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

"""Store an ONNX graph's linear-layer weights as INT4 (TensorRT weight-only quantization).

The DiT in the action head has about 1.1 billion weights but only 41 query tokens, so on
a Jetson AGX Orin a large part of each denoising step is spent streaming weights from
memory. TensorRT can keep ``MatMul`` weights as 4-bit integers and dequantize them inside
the GEMM kernel, which cuts that traffic by 4x. Activations stay BF16.

This is an opt-in experiment, not part of the default pipeline: it changes the weights.
Measured on GR00T-N1.7-3B / DROID on an AGX Orin (2 denoising steps): the DiT engine went
from 19.3 to 17.1 ms per call, open-loop MSE was unchanged (0.01672), and the worst-case
cosine similarity of the action chunk against PyTorch dropped from 0.99992 to 0.99951.
Validate on your own checkpoint and task before relying on it.

Usage, after ``build_trt_pipeline.py --steps export`` (or a full run)::

    python scripts/deployment/quantize_onnx_int4.py \\
        --onnx-dir gr00t_trt_deployment/onnx --output-dir gr00t_trt_deployment/onnx_int4
    python scripts/deployment/build_tensorrt_engine.py --mode full_pipeline \\
        --onnx-dir gr00t_trt_deployment/onnx_int4 --engine-dir gr00t_trt_deployment/engines_int4

The output directory mirrors the input one: the DiT (``dit_bf16.onnx``) and the DiT inside
any fused action head (``action_head_fused_<N>step.onnx``) are rewritten, every other file
is symlinked, so the regular builder produces a complete engine directory to pass as
``--trt-engine-path``.

Keep ``--block-size 128``: TensorRT 10.16 only has a fast INT4 kernel for that block size
(block 32 was four times slower than no quantization at all on Orin).
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

# Clipping ratios tried per block; the one with the lowest squared error wins.
_CLIP_RATIOS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
# DequantizeLinear gained ``block_size`` and INT4 in opset 21.
_MIN_OPSET = 21


def _read_bf16(tensor: onnx.TensorProto) -> torch.Tensor:
    raw = np.frombuffer(tensor.raw_data, dtype=np.int16).copy()
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(tuple(tensor.dims)).float()


def _bf16_tensor(name: str, values: torch.Tensor) -> onnx.TensorProto:
    raw = values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    return helper.make_tensor(name, TensorProto.BFLOAT16, list(values.shape), raw, raw=True)


def _int4_tensor(name: str, values: torch.Tensor) -> onnx.TensorProto:
    """Pack int8 values in [-8, 7] two per byte, low nibble first (ONNX INT4 raw layout)."""
    nibbles = values.flatten().numpy().astype(np.uint8) & 0x0F
    if nibbles.size % 2:
        nibbles = np.append(nibbles, np.uint8(0))
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8).tobytes()
    return helper.make_tensor(name, TensorProto.INT4, list(values.shape), packed, raw=True)


def quantize_weight(weight: torch.Tensor, block_size: int):
    """Quantize a ``[K, N]`` MatMul weight to INT4 with one scale per (block of K, column).

    Returns ``(int8 values in [-8, 7], scales [K / block_size, N], relative error)``.
    """
    k, n = weight.shape
    blocks = weight.reshape(k // block_size, block_size, n)
    amax = blocks.abs().amax(dim=1).clamp_min(1e-8)
    best_err = torch.full_like(amax, float("inf"))
    best_scale = amax / 7.0
    for ratio in _CLIP_RATIOS:
        # Round the scale to BF16 first: that is the value TensorRT dequantizes with.
        scale = (amax * ratio / 7.0).to(torch.bfloat16).float()
        quantized = torch.clamp(torch.round(blocks / scale[:, None, :]), -8, 7)
        err = ((quantized * scale[:, None, :] - blocks) ** 2).sum(dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
    quantized = torch.clamp(torch.round(blocks / best_scale[:, None, :]), -8, 7)
    relative_error = (best_err.sum() / (weight**2).sum().clamp_min(1e-30)).sqrt().item()
    return quantized.to(torch.int8).reshape(k, n), best_scale, relative_error


def quantize_model(
    model: onnx.ModelProto,
    block_size: int = 128,
    only: str | None = None,
    skip: str | None = None,
) -> dict:
    """Rewrite constant BF16 ``MatMul`` weights of ``model`` in place as INT4 + DequantizeLinear.

    ``only`` / ``skip`` are regular expressions matched against MatMul node names.
    Returns a summary dict (layers, parameters, mean and max relative weight error).
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    # A weight used more than once (e.g. by every unrolled denoising step of a fused
    # action head) reaches its later uses through an Identity node; resolve those so
    # all uses share the one INT4 tensor.
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity" and node.input[0] in initializers
    }
    uses: dict[str, int] = {}  # quantized weight -> number of MatMul consumers so far
    quantized_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    new_nodes, errors, params = [], [], 0

    for node in graph.node:
        weight_name = node.input[1] if node.op_type == "MatMul" and len(node.input) > 1 else None
        weight_name = aliases.get(weight_name, weight_name)
        tensor = initializers.get(weight_name)
        if tensor is not None and weight_name not in uses:
            eligible = (
                len(tensor.dims) == 2
                and tensor.data_type == TensorProto.BFLOAT16
                and tensor.dims[0] % block_size == 0
                and (only is None or re.search(only, node.name))
                and not (skip is not None and re.search(skip, node.name))
            )
            if eligible:
                weight = _read_bf16(tensor)
                quantized, scale, relative_error = quantize_weight(weight, block_size)
                quantized_weights[weight_name] = (quantized, scale)
                uses[weight_name] = 0
                errors.append(relative_error)
                params += weight.numel()
        if weight_name in uses:
            # Every consumer gets its own INT4 tensor and DequantizeLinear. TensorRT
            # 10.16 only fuses the dequantize into the GEMM for an unshared constant; a
            # weight shared by two MatMuls (the unrolled steps of a fused action head)
            # fails to build. The copies cost 0.5 byte per weight per extra use.
            index = uses[weight_name]
            uses[weight_name] += 1
            quantized, scale = quantized_weights[weight_name]
            prefix = f"{weight_name}_use{index}"
            graph.initializer.append(_int4_tensor(prefix + "_int4", quantized))
            graph.initializer.append(_bf16_tensor(prefix + "_scale", scale))
            new_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [prefix + "_int4", prefix + "_scale"],
                    [prefix + "_dequantized"],
                    name=prefix + "_dequantize",
                    axis=0,
                    block_size=block_size,
                )
            )
            node.input[1] = prefix + "_dequantized"
        new_nodes.append(node)

    # Identity aliases of replaced weights have no consumers left.
    consumed = {name for node in new_nodes for name in node.input}
    new_nodes = [
        node
        for node in new_nodes
        if not (node.op_type == "Identity" and node.output[0] not in consumed)
        or node.output[0] in {out.name for out in graph.output}
    ]
    used = {name for node in new_nodes for name in node.input}
    kept = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.node[:]
    graph.node.extend(new_nodes)
    for opset in model.opset_import:
        if opset.domain == "" and opset.version < _MIN_OPSET:
            opset.version = _MIN_OPSET
    model.ir_version = max(model.ir_version, 10)

    return {
        "layers": len(uses),
        "params": params,
        "mean_relative_error": float(np.mean(errors)) if errors else 0.0,
        "max_relative_error": float(np.max(errors)) if errors else 0.0,
    }


# Graphs to rewrite, with the MatMul-name filter that selects the DiT's layers in each.
_TARGETS = (
    (re.compile(r"dit_bf16\.onnx"), None),
    (re.compile(r"action_head_fused_\d+step\.onnx"), "/model/"),
)


def quantize_file(
    input_path: str,
    output_path: str,
    block_size: int = 128,
    only: str | None = None,
    skip: str | None = None,
) -> dict:
    """Quantize one ONNX file; weights are written to ``<output_path>.data``."""
    model = onnx.load(input_path, load_external_data=True)
    summary = quantize_model(model, block_size, only, skip)
    if summary["layers"] == 0:
        raise ValueError(f"No BF16 MatMul weights matched in {input_path}; nothing to quantize")
    data_name = os.path.basename(output_path) + ".data"
    data_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), data_name)
    if os.path.exists(data_path):
        os.remove(data_path)  # onnx appends to an existing external-data file
    onnx.save(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    logger.info(
        "%s: %d MatMul weights (%.0fM parameters) -> INT4, block size %d. Relative weight "
        "error: mean %.3f, max %.3f. %.0f MB of weights.",
        os.path.basename(output_path),
        summary["layers"],
        summary["params"] / 1e6,
        block_size,
        summary["mean_relative_error"],
        summary["max_relative_error"],
        os.path.getsize(data_path) / 1e6,
    )
    return summary


@dataclass
class QuantizeConfig:
    """Mirror an exported ONNX directory with the DiT's weights stored as INT4."""

    onnx_dir: str
    """Directory written by the export step (contains dit_bf16.onnx)."""

    output_dir: str
    """New directory to create; pass it to build_tensorrt_engine.py as --onnx-dir."""

    block_size: int = 128
    """Rows of the weight that share one scale per column. Keep 128 (see module docstring)."""

    skip: str | None = None
    """Leave MatMul nodes whose name matches this regular expression in BF16."""


def main(cfg: QuantizeConfig) -> dict[str, dict]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src, dst = os.path.abspath(cfg.onnx_dir), os.path.abspath(cfg.output_dir)
    if src == dst:
        raise ValueError("--output-dir must differ from --onnx-dir")
    os.makedirs(dst, exist_ok=True)

    names = sorted(os.listdir(src))
    targets = {
        name: only for name in names for pattern, only in _TARGETS if pattern.fullmatch(name)
    }
    if not targets:
        raise ValueError(f"No dit_bf16.onnx or fused action-head graph found in {src}")

    summaries = {}
    for name in names:
        out_path = os.path.join(dst, name)
        if name in targets:
            summaries[name] = quantize_file(
                os.path.join(src, name), out_path, cfg.block_size, targets[name], cfg.skip
            )
        elif name[: -len(".data")] not in targets:
            # Everything else (other graphs, their weights, export_metadata.json) is shared.
            if os.path.lexists(out_path):
                os.remove(out_path)
            os.symlink(os.path.join(src, name), out_path)
    logger.info("Wrote %s; build it with build_tensorrt_engine.py --mode full_pipeline.", dst)
    return summaries


if __name__ == "__main__":
    main(tyro.cli(QuantizeConfig))
