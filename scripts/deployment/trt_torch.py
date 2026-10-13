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

"""TensorRT Engine wrapper for GR00T inference.

Loads serialized TRT engines, manages input/output tensor bindings, and
executes inference. Supports dynamic shapes and BF16/FP16/FP32 dtypes.

Two opt-in runtime modes cut host overhead on devices with a slow CPU (Jetson):

* ``sync=False`` drops the ``stream.synchronize()`` after every engine call. Outputs
  are ordinary CUDA tensors on the caller's stream, so later torch ops (or a final
  ``.cpu()``) are ordered after the engine without blocking the host in between.
* ``cuda_graph=True`` captures the engine's kernel launches into a CUDA graph on first
  use and replays it afterwards, which replaces hundreds of per-layer launches with one.
"""

import ctypes
import logging
import os

import tensorrt as trt
import torch


logger = logging.getLogger(__name__)

# A CUDA graph is re-captured when an engine's input shapes change (e.g. a prompt with a
# different token count). Past this many captures the engine stops using graphs.
_MAX_GRAPH_CAPTURES = 16


def torch_type(trt_type):
    """Convert TensorRT data type to PyTorch equivalent."""
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve TensorRT datatype to an equivalent PyTorch datatype. {trt_type}"
    )


class Engine(object):
    """TensorRT engine wrapper for loading and executing inference.

    Args:
        file: Serialized engine path.
        plugins: Optional plugin libraries to load first.
        sync: Block on the CUDA stream after every call (the conservative default).
            ``False`` returns as soon as the work is enqueued; the engine keeps its
            inputs alive until the next call, and outputs stay stream-ordered.
        cuda_graph: Replay the engine through a captured CUDA graph. Inputs are copied
            into fixed buffers and **outputs are fixed buffers that the next call
            overwrites** -- consume or clone them before calling the engine again. The
            graph is re-captured when input shapes change. Engines that cannot be
            captured (for example ones with data-dependent shapes) fall back to plain
            execution with a warning.
    """

    def __init__(self, file, plugins=[], sync=True, cuda_graph=False):
        super().__init__()

        self._closed = False
        self.execution_context = None
        self.handle = None
        self.sync = sync
        self.cuda_graph = cuda_graph
        self._graph = None  # _CapturedGraph for the input shapes in _graph_key
        self._graph_key = None
        self._graph_captures = 0
        self._graph_failed = False
        self._shapes = {}  # last shape set per input, to skip redundant TRT calls
        self._held = None  # tensors of the last sync=False call, kept alive until the next

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        self.print()

    def close(self):
        """Release the execution context, then the engine handle.

        TensorRT requires the context be dropped before the engine.
        Idempotent so it composes safely with ``__del__``.
        """
        if self._closed:
            return
        self._closed = True
        # Graphs hold kernels that reference the context's memory; drop them first.
        self._graph, self._graph_key = None, None
        self._held = None
        self.execution_context = None
        self.handle = None

    def __del__(self):
        # tensorrt / CUDA context may already be gone at shutdown; swallow
        # so the traceback doesn't surface as a spurious error on exit.
        try:
            self.close()
        except Exception:
            pass

    def print(self):
        """Display engine details (inputs/outputs) on rank 0 only."""
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        """Deserialize and load a TensorRT engine from file."""
        runtime = trt.Runtime(self.logger)

        try:
            with open(file, "rb") as f:
                self.handle = runtime.deserialize_cuda_engine(f.read())
                assert self.handle is not None, (
                    f"Failed to deserialize the cuda engine from file: {file}"
                )

            self.execution_context = self.handle.create_execution_context()
            self.meta, self.in_meta, self.out_meta = [], [], []
            for tensor_name in self.handle:
                shape = self.handle.get_tensor_shape(tensor_name)
                dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
                if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.in_meta.append([tensor_name, shape, dtype])
                else:
                    self.out_meta.append([tensor_name, shape, dtype])
        except BaseException:
            # Roll back a half-loaded engine (e.g. create_execution_context
            # failed after the handle was set) so it can't leak GPU memory.
            self.close()
            raise

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def dtype_of(self, tensor_name: str) -> torch.dtype:
        """Return the expected PyTorch dtype for a named input tensor."""
        for name, _shape, dtype in self.in_meta:
            if name == tensor_name:
                return dtype
        raise KeyError(f"Input tensor '{tensor_name}' not found in engine.")

    def set_runtime_tensor_shape(self, name, shape):
        """Set runtime input shape for dynamic dimensions (no-op if unchanged)."""
        shape = tuple(shape)
        if self._shapes.get(name) == shape:
            return
        self.execution_context.set_input_shape(name, shape)
        self._shapes[name] = shape

    def _bind_inputs(self, args, kwargs):
        """Validate the call's tensors and return them as ``[(name, tensor)]`` in engine order."""
        bound = []
        for iarg, x in enumerate(args):
            bound.append((self.in_meta[iarg], x))
        for meta in self.in_meta[len(args) :]:
            if meta[0] in kwargs:
                bound.append((meta, kwargs[meta[0]]))
        assert len(bound) == len(self.in_meta), (
            f"Invalid input tensors. The engine expects {len(self.in_meta)} inputs "
            f"({[m[0] for m in self.in_meta]}), but got {len(bound)}"
        )

        inputs = []
        for (name, static_shape, dtype), x in bound:
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            expected = self._shapes.get(name)
            if expected is None:
                expected = tuple(static_shape)
                if -1 in expected:
                    expected = tuple(self.execution_context.get_tensor_shape(name))
            assert expected == tuple(x.shape), (
                f"Invalid input[{name}] shape: {tuple(x.shape)}, but the expected shape is: {expected}"
            )
            assert dtype == x.dtype, (
                f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            )
            assert x.is_cuda, (
                f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            )
            inputs.append((name, x.contiguous()))
        return inputs

    def _capture_graph(self, inputs):
        """Capture one execution into a CUDA graph with fixed I/O buffers.

        Returns ``None`` (and disables graphs for this engine) if capture fails.
        """
        context = self.execution_context
        device = inputs[0][1].device
        in_bufs = [(name, x.clone()) for name, x in inputs]
        out_bufs = {}
        for name, _shape, dtype in self.out_meta:
            out_bufs[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=dtype, device=device
            )
        for name, buf in in_bufs:
            context.set_tensor_address(name, buf.data_ptr())
        for name, buf in out_bufs.items():
            context.set_tensor_address(name, buf.data_ptr())

        # TensorRT must run once at these shapes before capture so that it does not
        # record its lazy shape-dependent initialisation into the graph.
        stream = torch.cuda.current_stream()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                # torch captures on its own stream, which is current inside this block.
                ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 failed during CUDA graph capture")
        except Exception as e:  # noqa: BLE001 - any capture failure means "run eagerly"
            self._graph_failed = True
            logger.warning(
                "CUDA graph capture failed for %s (%s); running it without a graph. Engines "
                "with data-dependent shapes cannot be captured.",
                os.path.basename(self.file),
                str(e).splitlines()[0],
            )
            return None
        return _CapturedGraph(graph, in_bufs, out_bufs)

    def _forward_graph(self, inputs):
        key = tuple(tuple(x.shape) for _, x in inputs)
        # Graph buffers are created and written under inference mode whatever the caller uses.
        with torch.inference_mode():
            if self._graph_key != key:
                # One graph at a time: a graph bakes in addresses of the context's
                # scratch memory, which TensorRT may lay out differently per shape.
                self._graph, self._graph_key = None, None
                if self._graph_captures >= _MAX_GRAPH_CAPTURES:
                    self._graph_failed = True
                    logger.warning(
                        "%s: input shapes changed %d times; CUDA graph disabled for this "
                        "engine to avoid re-capturing on every call.",
                        os.path.basename(self.file),
                        self._graph_captures,
                    )
                    return None
                self._graph_captures += 1
                self._graph = self._capture_graph(inputs)
                if self._graph is None:
                    return None
                self._graph_key = key
            for (_, x), (_, buf) in zip(inputs, self._graph.inputs):
                buf.copy_(x)
            self._graph.graph.replay()
            return self._graph.outputs

    def forward(self, *args, **kwargs):
        """Execute TRT inference with the given input tensors.

        Accepts both positional and keyword arguments.
        Returns a dict of output tensors by default, or a list if return_list=True.
        """
        return_list = kwargs.pop("return_list", False)
        inputs = self._bind_inputs(args, kwargs)
        stream = torch.cuda.current_stream()

        outputs = None
        if self.cuda_graph and not self._graph_failed:
            outputs = self._forward_graph(inputs)

        if outputs is None:
            context = self.execution_context
            for name, x in inputs:
                context.set_tensor_address(name, x.data_ptr())
            outputs = {}
            for name, _shape, dtype in self.out_meta:
                out = torch.empty(
                    tuple(context.get_tensor_shape(name)), dtype=dtype, device=inputs[0][1].device
                )
                context.set_tensor_address(name, out.data_ptr())
                outputs[name] = out
            context.execute_async_v3(stream.cuda_stream)
            # Without a sync the kernels may still be reading the inputs after we return.
            self._held = None if self.sync else [x for _, x in inputs]

        if self.sync:
            stream.synchronize()

        if return_list:
            return [outputs[item[0]] for item in self.out_meta]
        return dict(outputs)


class _CapturedGraph:
    """A captured CUDA graph plus the fixed I/O buffers its kernels read and write."""

    __slots__ = ("graph", "inputs", "outputs")

    def __init__(self, graph, inputs, outputs):
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
