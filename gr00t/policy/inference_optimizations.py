"""Inference-time optimizations for ``Gr00tPolicy`` on edge devices (Jetson Orin / Thor).

Ported from the N1.6 Orin AGX optimization work (see ``INFERENCE_OPTIMIZATIONS.md``):

* :func:`apply_inference_optimizations` -- torch.compile for the backbone / DiT /
  action encoder+decoder, cuDNN benchmark, TF32, and a runtime action-horizon override.
* :class:`PipelinedInference` -- enqueue backbone(N+1) on a side CUDA stream right after
  DiT(N) so it overlaps with frame N's CPU-side action decoding.
* :func:`load_tensorrt_engines` -- install full-pipeline TensorRT engines on a policy.
* :func:`run_frames` -- drives a one-frame-lookahead prefetching inference loop
  (optionally pipelined) and reports per-frame timings.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy


CompileMode = Literal["default", "reduce-overhead", "max-autotune"]

# A prepared frame: (collated_inputs, states, payload). collated_inputs/states are None for
# a remote policy (PolicyClient); payload is then the parsed observation to send.
Frame = tuple[dict[str, Any] | None, list[dict[str, Any]] | None, Any]


def _forward_is_patched(module: torch.nn.Module) -> bool:
    """True if ``module.forward`` was replaced on the instance (TensorRT engine or torch.compile)."""
    return "forward" in vars(module)


def _uses_flash_attention(backbone: torch.nn.Module) -> bool:
    """True if any submodule of the backbone is configured for flash attention."""
    for module in backbone.modules():
        config = getattr(module, "config", None)
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            return True
    return False


def apply_inference_optimizations(
    policy: BasePolicy,
    *,
    compile_backbone: bool = False,
    compile_dit: bool = False,
    compile_action_head: bool = False,
    compile_mode: CompileMode = "default",
    cudnn_benchmark: bool = False,
    tf32: bool = False,
    model_action_horizon: int | None = None,
) -> None:
    """Apply optional inference-time optimizations to a local ``Gr00tPolicy`` in place.

    Call this *after* any TensorRT engines are installed (``setup_tensorrt_engines`` /
    ``replace_dit_with_tensorrt``): components already replaced by TRT are skipped.
    For a remote policy (``PolicyClient``) this is a no-op with a warning.
    """
    requested = {
        "compile_backbone": compile_backbone,
        "compile_dit": compile_dit,
        "compile_action_head": compile_action_head,
        "cudnn_benchmark": cudnn_benchmark,
        "tf32": tf32,
        "model_action_horizon": model_action_horizon,
    }
    model = getattr(policy, "model", None)
    if model is None:
        active = [name for name, value in requested.items() if value]
        if active:
            logging.warning(
                "Inference optimizations %s are ignored for a remote policy; "
                "apply them on the server (run_gr00t_server.py) instead.",
                active,
            )
        return
    action_head = model.action_head

    if tf32:
        # TF32 tensor cores for the FP32 matmuls left in the pipeline (state/action
        # encoders, Euler update). The legacy flags are used on purpose: inductor still
        # reads them for its cache key, and mixing them with the newer
        # torch.backends.*.fp32_precision API raises under torch.compile.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("TF32 matmul/cuDNN enabled")
    if cudnn_benchmark:
        # Fixed input shapes at inference -> let cuDNN autotune conv algorithms once.
        torch.backends.cudnn.benchmark = True
        logging.info("cuDNN benchmark mode enabled (autotunes on first inference)")

    if model_action_horizon is not None:
        original = action_head.action_horizon
        action_head.config.action_horizon = model_action_horizon
        action_head.action_horizon = model_action_horizon
        logging.info("Overriding model action_horizon: %d -> %d", original, model_action_horizon)

    if compile_backbone:
        backbone = model.backbone
        if _forward_is_patched(backbone):
            logging.info("Backbone forward is already replaced (TensorRT); skipping torch.compile")
        elif _uses_flash_attention(backbone):
            # Dynamo cannot trace flash_attn's varlen kernel in the Qwen3-VL backbone
            # (it is handed a tensor where the op schema wants an int) and the failure
            # only surfaces on the first inference. SDPA traces fine.
            logging.warning(
                "--compile-backbone is skipped: torch.compile cannot trace the flash-attention "
                "backbone. Load the policy with attn_implementation='sdpa' "
                "(--attn-implementation sdpa) to compile it."
            )
        else:
            backbone.forward = torch.compile(backbone.forward, mode=compile_mode)
            logging.info(
                "Backbone compiled with torch.compile(mode=%r); warms up on first inference",
                compile_mode,
            )

    if compile_dit:
        dit = getattr(action_head, "model", None)
        if dit is None or _forward_is_patched(dit):
            logging.info("DiT is already replaced (TensorRT); skipping torch.compile")
        else:
            dit.forward = torch.compile(dit.forward, mode=compile_mode)
            logging.info(
                "DiT compiled with torch.compile(mode=%r); warms up on first inference",
                compile_mode,
            )

    if compile_action_head:
        for name in ("action_encoder", "action_decoder"):
            module = getattr(action_head, name, None)
            if module is None:
                logging.info("%s is already replaced (TensorRT); skipping torch.compile", name)
                continue
            setattr(action_head, name, torch.compile(module, mode="default"))
            logging.info("%s compiled with torch.compile(mode='default')", name)


@dataclass
class _PendingFrame:
    backbone_output: Any
    action_inputs: Any
    done_event: torch.cuda.Event


class PipelinedInference:
    """Overlap backbone(N+1) with the CPU tail of frame N.

    Sequential::

        backbone(N) -> DiT(N) -> decode(N) -> backbone(N+1) -> DiT(N+1) -> decode(N+1)

    Pipelined::

        backbone(N) -> DiT(N) -> [backbone(N+1) on side stream] || decode(N) -> DiT(N+1) ...

    The next frame's backbone is enqueued on a side CUDA stream right after DiT(N), so the
    GPU keeps working while frame N's actions are unnormalized on the CPU. On a single-GPU
    unified-memory device (Orin) this is CPU/GPU overlap, not GPU/GPU concurrency -- it
    was worth ~10% on N1.6 (INFERENCE_OPTIMIZATIONS.md sections 6 and 14).

    Frames must be fed strictly in order through :meth:`step`; frame inputs come from
    ``Gr00tPolicy.prepare_inputs``.
    """

    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.model = policy.model
        self.stream = torch.cuda.Stream()
        self._pending: _PendingFrame | None = None

    def start_backbone_async(self, collated_inputs: dict[str, Any]) -> None:
        """Enqueue the backbone for the next frame on the side stream without waiting."""
        inputs = collated_inputs.get("inputs", collated_inputs)
        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(inputs)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                backbone_output = self.model.backbone(backbone_inputs)
        self._pending = _PendingFrame(backbone_output, action_inputs, self.stream.record_event())

    def _finish_pending(self):
        pending, self._pending = self._pending, None
        current = torch.cuda.current_stream()
        current.wait_event(pending.done_event)
        # The backbone outputs were allocated on the side stream; tell the caching
        # allocator they are consumed here so their memory is not recycled early.
        values = getattr(pending.backbone_output, "values", None)
        if values is not None:
            for value in values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(current)
        with torch.inference_mode():
            return self.model.action_head.get_action(pending.backbone_output, pending.action_inputs)

    def step(
        self,
        collated_inputs: dict[str, Any],
        states: list[dict[str, Any]],
        next_frame: Future[Frame] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run frame N (whose backbone may already be in flight), then launch backbone(N+1).

        ``next_frame`` resolves to the next frame's ``(collated_inputs, states, payload)``;
        its backbone is enqueued *before* frame N's actions are decoded on the CPU.
        """
        if self._pending is None:
            # First frame (nothing in flight yet): run sequentially.
            model_pred = self.policy.run_model(collated_inputs)
        else:
            model_pred = self._finish_pending()
        if next_frame is not None:
            next_collated_inputs, _, _ = next_frame.result()
            self.start_backbone_async(next_collated_inputs)
        return self.policy.decode_model_output(model_pred, states)


def run_frames(
    policy: BasePolicy,
    frame_futures: Iterator[Future[Frame]],
    pipeline: PipelinedInference | None = None,
) -> Iterator[tuple[dict[str, np.ndarray], Any, float, float]]:
    """Drive a prefetching inference loop over prepared frames.

    ``frame_futures`` yields one Future per frame, in order. Pulling the next future is
    what submits its CPU preprocessing, so this loop keeps exactly one frame of lookahead
    in flight while the GPU works on the current one.

    Yields ``(actions, payload, data_prep_wait_s, inference_s)`` per frame.
    """
    current = next(frame_futures, None)
    while current is not None:
        wait_start = time.perf_counter()
        collated_inputs, states, payload = current.result()
        data_prep_wait = time.perf_counter() - wait_start

        upcoming = next(frame_futures, None)  # submits preprocessing for N+1 now

        inference_start = time.perf_counter()
        if collated_inputs is None:
            # Remote policy: preprocessing happens server-side.
            actions, _ = policy.get_action(payload)
        elif pipeline is not None:
            actions = pipeline.step(collated_inputs, states, upcoming)
        else:
            actions = policy.run_inference(collated_inputs, states)
        inference_time = time.perf_counter() - inference_start

        yield actions, payload, data_prep_wait, inference_time
        current = upcoming


def load_tensorrt_engines(
    policy: Gr00tPolicy,
    engine_dir: str,
    *,
    use_vit_engine: bool = True,
    batch_size: int = 1,
    cuda_graph: bool = True,
) -> None:
    """Swap a local ``Gr00tPolicy``'s backbone and action head for full-pipeline TRT engines.

    ``engine_dir`` is the ``engines/`` directory written by
    ``scripts/deployment/build_trt_pipeline.py``. The TRT runtime glue lives in
    ``scripts/deployment`` (not an importable package), so this needs a source checkout.
    Call it before :func:`apply_inference_optimizations`.

    The engines run without a host sync between them, and with ``cuda_graph`` each one is
    replayed from a captured CUDA graph. Both only remove host overhead; the actions are
    unchanged. Graphs are captured during the first inference, which is therefore slower.
    """
    deploy_dir = Path(__file__).resolve().parents[2] / "scripts" / "deployment"
    if not (deploy_dir / "trt_model_forward.py").is_file():
        raise FileNotFoundError(
            f"TensorRT runtime glue not found at {deploy_dir}; loading TRT engines "
            "requires running from a source checkout of this repository."
        )
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    from _trt_contract import resolve_batch_size
    from trt_model_forward import setup_tensorrt_engines

    from gr00t.deployment.modes import InferenceMode

    # Engines bake a static batch size; fail fast on a mismatch instead of a cryptic
    # "Invalid input shape" inside the first inference.
    resolve_batch_size(engine_dir, batch_size, source="load_tensorrt_engines")
    setup_tensorrt_engines(
        policy,
        engine_dir,
        mode=InferenceMode.n17_full_pipeline,
        use_vit_engine=use_vit_engine,
        sync=False,
        cuda_graph=cuda_graph,
    )
    logging.info(
        "TensorRT full-pipeline engines loaded from %s%s",
        engine_dir,
        "" if use_vit_engine else " (ViT kept in PyTorch)",
    )
