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

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.inference_optimizations import (
    CompileMode,
    apply_inference_optimizations,
    load_tensorrt_engines,
)
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PipelinedPolicyServer, PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Inference optimizations (Gr00t policy only; see INFERENCE_OPTIMIZATIONS_N1D7.md)
    denoising_steps: int | None = None
    """Number of denoising steps. None keeps the checkpoint's setting."""

    trt_engine_path: str | None = None
    """Directory of full-pipeline TensorRT engines built by
    scripts/deployment/build_trt_pipeline.py. None runs the model in PyTorch."""

    trt_pytorch_vit: bool = False
    """With --trt-engine-path: keep the ViT in PyTorch (LLM and action head still in TensorRT)."""

    trt_cuda_graph: bool = True
    """With TensorRT engines: replay each engine from a captured CUDA graph (less host
    overhead, identical actions). The first inference is slower while graphs are captured."""

    attn_implementation: Literal["flash_attention_2", "sdpa"] | None = None
    """Override the backbone attention implementation. None keeps the checkpoint's setting."""

    pipelined: bool = False
    """Accept several get_action requests at once and preprocess the next one while the
    model runs the current one (needs a client that keeps two in flight, e.g.
    gr00t.policy.server_client.PipelinedPolicyClient; plain clients still work)."""

    gpu_image_preprocessing: bool = True
    """Resize, normalize and patchify camera images on the GPU instead of the CPU (pixel
    values match to within one 8-bit step). Ignored when --device is not a CUDA device."""

    images_pretransformed: bool = False
    """Camera frames arrive already centre-cropped (crop_fraction, 0.95) and resized to the
    final model input size: short edge 256, both sides multiples of 32, e.g. 352x256 for
    4:3 or 448x256 for 16:9 cameras. Skips the crop and both resizes; other sizes raise."""

    compile_backbone: bool = False
    """torch.compile the backbone (needs --attn-implementation sdpa; skipped under TensorRT)."""

    compile_dit: bool = False
    """torch.compile the DiT (skipped under TensorRT)."""

    compile_action_head: bool = False
    """torch.compile the action encoder/decoder (skipped under TensorRT)."""

    compile_mode: CompileMode = "default"
    """torch.compile mode for --compile-backbone / --compile-dit."""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            attn_implementation=config.attn_implementation,
            gpu_image_preprocessing=config.gpu_image_preprocessing,
            images_pretransformed=config.images_pretransformed,
        )
        if config.images_pretransformed:
            print("  Images pre-transformed: crop and resize skipped")
        if config.denoising_steps is not None:
            # The action head reads num_inference_timesteps at sampling time.
            policy.model.action_head.num_inference_timesteps = config.denoising_steps
            print(f"  Denoising steps: {config.denoising_steps}")
        if config.trt_engine_path is not None:
            load_tensorrt_engines(
                policy,
                config.trt_engine_path,
                use_vit_engine=not config.trt_pytorch_vit,
                cuda_graph=config.trt_cuda_graph,
            )
            print(f"  TensorRT engines: {config.trt_engine_path}")
        # After the TRT setup, so components already replaced by engines are skipped.
        apply_inference_optimizations(
            policy,
            compile_backbone=config.compile_backbone,
            compile_dit=config.compile_dit,
            compile_action_head=config.compile_action_head,
            compile_mode=config.compile_mode,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server_class = PipelinedPolicyServer if config.pipelined else PolicyServer
    with server_class(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
