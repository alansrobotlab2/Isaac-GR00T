# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline smoothness of chunked execution, with and without real-time chunking (RTC).

Replays dataset episodes through a policy the way a robot client executes chunks: an
observation every ``--steps-per-chunk`` control steps, a new chunk that arrives
``--latency`` steps later, and in between the previous chunk keeps running. Each chunk
therefore commands ``chunk[latency : latency + steps_per_chunk]``. With ``--rtc`` the
request carries the previous chunk's committed actions for the new chunk's first
``--rtc-overlap`` steps (``latency`` of them frozen), which the model inpaints from.

Metrics on the executed trajectory (physical units, all action dimensions):
  seam jump   mean |a(t) - a(t-1)| at the steps where a new chunk takes over
  step size   the same away from seams; seam/step > 1 means visible chunk boundaries
  jerk        RMS of the second difference of the executed actions
  MAE         mean |executed - ground-truth action| at the same steps

The robot's own state follows the dataset (open loop), so this isolates the policy's
chunk-to-chunk consistency rather than closed-loop behaviour.

Example:
    python scripts/deployment/rtc_smoothness_eval.py \\
        --model-path checkpoints/alfie/checkpoint-20000 --dataset-path <lerobot> \\
        --embodiment-tag new_embodiment --trt-engine-path <engines> \\
        --denoising-steps 4 --modes off rtc --traj-ids 0 1 2
"""

from dataclasses import dataclass, field
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import tyro


_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.utils import parse_observation_gr00t  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


@dataclass
class Config:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    trt_engine_path: str | None = None
    """Full-pipeline engine directory; None runs PyTorch."""
    denoising_steps: int = 4
    traj_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_steps: int = 400
    """Control steps per episode at most."""
    steps_per_chunk: int = 4
    """Executed actions per chunk (the client's n_action_steps)."""
    latency: int = 2
    """Control steps between the observation and the new chunk taking over."""
    rtc_overlap: int = 6
    rtc_ramp_rate: float = 3.0
    modes: list[str] = field(default_factory=lambda: ["off", "rtc"])
    """'off' and/or 'rtc'."""
    seed: int = 0
    output_json: str | None = None


def _observation(policy, traj, step):
    configs = policy.get_modality_config()
    data = extract_step_data(
        traj, step, modality_configs=configs, embodiment_tag=policy.embodiment_tag
    )
    obs = {f"state.{k}": v for k, v in data.states.items()}
    obs.update({f"video.{k}": np.array(v) for k, v in data.images.items()})
    for key in configs["language"].modality_keys:
        obs[key] = data.text
    return parse_observation_gr00t(obs, configs)


def _flat(chunk, keys):
    return np.concatenate([np.asarray(chunk[k]).reshape(chunk[k].shape[-2], -1) for k in keys], -1)


def run_episode(policy, traj, cfg, rtc: bool):
    configs = policy.get_modality_config()
    keys = configs["action"].modality_keys
    horizon = len(configs["action"].delta_indices)
    history = max(0, -min(configs["video"].delta_indices))
    n, lat = cfg.steps_per_chunk, cfg.latency
    if lat + n > horizon:
        raise ValueError(f"latency + steps_per_chunk ({lat + n}) exceeds the horizon ({horizon})")
    last = min(len(traj), cfg.max_steps) - horizon
    executed, times, seams, latencies = [], [], [], []
    prev_chunk = None
    for k, t in enumerate(range(history, last, n)):
        options = None
        if rtc and prev_chunk is not None:
            # Committed actions for steps t, t+1, ...: the previous chunk (observed at t - n)
            # from its index n on.
            overlap = min(cfg.rtc_overlap, horizon - n)
            options = {
                "rtc_prev_actions": {key: prev_chunk[key][0, n : n + overlap] for key in keys},
                "rtc_frozen_steps": lat,
                "rtc_overlap_steps": overlap,
                "rtc_ramp_rate": cfg.rtc_ramp_rate,
            }
        torch.manual_seed(cfg.seed * 100003 + k)  # same noise per request across modes
        start = time.perf_counter()
        chunk, _ = policy.get_action(_observation(policy, traj, t), options)
        latencies.append(time.perf_counter() - start)
        flat = _flat(chunk, keys)  # (horizon, D)
        for i in range(lat, lat + n):
            if k > 0 and i == lat:
                seams.append(len(executed))
            executed.append(flat[i])
            times.append(t + i)
        prev_chunk = chunk
    executed = np.stack(executed)
    gt = np.concatenate(
        [np.stack(traj[f"action.{key}"].to_numpy())[times].reshape(len(times), -1) for key in keys],
        -1,
    )
    return executed, gt, seams, latencies[1:]


def metrics(executed, gt, seams):
    deltas = np.abs(np.diff(executed, axis=0)).mean(axis=1)  # per transition t-1 -> t
    seam_idx = np.array([s - 1 for s in seams if s > 0])
    mask = np.ones(len(deltas), dtype=bool)
    mask[seam_idx] = False
    second = np.diff(executed, n=2, axis=0)
    return {
        "seam_jump": float(deltas[seam_idx].mean()),
        "step_size": float(deltas[mask].mean()),
        "seam_ratio": float(deltas[seam_idx].mean() / deltas[mask].mean()),
        "jerk_rms": float(np.sqrt((second**2).mean())),
        "mae": float(np.abs(executed - gt).mean()),
    }


def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)
    tag = EmbodimentTag.resolve(cfg.embodiment_tag)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device="cuda")
    if cfg.trt_engine_path:
        from gr00t.policy.inference_optimizations import load_tensorrt_engines

        load_tensorrt_engines(policy, cfg.trt_engine_path)
    policy.model.action_head.num_inference_timesteps = cfg.denoising_steps
    dataset = LeRobotEpisodeLoader(cfg.dataset_path, modality_configs=policy.get_modality_config())

    results = {}
    for mode in cfg.modes:
        per_episode, all_lat = [], []
        for traj_id in cfg.traj_ids:
            executed, gt, seams, lat = run_episode(policy, dataset[traj_id], cfg, mode == "rtc")
            per_episode.append(metrics(executed, gt, seams))
            all_lat += lat
        summary = {k: float(np.mean([m[k] for m in per_episode])) for k in per_episode[0]}
        summary["latency_ms"] = 1000 * float(np.mean(all_lat))
        results[mode] = summary
        print(
            f"[{mode:3s}] seam jump {summary['seam_jump']:.5f}  step {summary['step_size']:.5f}  "
            f"ratio {summary['seam_ratio']:.2f}  jerk {summary['jerk_rms']:.5f}  "
            f"MAE {summary['mae']:.5f}  ({summary['latency_ms']:.0f} ms/request)"
        )
    if cfg.output_json:
        with open(cfg.output_json, "w") as f:
            json.dump({"config": vars(cfg), "results": results}, f, indent=2)


if __name__ == "__main__":
    main(tyro.cli(Config))
