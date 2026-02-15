from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import time
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
):
    # Ensure steps doesn't exceed trajectory length
    t_load_start = time.perf_counter()
    traj = loader[traj_id]
    t_load_end = time.perf_counter()
    episode_load_time = t_load_end - t_load_start

    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []
    data_prep_times = []
    inference_times = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, action_horizon):
        t_prep_start = time.perf_counter()
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        t_prep_end = time.perf_counter()
        data_prep_times.append(t_prep_end - t_prep_start)

        logging.info(f"inferencing at step: {step_count}")
        t_inf_start = time.perf_counter()
        _action_chunk, _ = policy.get_action(parsed_obs)
        t_inf_end = time.perf_counter()
        inference_times.append(t_inf_end - t_inf_start)

        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    timing = {
        "episode_load_time": episode_load_time,
        "data_prep_times": data_prep_times,
        "inference_times": inference_times,
    }
    return mse, mae, timing


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    inference_mode: str = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str | None = None
    """Path to the TensorRT engine file (.trt) for the DiT model. Required when inference_mode is 'tensorrt'."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    t_model_start = time.perf_counter()
    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if args.inference_mode == "tensorrt":
            if args.trt_engine_path is None:
                raise ValueError("--trt-engine-path is required when --inference-mode is 'tensorrt'")
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "deployment"))
            from standalone_inference_script import replace_dit_with_tensorrt

            replace_dit_with_tensorrt(policy, args.trt_engine_path)
    else:
        policy = PolicyClient(host=args.host, port=args.port)
    t_model_end = time.perf_counter()
    model_load_time = t_model_end - t_model_start

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    t_dataset_start = time.perf_counter()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    t_dataset_end = time.perf_counter()
    dataset_load_time = t_dataset_end - t_dataset_start

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []
    all_episode_load_times = []
    all_data_prep_times = []
    all_inference_times = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae, timing = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)
        all_episode_load_times.append(timing["episode_load_time"])
        all_data_prep_times.extend(timing["data_prep_times"])
        all_inference_times.extend(timing["inference_times"])

    # Print evaluation summary
    logging.info("=" * 80)
    logging.info("=== EVALUATION SUMMARY ===")
    logging.info("=" * 80)
    logging.info("")
    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info("Metrics:")
        logging.info(f"  Average MSE across all trajs: {avg_mse:.6f}")
        logging.info(f"  Average MAE across all trajs: {avg_mae:.6f}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("")

    # Print detailed timing summary
    num_trajs = len(all_episode_load_times)
    num_steps = len(all_inference_times)
    total_episode_load = sum(all_episode_load_times)
    total_data_prep = sum(all_data_prep_times)
    total_inference = sum(all_inference_times)
    inf_times = np.array(all_inference_times) if all_inference_times else np.array([0.0])

    logging.info("=" * 80)
    logging.info("=== DETAILED TIMING SUMMARY ===")
    logging.info("=" * 80)
    logging.info("")
    logging.info("Initialization:")
    logging.info(f"  Model loading time:          {model_load_time:.4f}s")
    logging.info(f"  Dataset loader creation:     {dataset_load_time:.4f}s")
    logging.info("")
    logging.info(f"Per-Trajectory Timings ({num_trajs} trajectories):")
    logging.info(
        f"  Total episode loading:       {total_episode_load:.4f}s"
        f"  (avg: {total_episode_load / max(num_trajs, 1):.4f}s)"
    )
    logging.info(
        f"  Total data preparation:      {total_data_prep:.4f}s"
        f"  (avg: {total_data_prep / max(num_steps, 1):.4f}s per step)"
    )
    logging.info(
        f"  Total inference:             {total_inference:.4f}s"
        f"  (avg: {total_inference / max(num_steps, 1):.4f}s per step)"
    )
    logging.info("")
    logging.info("Inference Statistics:")
    logging.info(f"  Total inference steps:       {num_steps}")
    logging.info(f"  Avg inference time per step: {np.mean(inf_times):.4f}s")
    logging.info(f"  Min inference time:          {np.min(inf_times):.4f}s")
    logging.info(f"  Max inference time:          {np.max(inf_times):.4f}s")
    logging.info(f"  P90 inference time:          {np.percentile(inf_times, 90):.4f}s")
    logging.info("")

    if args.save_plot_path:
        logging.info("=" * 80)
        logging.info(f"Plot saved to: {Path(args.save_plot_path).resolve()}")
        logging.info("=" * 80)
    logging.info("Done")



if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
