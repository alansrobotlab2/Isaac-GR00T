from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import gc
import logging
from pathlib import Path
import re
import time
from typing import Any, Literal
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
    action_dim_labels: list[str] | None = None,
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
        action_dim_labels: Optional list of per-dimension labels (e.g. ["left_arm[0]", "left_arm[1]", ...])
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

        if action_dim_labels and action_idx < len(action_dim_labels):
            ax.set_title(action_dim_labels[action_idx])
        else:
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


def _prepare_model_inputs(
    policy: Gr00tPolicy,
    traj: pd.DataFrame,
    step_count: int,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
    loader: LeRobotEpisodeLoader,
) -> tuple[dict, list]:
    """Full preprocessing pipeline: observation extraction + VLA processor + collation.

    CPU-only operations, safe to run in a background thread while GPU processes
    the previous step. Returns model-ready inputs for policy.run_inference().
    """
    data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in loader.modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text
    parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
    collated_inputs, states = policy.prepare_inputs(parsed_obs)
    return collated_inputs, states


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    skip_timing_steps=1,
    pipeline=None,
):
    timing_dict = {
        "episode_load_time": 0.0,
        "data_prep_times": [],
        "inference_times": [],
    }

    # Ensure steps doesn't exceed trajectory length
    episode_load_start = time.time()
    traj = loader[traj_id]
    timing_dict["episode_load_time"] = time.time() - episode_load_start

    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    step_counts = list(range(0, actual_steps, action_horizon))
    num_inference_steps = len(step_counts)
    use_pipeline = pipeline is not None
    logging.info(f"Running {num_inference_steps} inference steps (skipping first {skip_timing_steps} for timing)")
    logging.info(f"Async CPU prefetching: enabled | Pipeline parallelism: {use_pipeline}")

    # Async CPU preprocessing: prefetch next step's data while GPU runs inference
    executor = ThreadPoolExecutor(max_workers=1)
    future_inputs = executor.submit(
        _prepare_model_inputs, policy, traj, step_counts[0],
        modality_configs, embodiment_tag, loader,
    )

    for step_idx, step_count in enumerate(step_counts):
        logging.info(f"inferencing at step: {step_count}")

        # Wait for preprocessing (should already be done from prefetch)
        data_prep_start = time.time()
        collated_inputs, states = future_inputs.result()
        data_prep_time = time.time() - data_prep_start

        # Prefetch NEXT step's preprocessing while GPU runs inference
        if step_idx + 1 < len(step_counts):
            future_inputs = executor.submit(
                _prepare_model_inputs, policy, traj, step_counts[step_idx + 1],
                modality_configs, embodiment_tag, loader,
            )

        # Run inference
        inference_start = time.time()
        if use_pipeline:
            import torch
            if step_idx == 0:
                # First frame: run full pipeline sequentially
                _action_chunk = policy.run_inference(collated_inputs, states)
                # Start backbone for next frame async
                if step_idx + 1 < len(step_counts):
                    next_collated, next_states = future_inputs.result()
                    pipeline.start_backbone_async(next_collated)
                    pipeline._next_states = next_states
                    done_future = Future()
                    done_future.set_result((next_collated, next_states))
                    future_inputs = done_future
            else:
                # Subsequent frames: backbone already running from previous iteration
                with torch.inference_mode():
                    model_pred = pipeline.finish_frame()
                normalized_action = model_pred["action_pred"].float()
                batched_states = {}
                for k in policy.modality_configs["state"].modality_keys:
                    batched_states[k] = np.stack([s[k] for s in states], axis=0)
                unnormalized_action = policy.processor.decode_action(
                    normalized_action.cpu().numpy(), policy.embodiment_tag, batched_states
                )
                _action_chunk = {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
                # Start backbone for NEXT frame async
                if step_idx + 1 < len(step_counts):
                    next_collated, next_states = future_inputs.result()
                    pipeline.start_backbone_async(next_collated)
                    pipeline._next_states = next_states
                    done_future = Future()
                    done_future.set_result((next_collated, next_states))
                    future_inputs = done_future
        else:
            _action_chunk = policy.run_inference(collated_inputs, states)
        inference_time = time.time() - inference_start

        if step_idx >= skip_timing_steps:
            timing_dict["data_prep_times"].append(data_prep_time)
            timing_dict["inference_times"].append(inference_time)

        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    executor.shutdown(wait=True)

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

    # Build per-dimension labels from action keys, using joint names from info.json
    action_dim_labels = []
    all_joint_names = loader.feature_config.get("action", {}).get("names", None)
    for key in action_keys:
        modality_info = loader.modality_meta.get("action", {}).get(key, {})
        start_idx = modality_info.get("start", None)
        end_idx = modality_info.get("end", None)
        col = f"action.{key}"
        dim = np.atleast_1d(traj[col].iloc[0]).shape[0]
        for i in range(dim):
            joint_name = None
            if all_joint_names and start_idx is not None:
                abs_idx = start_idx + i
                if abs_idx < len(all_joint_names):
                    joint_name = all_joint_names[abs_idx]
            if joint_name:
                label = f"{key}[{i}] ({joint_name})" if dim > 1 else f"{key} ({joint_name})"
            else:
                label = f"{key}[{i}]" if dim > 1 else key
            action_dim_labels.append(label)

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
        action_dim_labels=action_dim_labels,
    )

    return mse, mae, timing_dict


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

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str = ""
    """Path to TensorRT DiT engine file (.trt). Used only when inference_mode='tensorrt'."""

    backbone_trt_engine_path: str = ""
    """Path to TensorRT engine file for the backbone. When set, replaces PyTorch backbone with TRT."""

    attn_implementation: str | None = None
    """Override backbone attention implementation. Options: 'flash_attention_2' (default), 'sdpa' (ONNX/TRT-compatible)."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    skip_timing_steps: int = 1
    """Number of initial inference steps to skip when calculating timing statistics (default: 1 to exclude warmup)."""

    compile_backbone: bool = False
    """Apply torch.compile to the backbone for kernel fusion."""

    compile_backbone_mode: str = "max-autotune"
    """torch.compile mode for backbone. Options: 'default', 'reduce-overhead', 'max-autotune'."""

    use_cuda_graphs: bool = False
    """Wrap backbone forward pass in CUDA graphs to eliminate kernel launch overhead."""

    pipeline_backbone_dit: bool = False
    """Overlap backbone(N+1) with DiT(N) on separate CUDA streams for higher throughput."""

    compile_action_head: bool = False
    """Apply torch.compile to action encoder/decoder for kernel fusion in denoising loop."""

    model_action_horizon: int | None = None
    """Override model's internal action_horizon at inference time. Smaller values (e.g. 4)
    reduce tensor sizes in the denoising loop for faster inference, but may affect quality
    if the model was trained on a different horizon."""

    cudnn_benchmark: bool = False
    """Enable cuDNN benchmark mode for auto-selecting fastest conv algorithms (fixed input shapes)."""

    seed: int = 42
    """Seed to use for reproducibility."""


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

    model_load_start = time.time()

    if local_model_path is not None:
        import torch

        if args.inference_mode == "tensorrt" and torch.cuda.is_available():
            import importlib.util
            _script_path = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment" / "standalone_inference_script.py")
            _spec = importlib.util.spec_from_file_location("standalone_inference_script", _script_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            TensorRTBackboneWrapper = _mod.TensorRTBackboneWrapper
            TensorRTDiTWrapper = _mod.TensorRTDiTWrapper
            replace_backbone_with_tensorrt = _mod.replace_backbone_with_tensorrt
            replace_dit_with_tensorrt = _mod.replace_dit_with_tensorrt
            wrap_backbone_with_cuda_graphs = _mod.wrap_backbone_with_cuda_graphs

            # TRT needs contiguous GPU memory - load ALL TRT engines FIRST while GPU is empty.
            # Critical on Jetson unified memory where CPU/GPU share 16GB.
            logging.info("TensorRT mode: Loading ALL TRT engines first while GPU is empty...")

            # Load backbone TRT engine first (largest: ~3GB)
            backbone_trt = None
            if args.backbone_trt_engine_path:
                logging.info(f"Loading backbone TRT engine: {args.backbone_trt_engine_path}")
                backbone_trt = TensorRTBackboneWrapper(args.backbone_trt_engine_path, device=0)
                gc.collect()
                torch.cuda.empty_cache()

            # Load DiT TRT engine
            assert args.trt_engine_path, "trt_engine_path is required when inference_mode='tensorrt'"
            logging.info(f"Loading DiT TensorRT engine: {args.trt_engine_path}")
            trt_dit = TensorRTDiTWrapper(args.trt_engine_path, device=0, use_fp16_output=False)
            gc.collect()
            torch.cuda.empty_cache()

            # Load PyTorch model WITHOUT DiT (skip_dit=True saves ~2GB)
            skip_backbone = bool(args.backbone_trt_engine_path)
            use_fp16 = args.attn_implementation == "sdpa"
            logging.info(f"Loading PyTorch model (skip_dit=True, skip_backbone={skip_backbone})...")
            policy = Gr00tPolicy(
                embodiment_tag=args.embodiment_tag,
                model_path=local_model_path,
                device="cuda",
                skip_dit=True,
                skip_backbone=skip_backbone,
                use_fp16=use_fp16,
                attn_implementation=args.attn_implementation,
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Wire up TRT engines to replace the (empty) PyTorch components
            replace_dit_with_tensorrt(policy, args.trt_engine_path, preloaded_trt=trt_dit)
            if backbone_trt is not None:
                replace_backbone_with_tensorrt(policy, args.backbone_trt_engine_path, preloaded_trt=backbone_trt)

            # torch.compile on the PyTorch backbone (only when backbone is NOT replaced by TRT)
            if args.compile_backbone and not args.backbone_trt_engine_path:
                logging.info(f"Compiling backbone with torch.compile(mode='{args.compile_backbone_mode}')...")
                policy.model.backbone.forward = torch.compile(
                    policy.model.backbone.forward, mode=args.compile_backbone_mode
                )
                logging.info("Backbone compiled (will warmup on first inference)")

            # CUDA graphs on backbone (only when backbone is NOT replaced by TRT)
            if args.use_cuda_graphs and not args.backbone_trt_engine_path:
                wrap_backbone_with_cuda_graphs(policy)

            gc.collect()
            torch.cuda.empty_cache()
            logging.info("TensorRT mode enabled")
        else:
            policy = Gr00tPolicy(
                embodiment_tag=args.embodiment_tag,
                model_path=local_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                attn_implementation=args.attn_implementation,
            )

            # Optional: torch.compile on backbone
            if args.compile_backbone:
                import torch
                logging.info(f"Compiling backbone with torch.compile(mode='{args.compile_backbone_mode}')...")
                policy.model.backbone.forward = torch.compile(
                    policy.model.backbone.forward, mode=args.compile_backbone_mode
                )
                logging.info("Backbone compiled (will warmup on first inference)")

            # Optional: CUDA graphs on backbone
            if args.use_cuda_graphs:
                import importlib.util
                _script_path = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment" / "standalone_inference_script.py")
                _spec = importlib.util.spec_from_file_location("standalone_inference_script", _script_path)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _mod.wrap_backbone_with_cuda_graphs(policy)
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    model_load_time = time.time() - model_load_start
    logging.info(f"Model loading time: {model_load_time:.4f}s")

    # cuDNN benchmark mode: auto-select fastest conv algorithms for fixed input shapes
    if args.cudnn_benchmark and local_model_path is not None:
        import torch
        torch.backends.cudnn.benchmark = True
        logging.info("cuDNN benchmark mode enabled (will auto-tune on first inference)")

    # Override denoising steps if needed
    if hasattr(policy, 'model') and hasattr(policy.model, 'action_head'):
        model_denoise = policy.model.action_head.num_inference_timesteps
        if args.denoising_steps != model_denoise:
            logging.info(f"Overriding num_inference_timesteps: {model_denoise} -> {args.denoising_steps}")
            policy.model.action_head.num_inference_timesteps = args.denoising_steps

    # Override model action horizon if specified
    if args.model_action_horizon is not None and hasattr(policy, 'model') and hasattr(policy.model, 'action_head'):
        original_ah = policy.model.action_head.config.action_horizon
        policy.model.action_head.config.action_horizon = args.model_action_horizon
        policy.model.action_head.action_horizon = args.model_action_horizon
        logging.info(f"Overriding model action_horizon: {original_ah} -> {args.model_action_horizon}")

    # torch.compile action encoder/decoder for kernel fusion in denoising loop
    if args.compile_action_head and hasattr(policy, 'model') and hasattr(policy.model, 'action_head'):
        import torch
        logging.info("Compiling action encoder/decoder with torch.compile(mode='default')...")
        policy.model.action_head.action_encoder = torch.compile(
            policy.model.action_head.action_encoder, mode="default"
        )
        policy.model.action_head.action_decoder = torch.compile(
            policy.model.action_head.action_decoder, mode="default"
        )
        logging.info("Action encoder/decoder compiled (will warmup on first inference)")

    # Create pipeline for backbone/DiT overlap if requested
    pipeline = None
    if args.pipeline_backbone_dit and hasattr(policy, 'model'):
        import torch
        import importlib.util
        _script_path = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment" / "standalone_inference_script.py")
        _spec = importlib.util.spec_from_file_location("standalone_inference_script", _script_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        pipeline = _mod.PipelinedInference(policy)
        logging.info("Pipeline parallelism enabled: backbone(N+1) overlaps with DiT(N)")

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset_load_start = time.time()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    dataset_load_time = time.time() - dataset_load_start

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []
    all_timings = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae, timing_dict = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
            skip_timing_steps=args.skip_timing_steps,
            pipeline=pipeline,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)
        all_timings.append(timing_dict)

    # === EVALUATION SUMMARY ===
    logging.info("\n" + "=" * 80)
    logging.info("=== EVALUATION SUMMARY ===")
    logging.info("=" * 80)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info("\nMetrics:")
        logging.info(f"  Average MSE across all trajs: {avg_mse:.6f}")
        logging.info(f"  Average MAE across all trajs: {avg_mae:.6f}")
    else:
        logging.info("No valid trajectories were evaluated.")

    # === DETAILED TIMING SUMMARY ===
    logging.info("\n" + "=" * 80)
    logging.info("=== DETAILED TIMING SUMMARY ===")
    logging.info("=" * 80)
    logging.info("\nInitialization:")
    logging.info(f"  Model loading time:          {model_load_time:.4f}s")
    logging.info(f"  Dataset loader creation:     {dataset_load_time:.4f}s")

    if all_timings:
        total_episode_load = sum(t["episode_load_time"] for t in all_timings)
        total_inference = sum(sum(t["inference_times"]) for t in all_timings)
        total_inference_steps = sum(len(t["inference_times"]) for t in all_timings)

        logging.info(f"\nPer-Trajectory Timings ({len(all_timings)} trajectories):")
        logging.info(
            f"  Total episode loading:       {total_episode_load:.4f}s  (avg: {total_episode_load / len(all_timings):.4f}s)"
        )
        if total_inference_steps > 0:
            logging.info(
                f"  Total inference:             {total_inference:.4f}s  (avg: {total_inference / total_inference_steps:.4f}s per step)"
            )

            logging.info("\nInference Statistics:")
            logging.info(f"  Total inference steps:       {total_inference_steps}")
            logging.info(
                f"  Avg inference time per step: {total_inference / total_inference_steps:.4f}s"
            )

            all_inf_times = [t for timing in all_timings for t in timing["inference_times"]]
            logging.info(f"  Min inference time:          {min(all_inf_times):.4f}s")
            logging.info(f"  Max inference time:          {max(all_inf_times):.4f}s")
            logging.info(f"  P90 inference time:          {np.percentile(all_inf_times, 90):.4f}s")

            all_prep_times = [t for timing in all_timings for t in timing.get("data_prep_times", [])]
            if all_prep_times:
                logging.info(f"\nData Prep (async prefetch wait time):")
                logging.info(f"  Avg data prep wait:          {np.mean(all_prep_times):.4f}s")
                logging.info(f"  Max data prep wait:          {max(all_prep_times):.4f}s")

        logging.info(f"\nOptimizations:")
        logging.info(f"  Async CPU prefetch:          enabled")
        logging.info(f"  Pipeline parallelism:        {'enabled' if pipeline is not None else 'disabled'}")
        logging.info(f"  Compile action head:         {args.compile_action_head}")
        logging.info(f"  cuDNN benchmark:             {args.cudnn_benchmark}")
        logging.info(f"  Denoising steps:             {args.denoising_steps}")
        if args.model_action_horizon is not None:
            logging.info(f"  Model action horizon:        {args.model_action_horizon} (overridden)")

    logging.info("=" * 80)
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
