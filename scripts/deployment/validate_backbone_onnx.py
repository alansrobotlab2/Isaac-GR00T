#!/usr/bin/env python3
"""
Validate backbone ONNX model accuracy by comparing PyTorch vs ONNX Runtime outputs.
This isolates whether accuracy loss comes from the ONNX trace or from TRT compilation.
"""

import argparse
import logging
import os
from copy import deepcopy

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def swap_flash_to_eager(model):
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            if module.config._attn_implementation == "flash_attention_2":
                module.config._attn_implementation = "eager"
                count += 1
        if hasattr(module, "_attn_implementation"):
            if module._attn_implementation == "flash_attention_2":
                module._attn_implementation = "eager"
                count += 1
    logger.info(f"Swapped {count} flash_attention_2 → eager attention implementations")


def main():
    parser = argparse.ArgumentParser(description="Validate backbone ONNX accuracy")
    parser.add_argument("--model_path", type=str, default="alfie-gr00t/checkpoint-10000")
    parser.add_argument("--dataset_path", type=str, default="alfiebot.CanDoChallenge")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--onnx_path", type=str, default="./groot_n1d6_onnx/backbone_model.onnx")
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    args = parser.parse_args()

    # Load policy
    logger.info("Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
    )

    # Load dataset and prepare input
    modality_config = policy.get_modality_config()
    modality_configs = deepcopy(modality_config)
    modality_configs.pop("action", None)

    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
    )

    traj = dataset[0]
    data_point = extract_step_data(traj, 0, modality_configs, policy.embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for key in modality_configs["language"].modality_keys:
        obs[key] = data_point.text

    # Parse observation
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]

    # Prepare model inputs
    unbatched_obs = []
    batch_size = new_obs["video"][list(new_obs["video"].keys())[0]].shape[0]
    for i in range(batch_size):
        unbatched_value = {
            "video": {k: v[i] for k, v in new_obs["video"].items()},
            "state": {k: v[i] for k, v in new_obs["state"].items()},
            "language": {k: v[i] for k, v in new_obs["language"].items()},
        }
        unbatched_obs.append(unbatched_value)

    processed_inputs = []
    for o in unbatched_obs:
        vla_step_data = VLAStepData(
            images=o["video"],
            states=o["state"],
            actions={},
            text=o["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = collated_inputs["inputs"]
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    # ---- 1. Reference: PyTorch BF16 backbone (with flash attention) ----
    logger.info("\n=== Reference: PyTorch BF16 backbone (flash attention) ===")
    with torch.inference_mode():
        bb_inputs, _ = policy.model.prepare_input(collated_inputs)
        ref_output = policy.model.backbone(bb_inputs)

    ref_features = ref_output.backbone_features.detach().cpu().float()
    logger.info(f"Reference output shape: {ref_features.shape}")
    logger.info(f"Reference range: [{ref_features.min():.2f}, {ref_features.max():.2f}]")

    # ---- 2. PyTorch eager+FP32 backbone (same as ONNX export path) ----
    logger.info("\n=== Test A: PyTorch eager+FP32 backbone ===")
    backbone_eager = deepcopy(policy.model.backbone)
    swap_flash_to_eager(backbone_eager)
    backbone_eager = backbone_eager.float().cuda().eval()

    with torch.inference_mode():
        bb_inputs_copy, _ = policy.model.prepare_input(collated_inputs)
        # Cast to FP32 to match ONNX export
        pv = bb_inputs_copy["pixel_values"]
        if isinstance(pv, list):
            pv_fp32 = [t.float() for t in pv]
        else:
            pv_fp32 = pv.float()
        eager_input = BatchFeature(data={
            "pixel_values": pv_fp32,
            "input_ids": bb_inputs_copy["input_ids"],
            "attention_mask": bb_inputs_copy["attention_mask"],
        })
        eager_output = backbone_eager(eager_input)

    eager_features = eager_output.backbone_features.detach().cpu().float()
    logger.info(f"Eager output shape: {eager_features.shape}")
    logger.info(f"Eager range: [{eager_features.min():.2f}, {eager_features.max():.2f}]")

    diff_eager = ref_features - eager_features
    logger.info(f"MSE(ref, eager): {(diff_eager**2).mean():.6f}")
    logger.info(f"MAE(ref, eager): {diff_eager.abs().mean():.6f}")
    cos_eager = torch.nn.functional.cosine_similarity(
        ref_features.reshape(1, -1), eager_features.reshape(1, -1)
    ).item()
    logger.info(f"Cosine sim(ref, eager): {cos_eager:.6f}")

    del backbone_eager
    torch.cuda.empty_cache()

    # ---- 3. ONNX Runtime (CUDA) ----
    logger.info(f"\n=== Test B: ONNX Runtime (CUDA) from {args.onnx_path} ===")
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(
        args.onnx_path,
        sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Prepare ONNX inputs (same as what we feed during export)
    with torch.inference_mode():
        bb_inputs_onnx, _ = policy.model.prepare_input(collated_inputs)
        pv_onnx = bb_inputs_onnx["pixel_values"]
        if isinstance(pv_onnx, list):
            pixel_values_np = torch.cat(pv_onnx, dim=0).float().cpu().numpy()
        else:
            pixel_values_np = pv_onnx.float().cpu().numpy()

    input_ids_np = bb_inputs_onnx["input_ids"].cpu().numpy()
    attention_mask_np = bb_inputs_onnx["attention_mask"].cpu().numpy()

    logger.info(f"ONNX inputs: pixel_values={pixel_values_np.shape} ({pixel_values_np.dtype}), "
                f"input_ids={input_ids_np.shape}, attention_mask={attention_mask_np.shape}")

    ort_inputs = {
        "pixel_values": pixel_values_np,
        "input_ids": input_ids_np,
        "attention_mask": attention_mask_np,
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_features = torch.from_numpy(ort_outputs[0]).float()

    logger.info(f"ORT output shape: {ort_features.shape}")
    logger.info(f"ORT range: [{ort_features.min():.2f}, {ort_features.max():.2f}]")

    diff_ort = ref_features - ort_features
    logger.info(f"MSE(ref, ORT): {(diff_ort**2).mean():.6f}")
    logger.info(f"MAE(ref, ORT): {diff_ort.abs().mean():.6f}")
    cos_ort = torch.nn.functional.cosine_similarity(
        ref_features.reshape(1, -1), ort_features.reshape(1, -1)
    ).item()
    logger.info(f"Cosine sim(ref, ORT): {cos_ort:.6f}")

    # Also compare eager vs ORT (to see how much the ONNX trace adds beyond eager)
    diff_eager_ort = eager_features - ort_features
    logger.info(f"\nMSE(eager, ORT): {(diff_eager_ort**2).mean():.6f}")
    logger.info(f"MAE(eager, ORT): {diff_eager_ort.abs().mean():.6f}")
    cos_eager_ort = torch.nn.functional.cosine_similarity(
        eager_features.reshape(1, -1), ort_features.reshape(1, -1)
    ).item()
    logger.info(f"Cosine sim(eager, ORT): {cos_eager_ort:.6f}")

    # ---- Summary ----
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Comparison':<30} {'MSE':>12} {'MAE':>12} {'Cos Sim':>12}")
    logger.info("-" * 66)
    logger.info(f"{'ref vs eager (flash→eager)':<30} {(diff_eager**2).mean():>12.6f} {diff_eager.abs().mean():>12.6f} {cos_eager:>12.6f}")
    logger.info(f"{'ref vs ORT (ONNX trace)':<30} {(diff_ort**2).mean():>12.6f} {diff_ort.abs().mean():>12.6f} {cos_ort:>12.6f}")
    logger.info(f"{'eager vs ORT (trace quality)':<30} {(diff_eager_ort**2).mean():>12.6f} {diff_eager_ort.abs().mean():>12.6f} {cos_eager_ort:>12.6f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
