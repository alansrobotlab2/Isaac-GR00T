"""Debug: instrument denoising loop to compare TRT vs PyTorch per-step.

Monkey-patches get_action_with_features to log per-step model outputs.
"""
import torch
import os
import logging
import sys
import numpy as np
from functools import wraps

logging.basicConfig(level=logging.INFO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

sys.path.insert(0, "/workspace/gr00t/scripts/deployment")
from standalone_inference_script import TensorRTDiTWrapper, replace_dit_with_tensorrt
from standalone_inference_script import prepare_model_inputs
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from copy import deepcopy


def fix_action_horizon(policy):
    modality = policy.get_modality_config()
    model_ah = policy.model.action_head.action_horizon
    decode_ah = len(modality["action"].delta_indices)
    if model_ah != decode_ah:
        print(f"Fixing action_horizon: {model_ah} -> {decode_ah}")
        policy.model.action_head.config.action_horizon = decode_ah
        policy.model.action_head.action_horizon = decode_ah
    return modality


# Capture per-step outputs
step_outputs = {}

def instrument_denoising(action_head, label):
    """Monkey-patch to capture per-step DiT outputs."""
    orig_fn = action_head.get_action_with_features.__func__

    @wraps(orig_fn)
    def patched(self, backbone_features, state_features, embodiment_id, backbone_output):
        from transformers import BatchFeature

        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        model_dtype = vl_embeds.dtype

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        timestep_tensors = []
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timestep_tensors.append(
                torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            )

        pos_embs = None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(self.action_horizon, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)

        step_outputs[label] = []

        for t in range(self.num_inference_timesteps):
            timesteps_tensor = timestep_tensors[t]
            action_features = self.action_encoder(
                actions.to(model_dtype), timesteps_tensor, embodiment_id
            )
            if self.config.add_pos_embed:
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            # Capture raw model output BEFORE dtype cast
            raw_output = model_output.float().clone()

            if model_output.dtype != self.dtype:
                model_output = model_output.to(self.dtype)

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            step_outputs[label].append({
                "model_output": raw_output.cpu(),
                "model_output_dtype": str(model_output.dtype),
                "pred_velocity": pred_velocity.float().cpu().clone(),
                "actions_before": actions.cpu().clone(),
                "sa_embs_dtype": str(sa_embs.dtype),
                "sa_embs_norm": sa_embs.float().norm().item(),
            })

            actions = actions + dt * pred_velocity.float()

        return BatchFeature(
            data={
                "action_pred": actions.to(model_dtype),
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    import types
    action_head.get_action_with_features = types.MethodType(patched, action_head)


# Load TRT model
print("=== Loading TRT model ===")
trt_dit = TensorRTDiTWrapper("/workspace/gr00t/groot_n1d6_onnx/dit_fp16.trt")

policy_trt = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path="/workspace/gr00t/alfie-gr00t/checkpoint-10000",
    device="cuda",
    skip_dit=True,
    use_fp16=True,
    attn_implementation="sdpa",
)
replace_dit_with_tensorrt(
    policy_trt, "/workspace/gr00t/groot_n1d6_onnx/dit_fp16.trt", preloaded_trt=trt_dit
)
modality = fix_action_horizon(policy_trt)
instrument_denoising(policy_trt.model.action_head, "trt")

# Load PyTorch model
print("\n=== Loading PyTorch model ===")
policy_pt = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path="/workspace/gr00t/alfie-gr00t/checkpoint-10000",
    device="cuda",
    use_fp16=True,
    attn_implementation="sdpa",
)
fix_action_horizon(policy_pt)
instrument_denoising(policy_pt.model.action_head, "pytorch")

# Load data
dataset = LeRobotEpisodeLoader(
    dataset_path="/workspace/gr00t/alfiebot.CanDoChallenge",
    modality_configs=modality,
    video_backend="torchcodec",
)
traj = dataset[0]

# Prepare inputs using the standalone_inference_script helper
modality_configs_no_action = deepcopy(dataset.modality_configs)
modality_configs_no_action.pop("action")

collated_trt, states_trt = prepare_model_inputs(
    policy_trt, traj, 0, modality_configs_no_action,
    EmbodimentTag.NEW_EMBODIMENT, dataset
)
collated_pt, states_pt = prepare_model_inputs(
    policy_pt, traj, 0, modality_configs_no_action,
    EmbodimentTag.NEW_EMBODIMENT, dataset
)

# Run both with same seed
print("\n=== Running PyTorch inference ===")
torch.manual_seed(42)
pt_result = policy_pt.run_inference(collated_pt, states_pt)

print("\n=== Running TRT inference ===")
torch.manual_seed(42)
trt_result = policy_trt.run_inference(collated_trt, states_trt)

# Compare per-step
print(f"\n{'='*80}")
print(f"PER-DENOISING-STEP COMPARISON")
print(f"{'='*80}")

for t in range(len(step_outputs["pytorch"])):
    pt_step = step_outputs["pytorch"][t]
    trt_step = step_outputs["trt"][t]

    mo_pt = pt_step["model_output"]
    mo_trt = trt_step["model_output"]
    mse_mo = ((mo_pt - mo_trt) ** 2).mean().item()
    cos_mo = torch.nn.functional.cosine_similarity(
        mo_pt.flatten(), mo_trt.flatten(), dim=0
    ).item()

    pv_pt = pt_step["pred_velocity"]
    pv_trt = trt_step["pred_velocity"]
    mse_pv = ((pv_pt - pv_trt) ** 2).mean().item()

    act_pt = pt_step["actions_before"]
    act_trt = trt_step["actions_before"]
    mse_act = ((act_pt - act_trt) ** 2).mean().item()

    print(f"\nStep {t}:")
    print(f"  sa_embs dtype:     PT={pt_step['sa_embs_dtype']}, TRT={trt_step['sa_embs_dtype']}")
    print(f"  sa_embs norm:      PT={pt_step['sa_embs_norm']:.4f}, TRT={trt_step['sa_embs_norm']:.4f}")
    print(f"  model_output dtype: PT={pt_step['model_output_dtype']}, TRT={trt_step['model_output_dtype']}")
    print(f"  model_output MSE:  {mse_mo:.8f}")
    print(f"  model_output cos:  {cos_mo:.8f}")
    print(f"  pred_velocity MSE: {mse_pv:.8f}")
    print(f"  actions_in MSE:    {mse_act:.8f}")

# Compare final action predictions (unnormalized)
print(f"\n{'='*80}")
print(f"FINAL ACTION COMPARISON (unnormalized)")
print(f"{'='*80}")
for key in pt_result:
    pt_val = pt_result[key]
    trt_val = trt_result[key]
    mse = ((pt_val - trt_val) ** 2).mean()
    mae = np.abs(pt_val - trt_val).mean()
    print(f"  {key}: MSE={mse:.6f}, MAE={mae:.6f}")
