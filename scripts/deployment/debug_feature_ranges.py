"""Compare backbone feature ranges: BF16 flash_attn vs FP16 SDPA."""
import torch
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

sys.path.insert(0, "/workspace/gr00t/scripts/deployment")
from standalone_inference_script import prepare_model_inputs
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from copy import deepcopy

captured = {}


def hook(action_head, label):
    orig = action_head.get_action_with_features

    def hooked(backbone_features, state_features, embodiment_id, backbone_output):
        captured[label] = {
            "vl_range": [backbone_features.min().item(), backbone_features.max().item()],
            "vl_norm": backbone_features.float().norm().item(),
            "vl_dtype": str(backbone_features.dtype),
            "sa_range_state": [state_features.min().item(), state_features.max().item()],
        }
        return orig(
            backbone_features=backbone_features,
            state_features=state_features,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
        )

    action_head.get_action_with_features = hooked


def fix_ah(policy):
    m = policy.get_modality_config()
    ah = policy.model.action_head.action_horizon
    dah = len(m["action"].delta_indices)
    if ah != dah:
        policy.model.action_head.config.action_horizon = dah
        policy.model.action_head.action_horizon = dah
    return m


# BF16 + flash_attention_2 (default)
print("=== Loading BF16 + flash_attention_2 model ===")
p_bf16 = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path="/workspace/gr00t/alfie-gr00t/checkpoint-10000",
    device="cuda",
)
m = fix_ah(p_bf16)
hook(p_bf16.model.action_head, "bf16")

dataset = LeRobotEpisodeLoader(
    dataset_path="/workspace/gr00t/alfiebot.CanDoChallenge",
    modality_configs=m,
    video_backend="torchcodec",
)
traj = dataset[0]
mc = deepcopy(dataset.modality_configs)
mc.pop("action")

c, s = prepare_model_inputs(p_bf16, traj, 0, mc, EmbodimentTag.NEW_EMBODIMENT, dataset)
torch.manual_seed(42)
p_bf16.run_inference(c, s)

# Clean up
del p_bf16
torch.cuda.empty_cache()
import gc
gc.collect()

# FP16 + SDPA
print("\n=== Loading FP16 + SDPA model ===")
p_fp16 = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path="/workspace/gr00t/alfie-gr00t/checkpoint-10000",
    device="cuda",
    use_fp16=True,
    attn_implementation="sdpa",
)
fix_ah(p_fp16)
hook(p_fp16.model.action_head, "fp16")

c2, s2 = prepare_model_inputs(p_fp16, traj, 0, mc, EmbodimentTag.NEW_EMBODIMENT, dataset)
torch.manual_seed(42)
p_fp16.run_inference(c2, s2)

print(f"\n{'='*60}")
print("BACKBONE FEATURE COMPARISON")
print(f"{'='*60}")

for label, data in captured.items():
    print(f"\n{label}:")
    print(f"  vl_embeds range: {data['vl_range']}")
    print(f"  vl_embeds norm:  {data['vl_norm']:.4f}")
    print(f"  vl_embeds dtype: {data['vl_dtype']}")
    print(f"  state range:     {data['sa_range_state']}")
