"""Test TRT vs PyTorch DiT with real backbone features (not random).

Uses a hook on the action_head to capture real DiT inputs.
"""
import torch
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

sys.path.insert(0, "/workspace/gr00t/scripts/deployment")
from standalone_inference_script import TensorRTDiTWrapper, prepare_model_inputs
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from copy import deepcopy


# Global to capture inputs
captured = {}


def hook_get_action_with_features(action_head):
    """Capture the inputs to get_action_with_features."""
    orig = action_head.get_action_with_features

    def hooked(backbone_features, state_features, embodiment_id, backbone_output):
        captured["vl_embeds"] = backbone_features.clone()
        captured["state_features"] = state_features.clone()
        captured["embodiment_id"] = embodiment_id.clone()
        captured["backbone_output"] = backbone_output
        return orig(
            backbone_features=backbone_features,
            state_features=state_features,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
        )

    action_head.get_action_with_features = hooked


# Load full PyTorch model
policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path="/workspace/gr00t/alfie-gr00t/checkpoint-10000",
    device="cuda",
    use_fp16=True,
    attn_implementation="sdpa",
)
modality = policy.get_modality_config()
model_ah = policy.model.action_head.action_horizon
decode_ah = len(modality["action"].delta_indices)
if model_ah != decode_ah:
    print(f"Fixing action_horizon: {model_ah} -> {decode_ah}")
    policy.model.action_head.config.action_horizon = decode_ah
    policy.model.action_head.action_horizon = decode_ah

# Hook to capture
hook_get_action_with_features(policy.model.action_head)

# Load data
dataset = LeRobotEpisodeLoader(
    dataset_path="/workspace/gr00t/alfiebot.CanDoChallenge",
    modality_configs=modality,
    video_backend="torchcodec",
)
traj = dataset[0]
modality_configs_no_action = deepcopy(dataset.modality_configs)
modality_configs_no_action.pop("action")

collated, states = prepare_model_inputs(
    policy, traj, 0, modality_configs_no_action,
    EmbodimentTag.NEW_EMBODIMENT, dataset,
)

# Run full pipeline to capture DiT inputs
print("\n=== Running full PyTorch pipeline ===")
torch.manual_seed(42)
policy.run_inference(collated, states)

# Now we have the real DiT inputs
vl_embeds = captured["vl_embeds"]
state_features = captured["state_features"]
embodiment_id = captured["embodiment_id"]
bo = captured["backbone_output"]

ah = policy.model.action_head

print(f"\nCaptured inputs:")
print(f"  vl_embeds: shape={vl_embeds.shape}, dtype={vl_embeds.dtype}")
print(f"    range: [{vl_embeds.min():.4f}, {vl_embeds.max():.4f}]")
print(f"  state_features: shape={state_features.shape}, dtype={state_features.dtype}")
print(f"    range: [{state_features.min():.4f}, {state_features.max():.4f}]")

with torch.inference_mode():
    # Setup step 0 DiT inputs exactly like the denoising loop
    torch.manual_seed(42)
    actions = torch.randn(
        size=(1, ah.config.action_horizon, ah.action_dim),
        dtype=torch.float32,
        device="cuda",
    )
    t_disc = int(0 / float(ah.num_inference_timesteps) * ah.num_timestep_buckets)
    timestep = torch.full(size=(1,), fill_value=t_disc, device="cuda")

    action_features = ah.action_encoder(
        actions.to(vl_embeds.dtype), timestep, embodiment_id
    )
    if ah.config.add_pos_embed:
        pos_ids = torch.arange(ah.action_horizon, dtype=torch.long, device="cuda")
        pos_embs = ah.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    sa_embs = torch.cat((state_features, action_features), dim=1)

    print(f"\n  sa_embs: shape={sa_embs.shape}, dtype={sa_embs.dtype}")
    print(f"    range: [{sa_embs.min():.4f}, {sa_embs.max():.4f}], norm: {sa_embs.float().norm():.4f}")

    # Run PyTorch DiT
    pt_out = ah.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embeds,
        timestep=timestep,
        image_mask=bo.image_mask,
        backbone_attention_mask=bo.backbone_attention_mask,
    )
    print(f"\nPyTorch DiT output:")
    print(f"  shape={pt_out.shape}, dtype={pt_out.dtype}")
    print(f"  range: [{pt_out.min():.6f}, {pt_out.max():.6f}]")

    # Run TRT DiT
    trt_dit = TensorRTDiTWrapper("/workspace/gr00t/groot_n1d6_onnx/dit_fp32.trt")
    trt_out = trt_dit(
        sa_embs=sa_embs,
        vl_embs=vl_embeds,
        timestep=timestep,
        image_mask=bo.image_mask,
        backbone_attention_mask=bo.backbone_attention_mask,
    )
    print(f"TRT DiT output:")
    print(f"  shape={trt_out.shape}, dtype={trt_out.dtype}")
    print(f"  range: [{trt_out.min():.6f}, {trt_out.max():.6f}]")

    # Compare
    mse = ((pt_out.float() - trt_out.float()) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        pt_out.float().flatten(), trt_out.float().flatten(), dim=0
    ).item()
    max_err = abs(pt_out.float() - trt_out.float()).max().item()
    print(f"\nDiT comparison with REAL features:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Cosine similarity: {cos:.8f}")
    print(f"  Max abs error: {max_err:.6f}")

    # Check masks
    im = bo.image_mask
    bam = bo.backbone_attention_mask
    if im is not None:
        print(f"\n  image_mask: dtype={im.dtype}, frac_true={im.float().mean():.4f}")
    if bam is not None:
        print(f"  backbone_attention_mask: dtype={bam.dtype}, frac_true={bam.float().mean():.4f}")
