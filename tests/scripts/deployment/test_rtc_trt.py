# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time chunking (RTC) through the TensorRT action-head paths.

The PyTorch action head inpaints a new chunk from the previous one when it gets an
``action`` input and RTC options. The TensorRT runtime replaces ``get_action``; these tests
check that its per-step loop and the fused graph reproduce the PyTorch result with RTC, and
that a fused engine exported before RTC support falls back instead of dropping the request.
Stand-in engines call the PyTorch modules, so this runs on CPU.
"""

from pathlib import Path
import sys

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead, apply_rtc_prefix
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "scripts" / "deployment"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

BATCH, HORIZON, ACTION_DIM, VL_LEN, STEPS = 1, 6, 5, 7, 3
RTC_OPTIONS = {
    "action_horizon": HORIZON,
    "rtc_overlap_steps": 4,
    "rtc_frozen_steps": 1,
    "rtc_ramp_rate": 3.0,
}


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("tensorrt")
    from trt import trt_model_forward

    return trt_model_forward


def _head():
    config = Gr00tN1d7Config(
        backbone_embedding_dim=16,
        hidden_size=16,
        input_embedding_dim=16,
        max_state_dim=5,
        max_action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        state_history_length=1,
        num_inference_timesteps=STEPS,
        max_num_embodiments=2,
        add_pos_embed=True,
        max_seq_len=8,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 16,
            "interleave_self_attention": True,
        },
    )
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(config).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0, 0.3)
    return head


def _inputs(rtc=True):
    torch.manual_seed(3)
    image_mask = torch.zeros(BATCH, VL_LEN, dtype=torch.bool)
    image_mask[:, :3] = True
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(BATCH, VL_LEN, 16),
            "backbone_attention_mask": torch.ones(BATCH, VL_LEN, dtype=torch.bool),
            "image_mask": image_mask,
        }
    )
    data = {
        "state": torch.randn(BATCH, 1, 5),
        "embodiment_id": torch.zeros(BATCH, dtype=torch.long),
    }
    if rtc:
        data["action"] = torch.randn(BATCH, HORIZON, ACTION_DIM).clamp(-1, 1)
    return backbone_output, BatchFeature(data=data)


class _Engine:
    """Stand-in TensorRT engine calling the PyTorch module it replaces."""

    def __init__(self, fn, inputs):
        self.fn = fn
        self.in_meta = [[name, None, torch.float32] for name in inputs]
        self.calls = 0

    def set_runtime_tensor_shape(self, name, shape):
        pass

    def dtype_of(self, name):
        return torch.float32

    def __call__(self, *args, **kwargs):
        self.calls += 1
        names = {item[0] for item in self.in_meta}
        return self.fn(*args, **{k: v for k, v in kwargs.items() if k in names})


def _install_step_engines(head):
    head.state_encoder_engine = _Engine(
        lambda state, ids: {"output": head.state_encoder(state, ids)}, ["state"]
    )
    head.action_encoder_engine = _Engine(
        lambda actions, t, ids: {"output": head.action_encoder(actions, t, ids)}, ["actions"]
    )
    head.dit_engine = _Engine(
        lambda sa, vl, t, **masks: {"output": head.model(sa, vl, t, **masks)},
        ["sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"],
    )
    head.action_decoder_engine = _Engine(
        lambda output, ids: {"output": head.action_decoder(output, ids)}, ["model_output"]
    )


def _pytorch(head, rtc=True):
    backbone_output, action_input = _inputs(rtc)
    torch.manual_seed(11)
    with torch.no_grad():
        return head.get_action(backbone_output, action_input, RTC_OPTIONS if rtc else None)[
            "action_pred"
        ].float()


def _relative_error(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


def test_apply_rtc_prefix_seeds_freezes_and_ramps():
    actions = torch.zeros(1, HORIZON, 2)
    strength = torch.ones_like(actions)
    prev = torch.arange(HORIZON * 2, dtype=torch.float32).reshape(1, HORIZON, 2)
    apply_rtc_prefix(actions, strength, prev, RTC_OPTIONS)
    # New-chunk steps 0..3 are rows 2..5 of the previous chunk.
    assert torch.equal(actions[0, :4], prev[0, 2:6])
    assert torch.equal(actions[0, 4:], torch.zeros(2, 2))
    assert torch.equal(strength[0, 0], torch.zeros(2))  # frozen
    ramp = strength[0, 1:4, 0]
    assert torch.all(ramp > 0) and torch.all(ramp < 1) and torch.all(ramp[1:] > ramp[:-1])
    assert torch.equal(strength[0, 4:], torch.ones(2, 2))
    with pytest.raises(ValueError, match="rtc_frozen_steps"):
        apply_rtc_prefix(actions, strength, prev, {**RTC_OPTIONS, "rtc_frozen_steps": 5})


def test_rtc_changes_the_pytorch_result():
    head = _head()
    seeded = _pytorch(head, rtc=True)
    plain = _pytorch(head, rtc=False)
    assert _relative_error(seeded, plain) > 0.05
    # The frozen step is the previous chunk's committed action, exactly.
    _, action_input = _inputs()
    assert torch.allclose(seeded[0, 0], action_input["action"][0, 2], atol=1e-6)


def test_step_engines_match_pytorch_with_rtc(runtime):
    head = _head()
    expected = _pytorch(head)
    _install_step_engines(head)
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        actual = runtime.action_head_tensorrt_forward(
            head, backbone_output, action_input, RTC_OPTIONS
        )["action_pred"]
    assert _relative_error(actual, expected) < 1e-5


def _fused_engine(head):
    export = pytest.importorskip("trt.export_onnx_n1d7", exc_type=ImportError)
    wrapper = export._FusedActionHead(head, STEPS, embodiment_id=0).eval()
    wrapper.install_shared_kv()
    wrapper.install_export_forwards()

    def run(**kw):
        vel = kw.get("vel_strength", torch.ones_like(kw["init_actions"], dtype=torch.float32))
        out = wrapper(
            kw["backbone_features"].float(),
            kw["state"].float(),
            kw["init_actions"].float(),
            vel.float(),
            kw["image_index"],
            kw["text_index"],
        )
        return {"actions": out}

    names = ["backbone_features", "state", "init_actions", "vel_strength", "image_index"]
    return _Engine(run, [*names, "text_index"])


def test_fused_graph_matches_pytorch_with_rtc(runtime):
    head = _head()
    expected = _pytorch(head)
    engine = _fused_engine(head)
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        actual = runtime._fused_action_head_forward(
            head, engine, backbone_output, action_input, RTC_OPTIONS
        )["action_pred"]
    # The runtime feeds the fused graph BF16 features and state, as the engine expects.
    assert _relative_error(actual, expected) < 2e-2
    _, action_input = _inputs()
    assert torch.allclose(actual[0, 0].float(), action_input["action"][0, 2], atol=1e-6)


def test_pre_rtc_fused_engine_falls_back_to_step_engines(runtime):
    head = _head()
    expected = _pytorch(head)
    # A fused engine exported before RTC support: no vel_strength input. Must not run.
    old_fused = _Engine(
        lambda **kw: pytest.fail("RTC request reached a fused engine without vel_strength"),
        ["backbone_features", "state", "init_actions", "image_index", "text_index"],
    )
    _install_step_engines(head)
    head._trt_engine_source = ("unused", {})
    head._trt_fused_engines = {STEPS: old_fused}
    head._trt_step_engines_loaded = True
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        actual = runtime.action_head_tensorrt_forward(
            head, backbone_output, action_input, RTC_OPTIONS
        )["action_pred"]
    assert old_fused.calls == 0
    assert _relative_error(actual, expected) < 1e-5
