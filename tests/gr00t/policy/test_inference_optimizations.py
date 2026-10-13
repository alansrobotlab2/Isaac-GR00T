"""
Test gr00t.policy.inference_optimizations: the prefetching inference loop and the
opt-in optimization switches. CPU-only; models and policies are lightweight fakes.
"""

from concurrent.futures import Future
import logging
from types import SimpleNamespace

from gr00t.policy import inference_optimizations as opt
from gr00t.policy.inference_optimizations import apply_inference_optimizations, run_frames
import pytest
import torch


def _done(value) -> Future:
    future = Future()
    future.set_result(value)
    return future


class _FakeLocalPolicy:
    def __init__(self, events):
        self.events = events

    def run_inference(self, collated_inputs, states):
        self.events.append(("infer", collated_inputs["id"]))
        return {"a": collated_inputs["id"]}


class _FakeRemotePolicy:
    def __init__(self, events):
        self.events = events

    def get_action(self, observation):
        self.events.append(("remote", observation))
        return {"a": observation}, {}


def _lazy_frames(events, frames):
    """Yield futures the way the scripts do: a frame is 'submitted' when it is pulled."""
    for frame in frames:
        events.append(("submit", frame[2]))
        yield _done(frame)


def test_run_frames_keeps_one_frame_of_lookahead():
    events = []
    frames = [({"id": i}, [{}], f"obs{i}") for i in range(3)]

    results = list(run_frames(_FakeLocalPolicy(events), _lazy_frames(events, frames)))

    assert [r[0] for r in results] == [{"a": 0}, {"a": 1}, {"a": 2}]
    assert [r[1] for r in results] == ["obs0", "obs1", "obs2"]
    # Frame N+1 is submitted before inference on frame N starts, and never further ahead.
    assert events == [
        ("submit", "obs0"),
        ("submit", "obs1"),
        ("infer", 0),
        ("submit", "obs2"),
        ("infer", 1),
        ("infer", 2),
    ]
    assert all(wait >= 0 and infer >= 0 for _, _, wait, infer in results)


def test_run_frames_remote_policy_uses_get_action():
    events = []
    frames = [(None, None, "obs0"), (None, None, "obs1")]

    results = list(run_frames(_FakeRemotePolicy(events), _lazy_frames(events, frames)))

    assert [r[0] for r in results] == [{"a": "obs0"}, {"a": "obs1"}]
    assert ("remote", "obs0") in events and ("remote", "obs1") in events


def test_run_frames_hands_upcoming_future_to_pipeline():
    seen = []

    class _Pipeline:
        def step(self, collated_inputs, states, next_frame=None):
            seen.append(
                (collated_inputs["id"], None if next_frame is None else next_frame.result()[2])
            )
            return {"a": collated_inputs["id"]}

    frames = [({"id": i}, [{}], f"obs{i}") for i in range(3)]
    list(run_frames(_FakeLocalPolicy([]), _lazy_frames([], frames), pipeline=_Pipeline()))

    assert seen == [(0, "obs1"), (1, "obs2"), (2, None)]


def test_run_frames_empty():
    assert list(run_frames(_FakeLocalPolicy([]), iter(()))) == []


class _Block(torch.nn.Module):
    def forward(self, x):
        return x


def _fake_policy(attn_implementation=None):
    backbone = _Block()
    if attn_implementation is not None:
        backbone.config = SimpleNamespace(_attn_implementation=attn_implementation)
    action_head = _Block()
    action_head.model = _Block()
    action_head.action_encoder = _Block()
    action_head.action_decoder = _Block()
    action_head.action_horizon = 40
    action_head.config = SimpleNamespace(action_horizon=40)
    model = SimpleNamespace(backbone=backbone, action_head=action_head)
    return SimpleNamespace(model=model)


@pytest.fixture
def fake_compile(monkeypatch):
    """Replace torch.compile with a marker so no real compilation happens."""
    calls = []

    def _compile(target, mode="default"):
        calls.append(mode)
        if isinstance(target, torch.nn.Module):
            target._compiled_mode = mode
            return target
        return lambda *a, **k: target(*a, **k)

    monkeypatch.setattr(opt.torch, "compile", _compile)
    return calls


def test_remote_policy_is_noop_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        apply_inference_optimizations(SimpleNamespace(), compile_dit=True, model_action_horizon=8)
    assert "ignored for a remote policy" in caplog.text


def test_no_flags_changes_nothing(fake_compile):
    policy = _fake_policy()
    apply_inference_optimizations(policy)
    assert fake_compile == []
    assert "forward" not in vars(policy.model.backbone)
    assert policy.model.action_head.action_horizon == 40


def test_model_action_horizon_override():
    policy = _fake_policy()
    apply_inference_optimizations(policy, model_action_horizon=8)
    assert policy.model.action_head.action_horizon == 8
    assert policy.model.action_head.config.action_horizon == 8


def test_compile_targets_and_mode(fake_compile):
    policy = _fake_policy(attn_implementation="sdpa")
    apply_inference_optimizations(
        policy,
        compile_backbone=True,
        compile_dit=True,
        compile_action_head=True,
        compile_mode="reduce-overhead",
    )
    head = policy.model.action_head
    assert "forward" in vars(policy.model.backbone)
    assert "forward" in vars(head.model)
    assert head.action_encoder._compiled_mode == "default"
    assert head.action_decoder._compiled_mode == "default"
    # backbone + DiT use the requested mode; encoder/decoder always use 'default'
    assert sorted(fake_compile) == ["default", "default", "reduce-overhead", "reduce-overhead"]


def test_compile_backbone_skipped_for_flash_attention(fake_compile, caplog):
    policy = _fake_policy(attn_implementation="flash_attention_2")
    with caplog.at_level(logging.WARNING):
        apply_inference_optimizations(policy, compile_backbone=True)
    assert fake_compile == []
    assert "forward" not in vars(policy.model.backbone)
    assert "--attn-implementation sdpa" in caplog.text


def test_compile_skips_components_replaced_by_tensorrt(fake_compile):
    policy = _fake_policy(attn_implementation="sdpa")
    head = policy.model.action_head
    # Mimic the TRT setup: backbone/DiT forwards monkey-patched, encoder/decoder deleted.
    policy.model.backbone.forward = lambda x: x
    head.model.forward = lambda x: x
    del head.action_encoder
    del head.action_decoder

    apply_inference_optimizations(
        policy, compile_backbone=True, compile_dit=True, compile_action_head=True
    )

    assert fake_compile == []


def test_compile_dit_skipped_when_dit_deleted(fake_compile):
    policy = _fake_policy()
    del policy.model.action_head.model
    apply_inference_optimizations(policy, compile_dit=True)
    assert fake_compile == []
