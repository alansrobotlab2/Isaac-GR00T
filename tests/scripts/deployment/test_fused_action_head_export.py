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

"""CPU checks for what the fused action-head export resolves at export time.

``_FusedActionHead`` replaces three runtime computations by export-time constants or
host-provided indices (see its docstring). Each must be exact, so each is pinned
against the unmodified module here, without a checkpoint or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest


DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")


@pytest.fixture
def export(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    monkeypatch.syspath_prepend(DEPLOY_DIR)
    import export_onnx_n1d7

    return export_onnx_n1d7


def test_bake_embodiment_weights_matches_gather_and_restores(export):
    """Baked weights give the gathered embodiment's output; the module is restored after."""
    from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP
    import torch

    torch.manual_seed(0)
    mlp = CategorySpecificMLP(num_categories=3, input_dim=5, hidden_dim=7, output_dim=4).eval()
    x = torch.randn(2, 6, 5)
    with torch.no_grad():
        expected = mlp(x, torch.tensor([2, 2]))
        with export._bake_embodiment_weights(mlp, 2) as num_baked:
            assert num_baked == 2
            # The ids are ignored once the weights are baked.
            actual = mlp(x, torch.tensor([0, 1]))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        # Restored: the gather is back and no baked buffers are left behind.
        assert "forward" not in mlp.layer1.__dict__
        assert not hasattr(mlp.layer1, "_baked_weight")
        restored = mlp(x, torch.tensor([1, 1]))
        torch.testing.assert_close(restored, mlp(x, torch.tensor([1, 1])))
        assert not torch.allclose(restored, expected)

    with pytest.raises(ValueError, match="outside"):
        with export._bake_embodiment_weights(mlp, 3):
            pass


def test_subset_cross_attention_matches_masked_attention(export):
    """Attending unmasked to the gathered token subset equals the boolean-mask form."""
    from diffusers.models.attention import Attention
    import torch

    torch.manual_seed(0)
    attn = Attention(query_dim=48, heads=4, dim_head=12, cross_attention_dim=32, bias=True).eval()
    attn.set_processor(export._SharedKVCrossAttnProcessor())
    tokens = torch.randn(1, 11, 32)
    queries = torch.randn(1, 7, 48)
    image_mask = torch.zeros(1, 11, dtype=torch.bool)
    image_mask[:, 2:9] = True
    attended = torch.ones(1, 11, dtype=torch.bool)
    attended[:, 10] = False  # a padding token: in neither subset

    image_index = (image_mask[0] & attended[0]).nonzero().squeeze(1)
    text_index = (~image_mask[0] & attended[0]).nonzero().squeeze(1)
    assert image_index.tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert text_index.tolist() == [0, 1, 9]

    with torch.no_grad():
        for mask, index in (
            (image_mask & attended, image_index),
            (~image_mask & attended, text_index),
        ):
            attn.processor.cache = None
            expected = attn(queries, encoder_hidden_states=tokens, attention_mask=mask)
            attn.processor.cache = None
            actual = attn(queries, encoder_hidden_states=tokens.index_select(1, index))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_precomputed_adaln_matches_module(export):
    """The per-step constants reproduce AdaLayerNorm and the DiT output modulation."""
    from gr00t.model.modules.dit import AdaLayerNorm
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    norm = AdaLayerNorm(16).eval()
    temb = torch.randn(1, 16)
    x = torch.randn(3, 5, 16)
    with torch.no_grad():
        expected = norm(x, temb.expand(3, -1))
        # What _FusedActionHead._precompute_conditioning stores, applied as _dit does.
        scale, shift = norm.linear(F.silu(temb)).chunk(2, dim=1)
        actual = norm.norm(x) * (1 + scale)[:, None] + shift[:, None]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
