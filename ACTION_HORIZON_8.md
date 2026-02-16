# Action Horizon 16 → 8: Impact Analysis

## Summary

Reducing `action_horizon` from 16 to 8 yields a **modest ~5-7% E2E speedup** (~8-12ms saved per inference). The backbone, not the DiT, is the bottleneck on Orin AGX.

## How action_horizon flows through the model

```
actions [B, action_horizon, action_dim]        # [1, 16, 29] → [1, 8, 29]
        ↓ action encoder
action_features [B, action_horizon, 1536]      # [1, 16, 1536] → [1, 8, 1536]
        ↓ concat with state (1 token)
sa_embs [B, 1+action_horizon, 1536]            # [1, 17, 1536] → [1, 9, 1536]
        ↓ DiT forward (32 layers × 4 denoising steps)
output [B, 1+action_horizon, 1024]             # [1, 17, 1024] → [1, 9, 1024]
        ↓ action decoder
pred_velocity [B, action_horizon, action_dim]  # [1, 16, 29] → [1, 8, 29]
```

Key files:
- Config default: `gr00t/configs/model/gr00t_n1d6.py:58`
- Action encode/concat/decode: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py:210,219,243,352-381`
- DiT forward (32 layers): `gr00t/model/modules/dit.py:242-360`
- Denoising loop (4 Euler steps): `gr00t/model/gr00t_n1d6/gr00t_n1d6.py:349-385`
- ONNX dynamic axes: `scripts/deployment/export_onnx_n1d6.py:187-192`

## Compute cost breakdown

### DiT self-attention (quadratic in sa_seq_len)

| action_horizon | sa_seq_len | Self-attn cost |
|---|---|---|
| 16 | 17 | 17² = 289 |
| 8 | 9 | 9² = 81 |

Self-attention cost drops **3.6x**, but it's only ~30-40% of each DiT layer. Cross-attention, FFN, and LayerNorm scale linearly (47% reduction).

### E2E impact

| Component | Time (ms) | % of E2E | Speedup from horizon=8 |
|---|---|---|---|
| Backbone (PyTorch BF16) | ~157 | ~54% | None |
| DiT (TRT FP16) | ~72 | ~25% | ~20-30% faster |
| Data processing + overhead | ~60 | ~21% | None |

**Estimated savings: ~8-12ms → ~161ms total (~6.2 Hz vs 5.8 Hz)**

The denoising loop still runs 4 Euler steps regardless of action_horizon. Each step is faster, but the number of iterations is unchanged.

## The trade-off

- **Gain:** ~6% faster inference
- **Loss:** Planning horizon cut from 1.07s to 0.53s (at 15 FPS)

However, the eval loop (`gr00t/eval/real_robot/SO100/eval_so100.py:170`) already only executes 8 of the 16 predicted steps. Training with horizon=8 would ensure the model is optimized for the actions it actually uses, even though the latency improvement is marginal.

## Alternatives with larger impact

| Approach | Potential savings | Notes |
|---|---|---|
| Reduce denoising steps (4→2) | ~36ms (2 full DiT passes) | Requires retraining or distillation |
| Lower image resolution | Variable (backbone-bound) | Affects action quality |
| DiT INT8 TRT | ~0ms (memory-bound on SM87) | Already tested, INT8 ≈ FP16 latency |
| Backbone TRT | Not viable | FP16 overflow destroys quality on SM87 |
| torch.compile backbone | -34ms (slower) | 191ms vs 157ms flash attention |
