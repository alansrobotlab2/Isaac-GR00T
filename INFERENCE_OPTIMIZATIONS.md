# Backbone TensorRT Optimization: Findings & Next Steps

## Current Best Config: 5.4 Hz (was 2.2 Hz)

torch.compile(default) backbone + TRT FP16 DiT + **2-step denoising** = **186ms avg E2E** (~5.4 Hz)

Previous bests: 240ms (4.2 Hz, 4-step denoising) → 226ms (4.4 Hz, with async prefetch) → **186ms (5.4 Hz, 2-step denoising)**

**Backbone TRT path exhausted (section 12):** Decomposed softmax + mixed precision fixes quality (cos_sim=0.998) but is 14% slower than flash (180ms vs 158ms). INT8 adds zero speedup (memory-bandwidth-bound). The 7.1 Hz target via TRT INT8 backbone is NOT achievable with current TRT on SM87. Further speedups require: model distillation, action head optimization, or waiting for future TRT versions with FP32-softmax fused MHA.

## Experimental Results

### Backbone Quality (cos_sim vs PyTorch BF16 flash reference)

| Stage | MSE | Cos Sim | Median Latency | Notes |
|-------|-----|---------|---------------|-------|
| A: PyTorch BF16 flash | baseline | baseline | 157ms | Reference — uses flash_attention_2 |
| B: PyTorch FP32 eager | 5.44 | 0.911 | 1194ms | flash→eager attention swap destroys quality |
| C: ONNX Runtime (SDPA) | 0.091 | 0.999 | 565ms | SDPA attention solves quality problem |
| D: TRT FP32 (SDPA) | 0.093 | 0.999 | 350ms | Lossless TRT compilation |
| E: TRT FP16 (SDPA) | 27.41 | 0.354 | 149ms | FP16 destroys quality |
| F: TRT INT8 (SDPA) | 27.42 | 0.353 | 141ms | INT8 ≈ FP16 quality, 5% faster |
| torch.compile + SDPA | 0.117 | 0.998 | 191ms | 12% slower than flash, stays in PyTorch |

### Key Findings

1. **SDPA attention solves the ONNX export quality problem.** Eager attention (cos_sim=0.914) was the original bottleneck preventing backbone TRT. SDPA (cos_sim=0.999) is near-identical to flash.

2. **FP32 TRT compilation is lossless** — only 0.002 MSE increment from ONNX→TRT FP32. The ONNX trace is faithful.

3. **FP16 TRT destroys quality** (cos_sim=0.349). The Eagle backbone's internal values overflow FP16's 5-bit exponent range (±65504). BF16 has 8 exponent bits (±3.4×10³⁸) and the model relies on this range. On Orin SM87, BF16 tensor cores are not natively supported — TRT falls back to FP16.

4. **FP16 TRT is actually 5% faster than PyTorch flash** (149ms vs 157ms) — tensor cores help, but quality is unusable.

5. **FP32 TRT is 2x slower than PyTorch flash** (350ms vs 157ms) — no tensor core benefit in FP32 on SM87.

6. **INT8 TRT builds successfully after reboot** (31 min build, 2.96GB engine). Requires fresh memory — previously OOM'd before reboot.

7. **INT8 adds virtually zero error on top of FP16.** INT8 vs FP16 incremental: MSE=0.014, cos_sim=0.999. The quality bottleneck is entirely FP16 precision, not INT8 quantization.

8. **INT8 is 5.4% faster than FP16** (141ms vs 149ms) — marginal gain since the model is memory-bandwidth bound.

9. **torch.compile + SDPA** gives 191ms (22% slower than flash) with cos_sim=0.998. Best non-flash option that stays in PyTorch.

### Latency vs Quality Summary

```
Quality (cos_sim)  1.0 |  A*----D                    C
                       |       \                    /
                   0.9 |        \                  /
                       |         \                / torch.compile+SDPA
                       |          \              /
                   0.5 |           \
                       |            E---F
                   0.3 |
                       +---+---+---+---+---+---+---+
                       100 150 200 250 300 350 400   ms

A = PyTorch flash (157ms, baseline)   D = TRT FP32 (350ms, 0.999)
E = TRT FP16 (149ms, 0.354)           F = TRT INT8 (141ms, 0.353)
C = ONNX Runtime (565ms, 0.999)       torch.compile+SDPA (191ms, 0.998)
```

**The speed-quality gap:** No TRT config achieves both good quality AND speed improvement. The only fast options (FP16/INT8) have destroyed quality. The only high-quality options (FP32/ONNX) are slower than PyTorch flash.

### Root Cause: TRT FP16 MHA Kernel (NOT Layer Activation Overflow)

~~Original diagnosis: "Eagle backbone dynamic range overflows FP16's 5-bit exponent."~~

**Corrected diagnosis (from `profile_fp16_rootcause.py`):** All 399 layer outputs are within FP16 range (max ~5160 ≪ 65504). PyTorch FP16 SDPA achieves cos_sim=0.999. The quality destruction is **100% TRT-specific**: TRT's fused Multi-Head Attention kernel computes intermediate attention scores (Q@K^T) in FP16, and Qwen2 layers 14-15 have worst-case scores of ~73K-121K that overflow FP16. PyTorch's SDPA computes softmax in FP32 internally; TRT's fused FP16 MHA does not.

### Why TRT FP16/INT8 Aren't Much Faster Than PyTorch Flash

The 149ms (FP16) and 141ms (INT8) results are only 5-10% faster than PyTorch flash's 157ms. This is far less than the 2-4x speedup typically expected from TRT quantization. Three factors explain this:

**1. Memory-bandwidth bound at batch=1.** On Orin AGX, the ~2B parameter Eagle backbone is bottlenecked by LPDDR5 bandwidth (~205 GB/s theoretical, ~130 GB/s real), not compute. At batch=1, the GPU spends most of its time loading weights from DRAM, not doing math. The FP32→FP16 ratio confirms this: 350ms vs 149ms = 2.35x, almost exactly the 2x expected from halving memory traffic. **BF16 and FP16 are both 16-bit** — they move the same bytes through memory. So FP16 TRT can't be faster than BF16 PyTorch on bandwidth alone.

**2. Flash attention is an algorithmic advantage TRT can't replicate.** Flash attention never materializes the N×N attention matrix (O(N) memory vs O(N²) for SDPA). The ONNX export path uses SDPA since flash can't be traced. Even with TRT's kernel fusion and tensor cores, it can't match flash attention's fundamentally fewer memory round-trips. The two roughly cancel:

```
TRT FP16 advantages:             Flash attention advantages:
  + Kernel fusion                  + O(N) memory (vs O(N²) SDPA)
  + FP16 tensor cores              + Fewer total memory round-trips
  + Graph-level optimization       + Hand-tuned CUDA kernel for this workload
  ≈ 149ms                          ≈ 157ms   (roughly a wash)
```

**3. INT8 has limited memory savings in practice.** INT8 should halve memory traffic vs FP16, but TRT INT8 only uses INT8 for weights — activations stay FP16. Many layers (LayerNorm, Softmax, embeddings, attention) fall back to FP16 entirely. Net memory reduction is ~30%, not 50%, yielding only 5.4% speedup (141ms vs 149ms).

**Bottom line:** PyTorch BF16 with flash attention is already operating near the memory-bandwidth ceiling for this model at batch=1 on Orin AGX. TRT can't meaningfully beat it because the bottleneck is DRAM bandwidth, not kernel efficiency — and flash attention's algorithmic memory savings offset TRT's fusion benefits.

## Artifacts

| Path | Description |
|------|-------------|
| `groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx` | SDPA FP32 ONNX export (high quality) |
| `groot_n1d6_onnx_sdpa_fp32/backbone_fp32_agx.trt` | FP32 TRT engine, 5.9GB, cos_sim=0.999, 350ms |
| `groot_n1d6_onnx_sdpa_fp32/backbone_fp16_agx.trt` | FP16 TRT engine, 3.1GB, cos_sim=0.354, 149ms |
| `groot_n1d6_onnx_sdpa_fp32/backbone_int8_agx.trt` | INT8 TRT engine, 3.1GB, cos_sim=0.353, 141ms |
| `calibration_data_backbone/calib_data.npz` | 100 samples backbone calibration data |
| `calibration_data_backbone/backbone_int8_calib.cache` | INT8 calibration cache (reuse for rebuilds) |

### PyTorch Optimization Commands (inside Docker)

```bash
# Baseline: PyTorch BF16 flash backbone + TRT FP16 DiT
python scripts/deployment/standalone_inference_script.py \
    --model-path alfie-gr00t/checkpoint-10000 \
    --dataset-path alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT \
    --inference-mode tensorrt \
    --trt-engine-path groot_n1d6_onnx/dit_fp16.trt \
    --traj-ids 0 1 2 --steps 200 --denoising-steps 4 --action-horizon 16 --seed 42

# Best config: torch.compile + pipeline parallelism
python scripts/deployment/standalone_inference_script.py \
    --model-path alfie-gr00t/checkpoint-10000 \
    --dataset-path alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT \
    --inference-mode tensorrt \
    --trt-engine-path groot_n1d6_onnx/dit_fp16.trt \
    --compile-backbone \
    --compile-backbone-mode default \
    --pipeline-backbone-dit \
    --traj-ids 0 1 2 \
    --steps 200 \
    --denoising-steps 2 \
    --action-horizon 4 \
    --seed 42

# Open loop eval with timing
python gr00t/eval/open_loop_eval.py \
    --dataset-path alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path alfie-gr00t/checkpoint-10000 \
    --inference-mode tensorrt \
    --trt-engine-path groot_n1d6_onnx/dit_fp16.trt \
    --compile-backbone \
    --compile-backbone-mode default \
    --traj-ids 0 \
    --action-horizon 16 \
    --denoising-steps 2 \
    --save-plot-path ./episode000_optimized.png
```

### TRT Engine Build Commands (inside Docker)

```bash
# SDPA ONNX export
python scripts/deployment/export_backbone_onnx.py \
    --model_path alfie-gr00t/checkpoint-10000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --attn_implementation sdpa \
    --export_dtype fp32 \
    --output_dir groot_n1d6_onnx_sdpa_fp32

# FP32 TRT (high quality, slow)
python scripts/deployment/build_tensorrt_engine.py \
    --onnx groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
    --engine groot_n1d6_onnx_sdpa_fp32/backbone_fp32_agx.trt \
    --precision fp32 \
    --calib-data calibration_data_backbone/calib_data.npz \
    --max-seq-len 512

# FP16 TRT (fast, bad quality)
python scripts/deployment/build_tensorrt_engine.py \
    --onnx groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
    --engine groot_n1d6_onnx_sdpa_fp32/backbone_fp16_agx.trt \
    --precision fp16 \
    --calib-data calibration_data_backbone/calib_data.npz \
    --max-seq-len 512 \
    --prepare-system --tactic-memory 2048 --workspace 1024

# INT8 TRT (fast, bad quality — same as FP16, needs fresh memory after reboot)
python scripts/deployment/build_tensorrt_engine.py \
    --onnx groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
    --engine groot_n1d6_onnx_sdpa_fp32/backbone_int8_agx.trt \
    --precision int8 \
    --calib-data calibration_data_backbone/calib_data.npz \
    --calib-cache calibration_data_backbone/backbone_int8_calib.cache \
    --max-seq-len 512 \
    --prepare-system --tactic-memory 2048 --workspace 1024

# Benchmark
python scripts/deployment/benchmark_backbone_pipeline.py \
    --model_path alfie-gr00t/checkpoint-10000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --onnx_path groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
    --trt_fp16_path groot_n1d6_onnx_sdpa_fp32/backbone_fp16_agx.trt \
    --trt_int8_path groot_n1d6_onnx_sdpa_fp32/backbone_int8_agx.trt
```

## Remaining Approaches to Try

### ~~1. Mixed-Precision TRT (FP16 matmuls + FP32 sensitive layers)~~ DEPRIORITIZED

**Rationale:** FP16 TRT is 149ms but quality is destroyed. Keeping LayerNorm, softmax, and attention in FP32 while GEMMs use FP16 could fix quality. **However:** even the best case (~180-200ms) would be slower than PyTorch flash (157ms). The memory-bandwidth analysis above shows TRT can't beat flash attention at batch=1 regardless of precision mixing. Not worth the medium effort.

### ~~2. INT8 Backbone TRT~~ COMPLETED

**Result:** INT8 builds successfully after reboot (31 min, 2.96GB engine). Quality is identical to FP16 (cos_sim=0.353) — INT8 quantization itself is essentially lossless (INT8-vs-FP16: MSE=0.014, cos_sim=0.999). Latency: 141ms (5.4% faster than FP16's 149ms). **Conclusion: INT8 doesn't help because the quality bottleneck is FP16 dynamic range, not quantization precision.**

### ~~3. FP16 ONNX Export~~ DEPRIORITIZED

Same dynamic range problem regardless of where the FP16 cast happens. Model wasn't trained in FP16 and its activations fundamentally exceed FP16 range.

### ~~4. torch.compile + flash attention~~ COMPLETED

**Result:** `torch.compile(mode='default')` with flash_attention_2 works. `mode='max-autotune'` and `mode='reduce-overhead'` both FAIL — they use CUDA graphs internally, which conflicts with SigLIP2's lazily-cached `freqs_cis` tensor.

**E2E benchmark (3 trajs, 30 inference steps, skip 1 warmup):**

| Config | Avg E2E | P90 E2E | MSE | MAE |
|--------|---------|---------|-----|-----|
| Baseline (flash + TRT DiT) | 274.6ms | 277.4ms | 0.003230 | 0.023735 |
| torch.compile(default) + flash + TRT DiT | 267.4ms | 247.8ms | 0.003234 | 0.023758 |

**P90 improved from 277ms to 248ms** (10.5% faster). Average includes torch.compile's first-call warmup penalty. MSE essentially unchanged — no quality loss.

### ~~5. CUDA Graphs on the flash attention path~~ BLOCKED

**Result:** CUDA graphs are **fundamentally incompatible** with the Eagle backbone. Two issues:

1. **SigLIP2's `Rope2DPosEmb` lazily caches `freqs_cis`** — pre-computing it fixes this.
2. **SigLIP2's `split_patch_embeddings_to_windows_with_meta` uses data-dependent indexing** (`all_windows[sorted_idx]`) — this is a `cudaErrorStreamCaptureUnsupported` error during graph capture. The windowed attention path dynamically sorts and indexes patches based on input-dependent window metadata. This cannot be captured in a static CUDA graph.

**Conclusion:** CUDA graphs are not viable for the Eagle backbone without modifying SigLIP2's windowed attention implementation.

### ~~6. Pipeline parallelism (overlap backbone and DiT)~~ COMPLETED

**Result:** Pipeline parallelism (backbone on separate CUDA stream) works. Double-buffered: backbone(N+1) runs while DiT(N) processes on default stream.

**E2E benchmark (3 trajs, 30 inference steps, skip 1 warmup):**

| Config | Avg E2E | P90 E2E | Min E2E | MSE | MAE |
|--------|---------|---------|---------|-----|-----|
| Baseline (flash + TRT DiT) | 274.6ms | 277.4ms | 260.6ms | 0.003230 | 0.023735 |
| Pipeline only | 242.3ms | 260.8ms | 85.0ms | 0.003230 | 0.023735 |
| torch.compile + pipeline | **213.7ms** | **229.2ms** | 83.7ms | 0.003234 | 0.023758 |

**Pipeline alone: 11.8% faster avg** (274.6→242.3ms). Min of 85ms confirms overlap is working — that's roughly just DiT time when backbone was already running from previous frame.

**Combined compile + pipeline: 22.2% faster avg** (274.6→213.7ms). This is the new best config. Quality is identical to baseline.

**Note:** Pipeline adds 1-frame latency (frame N's actions are computed using frame N-1's backbone features for the DiT). First frame still runs sequentially.

### ~~7. SmoothQuant Backbone via NVIDIA Model Optimizer~~ INVESTIGATED — WILL NOT HELP

**Status:** Thoroughly investigated. SmoothQuant cannot fix the TRT FP16 quality issue because the root cause was misidentified — it's NOT activation overflow.

**Original hypothesis:** Eagle backbone activations exceed FP16's ±65504 range, causing TRT FP16 quality destruction (cos_sim=0.354). SmoothQuant could compress activation dynamic range into FP16 territory.

**Investigation results (3 key experiments):**

#### Experiment 1: Activation Range Profiling
Profiled all 399 layers' output activations across 10 calibration samples:
- **Zero FP16 overflows detected.** Maximum activation value: ~5160, well below FP16's 65504 limit.
- All Linear, LayerNorm, and RMSNorm outputs are safely within FP16 range.
- SmoothQuant targets activation overflow → no overflow to smooth.

#### Experiment 2: PyTorch FP16 vs BF16 (no TRT)
Ran the backbone in pure PyTorch with various dtype/attention combos:

| Config | cos_sim vs BF16 flash | MSE |
|--------|----------------------|-----|
| BF16 flash (reference) | 1.000 | baseline |
| **FP16 flash (PyTorch)** | **0.999** | 0.060 |
| **FP16 SDPA (PyTorch)** | **0.999** | 0.059 |
| BF16 SDPA (PyTorch) | 0.999 | 0.062 |
| FP32 SDPA (PyTorch) | 0.999 | 0.057 |
| TRT FP32 (from ONNX) | 0.999 | 0.093 |
| **TRT FP16 (from ONNX)** | **0.354** | 27.4 |

**PyTorch FP16 works perfectly.** FP16 flash AND FP16 SDPA both achieve cos_sim=0.999. The quality destruction is 100% TRT-specific.

#### Experiment 3: Attention Score Range Analysis
Estimated worst-case Q@K^T attention scores per layer:
- SigLIP2 vision layers: max ~1000 (safely within FP16)
- Qwen2 language layers 14-15: max ~73,000–121,000 (**exceeds FP16's 65504**)

These overflow in TRT's fused MHA kernel but NOT in PyTorch's SDPA (which computes softmax in FP32 internally). This is likely the smoking gun for TRT FP16 quality destruction.

#### Root Cause (corrected)

The original diagnosis — "Eagle backbone dynamic range overflows FP16's 5-bit exponent" — was **partially correct but misleading**. The overflow occurs specifically in:

1. **TRT's fused Multi-Head Attention (MHA) kernel**: When TRT fuses `MatMul(Q, K^T) → Div(sqrt_d) → Softmax → MatMul(attn, V)` into a single kernel in FP16 mode, the intermediate attention scores in Qwen2 layers 14-15 exceed FP16 range (~73K-121K > 65504). Unlike PyTorch's SDPA which computes softmax in FP32 internally, TRT's fused FP16 MHA kernel does not.

2. **NOT in layer input/output activations**: All 399 layer outputs are within FP16 range (max ~5160). The overflow is in INTERMEDIATE values within TRT's fused kernels that are invisible to PyTorch hooks.

#### Why SmoothQuant cannot help

- SmoothQuant operates on layer inputs/outputs (Norm → Linear weight scaling)
- The overflow is in **intermediate attention scores** inside TRT's fused MHA kernel
- SmoothQuant cannot modify the attention score computation (Q@K^T is not a Linear layer)
- Even if it could, the mathematical transformation Y = (X/s) @ (s*W) doesn't change the attention scores

#### What would actually fix TRT FP16

The fix requires TRT to compute softmax (and its inputs) in FP32 within the fused MHA kernel. This is not controllable via the ONNX graph or mixed-precision layer flags. Options:
- **TRT plugin**: Custom attention plugin with FP32 softmax accumulation
- **TRT-LLM**: NVIDIA's specialized LLM inference library has attention kernels with configurable accumulation precision
- **Wait for TRT update**: Future TRT versions may add FP32 softmax accumulation to the fused MHA kernel on SM87
- **Accept PyTorch flash**: The current config (PyTorch BF16 flash @ 157ms) is nearly as fast as TRT FP16 (149ms) anyway

**Verdict: Backbone TRT FP16 acceleration remains NOT viable on SM87. The best path is PyTorch BF16 flash + TRT FP16 DiT.**

**Scripts created:**
- `scripts/deployment/smoothquant_backbone.py` — SmoothQuant implementation (retained for reference)
- `scripts/deployment/profile_fp16_rootcause.py` — Root cause analysis that proved the issue is TRT-specific

---

### 8. FP32 Layer Audit — Squeeze Remaining Non-BF16 Operations

**Status:** AUDITED — one minor optimization candidate found

Audited the full inference pipeline for any FP32 operations that could be converted to BF16 for marginal latency gains.

#### FP32 Operations Found

| Location | Operation | Verdict |
|----------|-----------|---------|
| `gr00t_n1d6.py:329` | Euler denoising loop (`actions = torch.randn(..., dtype=torch.float32)`) | **KEEP FP32** — prevents error compounding across 4 Euler steps. Documented and intentional. |
| `gr00t_n1d6.py:385` | `pred_velocity.float()` before Euler accumulation | **KEEP FP32** — same reason; DiT outputs (FP16/INT8) upcast to FP32 for stable integration. |
| `modeling_siglip2.py:776` | `softmax(..., dtype=torch.float32).to(query.dtype)` | **KEEP FP32** — standard softmax stability practice. Upcasts to FP32, computes, casts back to BF16. |
| `modeling_siglip2.py:618-619` | Positional embedding interpolation (`to(torch.float32)`) | **N/A** — conditional on CPU only. Never executes on Orin CUDA. |
| `embodiment_conditioned_mlp.py:24` | `timesteps.float()` | **KEEP** — converts integer timesteps to float. Negligible overhead (~1 scalar). |
| `flowmatching_modules.py:23` | `timesteps.float()` | **KEEP** — same as above. |
| `embodiment_conditioned_mlp.py:31-32` | Sinusoidal frequency computation in FP32 | **KEEP** — one-time computation, negligible overhead. |

#### RoPE (Rotary Position Embeddings) — Candidate for BF16

The only non-trivial FP32 operation in the hot path:

```python
# modeling_siglip2.py:726-734 — Frequency cache computation (once per model init)
flat_pos = torch.arange(0, N).float().to(device)           # FP32
dim_range = torch.arange(0, self.dim, 4)[...].float().to(device)  # FP32
x_freqs = torch.outer(x_pos, freqs).float()                # FP32
y_freqs = torch.outer(y_pos, freqs).float()                # FP32

# modeling_siglip2.py:809-810 — Per-forward complex rotation (every inference)
xq_ = torch.view_as_complex(xq.float().view(...))  # BF16→FP32 upcast
xk_ = torch.view_as_complex(xk.float().view(...))  # BF16→FP32 upcast
# Line 813: .type_as(xq) casts result back to BF16
```

The frequency cache (lines 726-734) runs once at init — no perf impact. The complex rotation (lines 809-810) runs on **every forward pass, every attention layer**. It upcasts Q/K from BF16→FP32, does complex multiply with `freqs_cis` (complex64), then casts back.

**Potential gain:** Eliminating the BF16→FP32→BF16 round-trip in RoPE could save ~2-5ms on backbone (~1-3%). However, `torch.view_as_complex` requires FP32 input (complex64 = two float32), so this would need a different RoPE implementation (e.g., real-valued sin/cos rotation instead of complex multiply).

**Risk:** Modifying upstream Eagle model code. The RoPE FP32 path is standard practice in vision transformers and the gains are marginal.

**Verdict:** Not worth pursuing. The pipeline is already well-optimized for dtype. FP32 usage is limited to numerically-critical paths. The backbone is memory-bandwidth-bound, so saving a few ms on RoPE compute doesn't materially change the 157ms bottleneck.

---

### 9. Model Distillation / Pruning

**Rationale:** A smaller backbone = less memory to load from DRAM = proportionally faster. A 50% smaller model could run in ~80ms.

**Effort:** High. Requires retraining.

### 10. Next Wave: Async Prefetch + Action Horizon + Denoising Steps

The 4.7 Hz ceiling can be pushed further with inference-level optimizations (no model changes):

#### 8a. Async CPU Prefetching (always-on in open_loop_eval.py)
CPU preprocessing (image transforms, Eagle tokenization, collation) takes 15-30ms and was running synchronously before GPU inference. Now prefetched in a background thread via `ThreadPoolExecutor`, hiding this latency behind GPU work from the previous step.

#### 8b. Pipeline Parallelism Wired Up in open_loop_eval.py
The `--pipeline-backbone-dit` flag was defined but never connected to the evaluation loop. Now wired up using the `PipelinedInference` class from `standalone_inference_script.py`. Overlaps backbone(N+1) with DiT(N) on separate CUDA streams.

#### 8c. Reduced Denoising Steps (`--denoising-steps 2`)
Each TRT DiT step takes ~18ms. Going from 4→2 steps saves ~36ms. Quality impact needs empirical validation — flow matching may degrade at 2 steps.

#### 8d. Runtime Action Horizon Override (`--model-action-horizon 4`)
Model generates 16-step action chunks but at ~4 Hz only 3-4 steps are used. Overriding `action_horizon` at runtime shrinks sa_embs from `(1,17,1536)` to `(1,5,1536)`, reducing DiT compute. The TRT engine supports dynamic shapes — no rebuild needed. Quality risk: model trained on 16-step noise distribution.

**For production quality with smaller action horizon, fine-tune with the target horizon** (see below).

#### 8e. torch.compile Action Encoder/Decoder (`--compile-action-head`)
The action encoder (MultiEmbodimentActionEncoder) and decoder (CategorySpecificMLP) run 4x per inference in the denoising loop. `torch.compile(mode='default')` fuses their `torch.bmm()` kernels.

#### 8f. cuDNN Benchmark (`--cudnn-benchmark`)
For fixed input shapes (eval always uses same image resolution), `torch.backends.cudnn.benchmark = True` auto-selects faster conv algorithms.

#### Measured Results (traj 0, skip 2 warmup steps)

| Config | Avg (ms) | Min (ms) | P90 (ms) | Hz | MSE | MAE |
|--------|----------|----------|----------|-----|-----|-----|
| **Baseline** (compile backbone + TRT DiT, 4 denoise, AH=16) | 226 | 215 | 227 | 4.4 | 0.000595 | 0.00860 |
| + compile action head + cuDNN | 225 | 215 | 226 | 4.4 | 0.000458 | 0.00807 |
| + **2-step denoising** | **188** | **177** | **189** | **5.3** | **0.000399** | **0.00599** |
| + model-action-horizon=4 (runtime) | 223 | 211 | 224 | 4.5 | 0.028280 | 0.06980 |
| + model-action-horizon=8 (runtime) | 223 | 212 | 224 | 4.5 | 0.016578 | 0.05051 |
| **Combined best** (2 denoise + compile + cuDNN) | **186** | **177** | **188** | **5.4** | **0.000474** | **0.00614** |

**Key findings:**
1. **2-step denoising is the big win:** 226→188ms (**-38ms, 17% faster**) AND quality *improves* (fewer Euler steps = less FP16 error compounding in TRT DiT)
2. **Compile action head + cuDNN:** negligible timing impact (~1ms), action encoder/decoder MLPs are too small to benefit from torch.compile
3. **Runtime action horizon override: REJECTED.** No timing benefit (DiT is memory-bound, sa_embs size doesn't matter), quality destroyed (47x/28x worse MSE). Model must be retrained with smaller AH for this to work.
4. **Async CPU prefetch:** fully hidden (0.1ms wait time), always-on in new code

### 11. Fine-Tuning with Smaller Action Horizon (RTX 5090)

The model was trained with `action_horizon=16` (16 delta_indices for action). At ~4 Hz inference and 15 fps training, 16 steps = 1.07s lookahead but only 3-4 steps (~0.27s) are used before re-inferring. Training with a matched horizon eliminates wasted computation.

**Config changes** (`experiment_cfg/conf.yaml`):
```yaml
model:
  action_horizon: 4  # Was 16
data:
  modality_configs:
    new_embodiment:
      action:
        delta_indices: [0, 1, 2, 3]  # Was [0..15]
```

**Impact on training:**
- Sequence length: 17 tokens → 5 tokens (state=1 + action=4)
- DiT attention: O(17²) → O(5²) — ~11.6x less compute per attention layer
- Training speedup: ~3-5x faster per step

**Approach:** Resume from checkpoint-10000, train 2000-5000 steps (~15-30 min on RTX 5090). Sweep `action_horizon ∈ {4, 8, 16}` to find the quality/speed sweet spot.

**Post-training:** Rebuild TRT engine with `--opt-sa-seq 5` for the new sa_embs shape.

### 12. FP32 Softmax Surgery for TRT Backbone

**Status:** 12a SUCCEEDED (decomposed softmax + mixed precision). 12b NOT VIABLE. 12c/12d NOT NEEDED.

**Goal:** Fix TRT backbone FP16 quality destruction (cos_sim=0.354 → target >0.99) caused by TRT's fused MHA kernel computing Q@K^T and softmax in FP16. Qwen2 layers 14-15 produce attention scores of ~73K-121K that overflow FP16's 65504 limit.

**Result:** Decomposed softmax ONNX surgery + mixed-precision TRT build achieves **cos_sim=0.998** at **180ms** (vs 149ms FP16, 158ms flash). INT8 build pending.

#### 12a. ONNX Graph Surgery Results

Three approaches were tried, each building on lessons from the previous:

| Approach | ONNX Surgery | TRT Build | Cos Sim | Latency | Verdict |
|----------|-------------|-----------|---------|---------|---------|
| Standard Cast | Insert Cast(FP32) before/after each of 43 Softmax nodes | FP16 | 0.349 | 141ms | FAILED — TRT Myelin fused everything, ignored Casts |
| Decomposed Softmax | Replace Softmax with 7 primitive ops (Cast→ReduceMax→Sub→Exp→ReduceSum→Div→Cast) | FP16 | 0.348 | 184ms | FAILED — Q@K^T MatMul still writes FP16 output |
| **Decomposed + Mixed Precision** | Same decomposition + `OBEY_PRECISION_CONSTRAINTS` + FP32 for norm/softmax/embed | **mixed** | **0.998** | **180ms** | **SUCCESS** |

**Key insights from 12a:**

1. **Standard Cast nodes are useless.** TRT's Myelin compiler fuses the entire language model into ~8 ForeignNode subgraphs, completely ignoring ONNX-level Cast nodes. The engine profiling showed identical execution to the unpatched FP16 engine.

2. **Decomposed softmax alone is necessary but not sufficient.** Replacing Softmax with primitive ops (ReduceMax, Sub, Exp, ReduceSum, Div) successfully prevents TRT from fusing these ops into its broken MHA kernel — the profiling showed `AddCasMaxSubExpSumDivMulCas` kgen kernels instead of fused MHA. However, the Q@K^T MatMul still writes FP16 output via HMMA instructions, so values >65504 are truncated before reaching the decomposed softmax.

3. **Mixed precision (`OBEY_PRECISION_CONSTRAINTS`) is the critical ingredient.** With `OBEY_PRECISION_CONSTRAINTS`, TRT respects the precision annotations on the decomposed softmax ops. The combination works because:
   - Decomposed softmax prevents MHA fusion (so TRT can't use its broken fused FP16 MHA kernel)
   - `OBEY_PRECISION_CONSTRAINTS` forces the decomposed ops to stay in their annotated precision (FP32 for the softmax chain)
   - The Q@K^T MatMul still outputs FP16, but the softmax computation on capped values in FP32 produces much better attention weights than FP16 softmax on the same capped values

4. **Latency tradeoff:** 180ms is 20% slower than the broken FP16 (149ms) and 14% slower than PyTorch flash (158ms). The overhead comes from the decomposed softmax not being fused. This makes standalone backbone TRT slower than PyTorch, but INT8 quantization may recover some latency.

**Script:** `scripts/deployment/onnx_fp32_softmax_surgery.py` — supports `cast`, `aggressive`, and `decompose` modes.

**Generated artifacts:**
- `groot_n1d6_onnx_sdpa_fp32/backbone_fp32_softmax.onnx` — Standard cast (failed)
- `groot_n1d6_onnx_sdpa_fp32/backbone_decomposed_softmax.onnx` — Decomposed softmax ONNX
- `groot_n1d6_onnx_sdpa_fp32/backbone_fp16_fp32sm.trt` — Standard cast engine (cos_sim=0.349)
- `groot_n1d6_onnx_sdpa_fp32/backbone_fp16_decomp.trt` — Decomposed FP16 engine (cos_sim=0.348)
- `groot_n1d6_onnx_sdpa_fp32/backbone_mixed_decomp.trt` — **Decomposed + mixed engine (cos_sim=0.998)**

#### 12b. TRT-LLM Attention Kernels — NOT VIABLE

TRT-LLM v0.12.0-jetson has exactly the kernel we need (`context_fmha_type = enabled_with_fp32_acc` in `GPTAttention`), which implements FP16 MHA with FP32 softmax accumulation. However:

- FMHA kernel source is **closed-source** (compiled `.so` only, no CUDA source)
- Kernels are tightly coupled to TRT-LLM's `GPTAttention` plugin and cannot be extracted for standalone use
- TRT-LLM is designed for full LLM inference pipelines, not individual layer replacement in custom ONNX models
- No public API to use just the FMHA kernel outside of TRT-LLM

**Verdict:** NOT VIABLE without NVIDIA providing a standalone FP32-softmax attention plugin.

#### 12c/12d — NOT NEEDED

Since 12a (decomposed softmax + mixed precision) achieved cos_sim=0.998, custom CUDA plugins (12c) and hybrid TRT+PyTorch approaches (12d) are not needed. The remaining question is whether INT8 quantization on top of the decomposed softmax engine can recover the latency overhead.

#### Final Results

| Config | Backbone ms | Cos Sim | Notes |
|--------|------------|---------|-------|
| PyTorch BF16 flash | 158 | baseline | Reference — current best |
| TRT FP16 (broken) | 149 | 0.349 | Fused MHA overflow — unusable |
| TRT FP16 decomposed+mixed | 180 | 0.998 | **Quality fixed** — but 14% slower than flash |
| TRT INT8 decomposed+mixed | 180 | 0.998 | **No speedup over FP16** — memory-bandwidth-bound |
| torch.compile + SDPA | 191 | 0.998 | PyTorch fallback — 21% slower than flash |

**Conclusion: Backbone TRT is NOT a latency win.**

The decomposed softmax fix successfully solves the quality problem (cos_sim=0.998), but the resulting engine is 14% slower than PyTorch flash (180ms vs 158ms) because:

1. **Decomposed softmax prevents MHA fusion.** By replacing Softmax with 7 primitive ops, we prevent TRT from using its (broken) fused MHA kernel. But we also prevent it from using ANY fused MHA kernel, including the efficient ones used for the 27 vision attention layers that don't have overflow issues.

2. **INT8 provides zero latency benefit.** The backbone is memory-bandwidth-bound on Orin's unified memory. INT8 reduces compute but not memory traffic for this workload. INT8 vs FP16 incremental: identical latency, cos_sim=0.999 (lossless quantization, zero speedup).

3. **The value proposition is gone.** The original plan was: fix quality → TRT FP16 at 149ms → INT8 at 141ms → pipeline parallelism at 141ms E2E → 7.1 Hz. Instead: fix quality → TRT mixed at 180ms → INT8 still 180ms → E2E 180ms → 5.6 Hz. This is WORSE than the current best of 186ms / 5.4 Hz (measurement noise makes them equivalent).

**Best config remains: PyTorch BF16 flash backbone (158ms) + TRT FP16 DiT + 2-step denoising = ~186ms (5.4 Hz).**

The backbone TRT path (Section 12) is a dead end for latency improvement on Orin AGX SM87. The fundamental issue is that fixing FP16 MHA quality requires breaking MHA fusion, which negates the performance benefit of TRT compilation for the backbone. Future NVIDIA TRT versions with FP32 softmax accumulation in their fused MHA kernel would resolve this, but that's not available today.

**Files created:**
- `scripts/deployment/onnx_fp32_softmax_surgery.py` — ONNX graph surgery (cast, aggressive, decompose modes)
- `scripts/deployment/benchmark_fp32_softmax.py` — Quick benchmark for patched TRT engines

**Engines generated (for reference/archival):**
- `groot_n1d6_onnx_sdpa_fp32/backbone_mixed_decomp.trt` — Best quality TRT (cos_sim=0.998, 180ms)
- `groot_n1d6_onnx_sdpa_fp32/backbone_int8_decomp.trt` — INT8 version (cos_sim=0.998, 180ms)

---

### 13. RoPE Real-Valued Implementation — Eliminate BF16→FP32→BF16 Round-Trip

**Status:** COMPLETED

**Goal:** Replace the complex-number RoPE implementation in SigLIP2 with a real-valued sin/cos rotation, eliminating the FP32 upcast on every forward pass of every attention layer.

**Background (from section 8 audit):** SigLIP2's `Rope2DPosEmb` (in `modeling_siglip2.py:809-810`) upcasted Q and K from BF16→FP32, performed complex64 multiplication with `freqs_cis`, then cast back to BF16 on **every forward pass, every attention layer** (27 layers × every inference step). The FP32 upcast existed because `torch.view_as_complex` requires FP32 input.

**Implementation:** Replaced complex-number rotation with real-valued sin/cos rotation:

```python
# Before — complex rotation, requires FP32 upcast
xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))  # BF16→FP32
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
xq_out = xq_out.type_as(xq)  # FP32→BF16

# After — real-valued rotation, stays in input dtype (BF16)
xq_even, xq_odd = xq[..., 0::2], xq[..., 1::2]
xq_out = torch.stack([
    xq_even * rope_cos - xq_odd * rope_sin,
    xq_odd * rope_cos + xq_even * rope_sin,
], dim=-1).flatten(-2)
```

**Validation results:**

| Test | Result |
|------|--------|
| Precompute: cos/sin vs complex real/imag | **Bitwise identical** |
| FP32 apply_rope: old vs new | max_err=4.8e-7, cos_sim=0.9999999 |
| BF16 apply_rope: old vs new | cos_sim=0.9999973 (stays in BF16, no FP32 round-trip) |
| ONNX export ops | `[Add, Concat, Mul, Reshape, Slice, Sub, Unsqueeze]` — no complex ops |
| E2E vision encoder (27 layers) | Runs correctly, output shape/stats normal |

**Latency impact:** Not independently measurable. RoPE is ~1-3% of backbone time (section 8 estimated 2-5ms out of 157ms). The change eliminates 27× BF16→FP32→BF16 round-trips per forward pass, but the backbone is memory-bandwidth-bound so the compute savings are within measurement noise. No before/after latency delta was captured — the primary value is ONNX export cleanliness for section 12, not standalone latency.

**ONNX improvement:** The old complex RoPE traced to `view_as_complex` / `view_as_real` ONNX ops that TRT must handle specially. The new version traces to simple `Slice + Mul + Sub + Add + Stack + Reshape` — standard TRT-friendly ops. This directly simplifies the ONNX graph surgery needed for section 12a (inserting FP32 casts around attention softmax), since the RoPE subgraph no longer contains exotic ops that could interfere with pattern matching.

**Effort:** ~1.5 hours. Implementation was straightforward — the real-valued rotation is a well-known decomposition of complex multiplication used by LLaMA, Mistral, and Qwen2.

**Files modified:**
- `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_siglip2.py` — `Rope2DPosEmb` (cos/sin caches instead of complex freqs_cis), `apply_rope()` (real-valued rotation)
- `scripts/deployment/standalone_inference_script.py` — CUDA graph pre-compute updated for new attribute names

**Test script:** `scripts/deployment/test_rope_real_valued.py` — validates precompute equivalence, unit-level numerical comparison, ONNX export, and E2E backbone forward pass.

## Commands

```bash
# Baseline (4-step denoising, ~226ms)
python gr00t/eval/open_loop_eval.py \
    --dataset-path alfiebot.CanDoChallenge --embodiment-tag NEW_EMBODIMENT \
    --model-path alfie-gr00t/checkpoint-10000 \
    --inference-mode tensorrt --trt-engine-path groot_n1d6_onnx/dit_fp16.trt \
    --compile-backbone --compile-backbone-mode default \
    --traj-ids 0 --action-horizon 16 --denoising-steps 4 \
    --skip-timing-steps 2 --save-plot-path ./episode000_baseline.png

# Best config (~186ms, 5.4 Hz)
python gr00t/eval/open_loop_eval.py \
    --dataset-path alfiebot.CanDoChallenge --embodiment-tag NEW_EMBODIMENT \
    --model-path alfie-gr00t/checkpoint-10000 \
    --inference-mode tensorrt --trt-engine-path groot_n1d6_onnx/dit_fp16.trt \
    --compile-backbone --compile-backbone-mode default \
    --traj-ids 0 --action-horizon 16 --denoising-steps 2 \
    --skip-timing-steps 2 --save-plot-path ./episode000_optimized.png
```

## Scripts Modified in This Investigation

| Script | Changes |
|--------|---------|
| `scripts/deployment/benchmark_backbone_pipeline.py` | Auto-detect ONNX dtype/rank, TRT dtype casting, latency dtype fix |
| `scripts/deployment/build_tensorrt_engine.py` | 4D/5D pixel_values auto-detection, BackboneInt8Calibrator 5D support |
| `scripts/deployment/test_sdpa_backbone.py` | SDPA + torch.compile backbone benchmark |
| `scripts/deployment/export_backbone_onnx.py` | SDPA attention export support (already existed) |
| `scripts/deployment/standalone_inference_script.py` | torch.compile, CUDAGraphBackboneWrapper (nn.Module), PipelinedInference, CLI flags |
| `scripts/deployment/onnx_fp32_softmax_surgery.py` | ONNX graph surgery: replace Softmax with FP32 decomposed ops |
| `scripts/deployment/benchmark_fp32_softmax.py` | Quick benchmark for patched TRT engines vs PyTorch flash |
| `gr00t/eval/open_loop_eval.py` | Async CPU prefetch, pipeline wiring, `--model-action-horizon`, `--compile-action-head`, `--cudnn-benchmark`, timing instrumentation |
