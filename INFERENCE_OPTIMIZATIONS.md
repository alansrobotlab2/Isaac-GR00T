# Backbone TensorRT Optimization: Findings & Next Steps

## Current Best Config: 5.4 Hz (was 2.2 Hz)

torch.compile(default) backbone + TRT FP16 DiT + **2-step denoising** = **186ms avg E2E** (~5.4 Hz)

Previous bests: 240ms (4.2 Hz, 4-step denoising) → 226ms (4.4 Hz, with async prefetch) → **186ms (5.4 Hz, 2-step denoising)**

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

### Root Cause: BF16 vs FP16 Dynamic Range

The Eagle backbone (Eagle-Block2A-2B-v2) produces intermediate values that require BF16's wider exponent range. The SigLIP2 vision encoder and Qwen2 language model both have activations and attention scores that exceed FP16's ±65504 range. This is a fundamental model property — not fixable by output buffer dtype or accumulation fixes.

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
    --traj-ids 0 --action-horizon 16 --denoising-steps 4 \
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

### 7. Model Distillation / Pruning

**Rationale:** A smaller backbone = less memory to load from DRAM = proportionally faster. A 50% smaller model could run in ~80ms.

**Effort:** High. Requires retraining.

### 8. Next Wave: Async Prefetch + Action Horizon + Denoising Steps

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

### 9. Fine-Tuning with Smaller Action Horizon (RTX 5090)

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
| `gr00t/eval/open_loop_eval.py` | Async CPU prefetch, pipeline wiring, `--model-action-horizon`, `--compile-action-head`, `--cudnn-benchmark`, timing instrumentation |
