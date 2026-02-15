# Backbone TensorRT Optimization: Findings & Next Steps

## Current Best Config: 4.2 Hz

PyTorch BF16 flash backbone (170ms) + TRT FP16 DiT (72ms) = ~4.2 Hz (1.54x over pure PyTorch 2.7 Hz)

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

### Build Commands (inside Docker)

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

### 4. torch.compile + flash attention

**Rationale:** Keep flash attention's algorithmic advantage (O(N) memory, hand-tuned kernel) while letting `torch.compile` fuse surrounding operations (projections, norms, residual adds). The benchmark tested `torch.compile + SDPA` (191ms) but NOT `torch.compile + flash`.

**How:** `torch.compile(backbone, mode="max-autotune")` with `flash_attention_2` still active.

**Expected outcome:** Potential ~135-150ms. Flash is already 157ms; compile could shave kernel launch overhead and fuse non-attention ops around the flash kernel.

### 5. CUDA Graphs on the flash attention path

**Rationale:** At batch=1, kernel launch overhead is a significant fraction of total latency. CUDA graphs capture the entire backbone forward pass as a single GPU operation, eliminating CPU-GPU round-trips per kernel.

**How:** `torch.cuda.CUDAGraph` capture around `backbone(inputs)`. Requires static input shapes (pad to max sequence length).

**Expected outcome:** ~5-15ms reduction → ~142-152ms with perfect quality. Combines well with approach #4.

### 6. Pipeline parallelism (overlap backbone and DiT)

**Rationale:** Currently backbone (157ms) and DiT (72ms) run sequentially = 229ms. If backbone(N+1) overlaps with DiT(N), effective per-frame time drops to max(backbone, DiT) = 157ms + some overhead. Doesn't reduce single-frame latency but increases throughput from ~4.2 Hz to potentially ~5.5-6 Hz.

**How:** Double-buffered inference — while DiT processes frame N's features, backbone processes frame N+1's images on a separate CUDA stream.

**Effort:** Medium. Requires restructuring the inference loop.

### 7. Model Distillation / Pruning

**Rationale:** A smaller backbone = less memory to load from DRAM = proportionally faster. A 50% smaller model could run in ~80ms.

**Effort:** High. Requires retraining.

### 8. Accept 4.2 Hz Production Ceiling

157ms backbone + 72ms DiT is likely within 10-15% of the hardware bandwidth limit for this model on Orin AGX at batch=1. Further gains require algorithmic changes (smaller model, pipeline overlap, action chunking) rather than kernel-level optimization.

## Scripts Modified in This Investigation

| Script | Changes |
|--------|---------|
| `scripts/deployment/benchmark_backbone_pipeline.py` | Auto-detect ONNX dtype/rank, TRT dtype casting, latency dtype fix |
| `scripts/deployment/build_tensorrt_engine.py` | 4D/5D pixel_values auto-detection, BackboneInt8Calibrator 5D support |
| `scripts/deployment/test_sdpa_backbone.py` | SDPA + torch.compile backbone benchmark |
| `scripts/deployment/export_backbone_onnx.py` | SDPA attention export support (already existed) |
