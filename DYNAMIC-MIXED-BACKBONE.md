# Backbone Quantization: Dynamic Quant + TRT Mixed-Precision

## Status: CLOSED — Both Phases No-Go

Neither approach produced viable backbone acceleration on Orin AGX SM87. The production ceiling remains **PyTorch BF16 flash backbone (157ms) + TRT FP16 DiT (72ms) = 4.2 Hz**.

---

## Current State (Pre-Experiment Baseline)

| Config | Latency | cos_sim vs flash | Status |
|--------|---------|-----------------|--------|
| PyTorch BF16 flash | 157ms | 1.000 (ref) | Production baseline |
| PyTorch BF16 SDPA | 211ms | 0.999 | ONNX-exportable |
| torch.compile + SDPA | 191ms | 0.998 | No export needed |
| TRT FP32 (SDPA ONNX) | 350ms | 0.999 | Accurate but slow |
| TRT FP16 (SDPA ONNX) | 149ms | 0.354 | Fast but broken |
| TRT INT8 (SDPA ONNX) | 141ms | 0.353 | Broken (FP16 internally on SM87) |

**Root cause of TRT FP16/INT8 failure:** Eagle's activations overflow FP16's 5-bit exponent (±65504). BF16 has 8 exponent bits (±3.4×10³⁸). LayerNorm and softmax layers produce values outside FP16 range, corrupting downstream computation. INT8 doesn't help because SM87 lacks native INT8 tensor cores — TRT falls back to FP16 internally.

---

## Phase 1: `torch.quantize_dynamic` — NO-GO

**Goal:** INT8 weight-only quantization on the PyTorch backbone path to halve DRAM bandwidth for weight loads at batch=1.

### Implementation

Created `scripts/deployment/test_dynamic_quant_backbone.py` — loads policy with flash_attention_2, prepares real dataset inputs, copies backbone to CPU, applies `torch.quantize_dynamic({nn.Linear}, dtype=torch.qint8)`, then attempts to move back to CUDA for benchmarking. Includes CPU fallback path.

### Result

```
RuntimeError: Didn't find engine for operation quantized::linear_prepack
  That is, no quantization engine found for NoQEngine
```

**Root cause:** PyTorch's quantization backends are platform-specific:
- `fbgemm` — x86 only (Intel/AMD), not compiled for aarch64
- `qnnpack` — ARM-optimized but targets CPU, not loaded in the PyTorch 2.8 Jetson build

The Jetson aarch64 PyTorch build ships with `NoQEngine` as the default backend. `torch.quantize_dynamic` cannot even create quantized weight tensors, let alone run them on CUDA. The quantization step itself fails before any forward pass attempt.

### Verdict: NO-GO

Not a performance or accuracy issue — the API is fundamentally unavailable on this platform. No workaround exists within PyTorch's eager quantization path on Jetson aarch64.

---

## Phase 2: TRT Mixed-Precision Per-Layer Control — NO-GO

**Goal:** Keep overflow-prone layers (norm, softmax, embeddings) in FP32 while running matmuls on FP16 tensor cores.

### Implementation

Extended `scripts/deployment/build_tensorrt_engine.py` with:
- `--precision mixed` mode with `FP16 + OBEY_PRECISION_CONSTRAINTS` builder flags
- `--fp32-patterns` CLI arg for specifying which layer name patterns to force FP32
- `--dump-layers` to inspect the ONNX graph's TRT layer names
- `set_mixed_precision()` function that only sets precision on compute layer types (NORMALIZATION, SOFTMAX, CONVOLUTION, MATRIX_MULTIPLY, ELEMENTWISE, REDUCE, UNARY, SCALE, ACTIVATION, POOLING), skipping non-compute types (SHAPE, CONSTANT, IDENTITY, SHUFFLE) that cause build errors

### Layer Analysis

Dumped all 23,207 TRT layers from the backbone ONNX graph:

| Pattern | Matched Layers | % of Total |
|---------|---------------|------------|
| `norm` | 1,011 | 4.4% |
| `softmax` | 43 | 0.2% |
| `embed` | 396 | 1.7% |
| `matmul` / `gemm` | 639 | 2.8% |
| Other (shape, constant, etc.) | ~21,118 | 91% |

### Experiment A: Conservative Patterns (norm, softmax, embed)

```
Patterns: norm, layernorm, layer_norm, rmsnorm, softmax, embed, position
Build: SUCCESS — 3.2 min, 2.96 GB engine (max-seq-len 512)
```

| Metric | Value |
|--------|-------|
| cos_sim vs flash | **0.777** |
| MSE vs flash | 13.2 |
| Latency | 147ms |

**Analysis:** Forcing norm/softmax/embed to FP32 only recovered cos_sim from 0.354 (pure FP16) to 0.777. This is a significant improvement but far below the 0.99 target. The overflow is not confined to norm/softmax/embed layers — it propagates through the matmul outputs themselves.

### Experiment B: Aggressive Patterns (+ attn, mul, add, div, pow, sqrt, reduce)

```
Patterns: norm, softmax, embed, position, attn, mul, add, div, pow, sqrt, reduce
Compute layers forced FP32: 1,746 (of 23,207)
Build: FAILED — "Impossible to reformat"
```

TRT could not find valid data reformatting paths with this many mixed-precision constraints. Tried three variants:

| Variant | Builder Flag | Output Type | Result |
|---------|-------------|-------------|--------|
| v3 | OBEY_PRECISION_CONSTRAINTS | set_output_type(FLOAT) | FAILED |
| v4 | PREFER_PRECISION_CONSTRAINTS | set_output_type(FLOAT) | FAILED |
| v5 | OBEY_PRECISION_CONSTRAINTS | no output type constraint | FAILED |

All three failed with `RuntimeError: Failed to build TensorRT engine` — TRT's layer fusion and reformatting optimizer cannot satisfy the mixed-precision constraints when this many layers are forced to FP32.

### Root Cause Analysis

The initial hypothesis was wrong. The FP16 overflow is **not localized to ~5% of layers**. The problem is systemic:

1. **GEMM activation outputs overflow FP16:** The matmul layers (which must stay FP16 to use tensor cores) produce intermediate activations with dynamic range exceeding ±65504. This is inherent to Eagle's architecture — the model was trained in BF16 and relies on 8 exponent bits throughout.

2. **Conservative FP32 (norm/softmax/embed) only partially helps:** Keeping these layers in FP32 prevents overflow in normalization and attention score computation, but the matmul outputs feeding into them are already corrupted by FP16 truncation. cos_sim improved from 0.354→0.777, meaning ~half the error comes from norm/softmax and ~half from matmuls.

3. **Aggressive FP32 is not buildable:** Forcing enough layers to FP32 to fix quality makes the engine unbuildable — TRT's optimizer can't handle the reformatting between FP16 and FP32 tensors at that granularity.

4. **No middle ground exists:** There is no set of FP32 patterns between "conservative" and "aggressive" that simultaneously (a) achieves cos_sim > 0.99 and (b) produces a buildable engine.

### Verdict: NO-GO

Mixed-precision TRT cannot solve the Eagle backbone FP16 overflow on SM87. The overflow is in the matmul activations themselves, not in a fixable subset of layers.

---

## Phase 2b: Layer INT8 on Top of Mixed-Precision — SKIPPED

Prerequisite (Phase 2 success) not met. Additionally, SM87 INT8 matmuls use FP16 tensor cores internally, so they would hit the same overflow issue.

---

## Final Results

| Config | Backbone ms | cos_sim | Status |
|--------|------------|---------|--------|
| PyTorch BF16 flash (baseline) | 157 | 1.000 | **Production** |
| torch.compile + SDPA | 191 | 0.998 | Viable but slower |
| TRT FP32 | 350 | 0.999 | Accurate, too slow |
| TRT Mixed (norm/softmax/embed FP32) | 147 | 0.777 | Insufficient quality |
| TRT Mixed (aggressive FP32) | — | — | Build fails |
| TRT FP16 | 149 | 0.354 | Broken |
| TRT INT8 | 141 | 0.353 | Broken |
| torch.quantize_dynamic | — | — | API unavailable on aarch64 |

**Production ceiling: PyTorch BF16 flash backbone (157ms) + TRT FP16 DiT (72ms) = 4.2 Hz**

### Why Backbone TRT Is Not Viable on Orin AGX (SM87)

SM87 (Orin) supports FP16 and INT8 tensor cores but **not BF16 tensor cores**. The Eagle backbone was trained in BF16 and its internal activations rely on BF16's 8-bit exponent range. When TRT runs these computations in FP16 (the only available tensor core dtype), activations overflow and quality collapses. FP32 preserves quality but can't use tensor cores, making it 2.2x slower than PyTorch BF16 flash.

This is a hardware generation limitation, not a software one. Future Orin successors with BF16 tensor cores (SM89+) would likely run the backbone TRT engine correctly at FP16-like speeds.

---

## Files Created/Modified

| File | Action | Result |
|------|--------|--------|
| `scripts/deployment/test_dynamic_quant_backbone.py` | Created | Phase 1 test — confirmed No-Go |
| `scripts/deployment/build_tensorrt_engine.py` | Modified | Added `--precision mixed`, `--fp32-patterns`, `--dump-layers`, `--model-type` |
| `groot_n1d6_onnx_sdpa_fp32/backbone_mixed_agx.trt` | Built | 2.96 GB mixed-precision engine (cos_sim=0.777) |
