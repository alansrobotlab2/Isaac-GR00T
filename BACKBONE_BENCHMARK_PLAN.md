# Backbone Pipeline Benchmark: Deep Dive

## Context

The GR00T N1.6 backbone TRT shows a **148x MSE regression** vs PyTorch baseline (0.067 vs 0.00045). We need to pinpoint exactly where quality drops by benchmarking each intermediate stage of the deployment pipeline. INT8 backbone (~0.066 MSE) is actually slightly better than FP16 backbone (~0.067 MSE), confirming quantization isn't the bottleneck — the degradation enters earlier.

## Approach

Create a new script `scripts/deployment/benchmark_backbone_pipeline.py` that tests every stage against the flash_attention_2 reference in a single run:

| Stage | What it tests | Isolates |
|-------|--------------|----------|
| A: PyTorch BF16 flash | Reference baseline | — |
| B: PyTorch FP32 eager | Same weights, different attention | flash→eager attention swap |
| C: ONNX Runtime | ONNX graph from eager export | ONNX tracing fidelity |
| D: TRT FP16 engine | TensorRT FP16 compilation | TRT graph optimization + FP16 |
| E: TRT INT8 engine | TensorRT INT8 compilation | INT8 quantization delta |

**Output:** Summary table with MSE/MAE/cosine/max_abs + latency per stage, plus an **incremental degradation** section showing each stage vs its predecessor to pinpoint the exact bottleneck.

## Why a New Script (Not Extending Existing)

- `validate_backbone_onnx.py` — covers stages A-C but has no TRT, no latency, no multi-sample support
- `test_sdpa_backbone.py` — has latency benchmarking but no ONNX/TRT stages
- `verify_backbone_onnx.py` — best `compare_outputs()` and `BackboneIOCapture` but structured as pair comparison
- None of these test TRT engines directly

The new script **imports** from these rather than copy-pasting.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/deployment/benchmark_backbone_pipeline.py` | **Create** | Full pipeline benchmark |

## Reusable Functions (import, don't copy)

| Function | Source File | Used For |
|----------|------------|----------|
| `swap_flash_to_eager()` | `validate_backbone_onnx.py:36` | Stage B attention swap |
| `_rec_to_dtype()` | `validate_backbone_onnx.py:26` | Recursive dtype conversion |
| `BackboneIOCapture` | `verify_backbone_onnx.py:78` | Capturing backbone I/O via hooks |
| `compare_outputs()` | `verify_backbone_onnx.py:212` | Detailed metrics with percentiles + per-token breakdown |
| `prepare_observation()` | `verify_backbone_onnx.py:55` | Data prep from dataset |
| `compute_metrics()` | `benchmark_quantization.py:67` | Compact MSE/MAE/cos/max_abs |
| `TensorRTBackboneWrapper` | `standalone_inference_script.py:321` | TRT backbone inference |

## Implementation Structure

### 1. Data Collection (10 samples: 5 trajectories x 2 steps)
- Load policy with flash_attention_2 (BF16)
- Register `BackboneIOCapture` hook on backbone
- For each trajectory/step: run backbone, store inputs + reference outputs
- Reset capture between calls

### 2. Stage Runners

**Stage B (eager):** `deepcopy` backbone → `swap_flash_to_eager()` → cast to FP32 → run all samples → delete copy

**Stage C (ONNX):** Load `ort.InferenceSession` → convert pixel_values from list to `[N,C,H,W]` numpy → run → unload

**Stage D/E (TRT):** Load `TensorRTBackboneWrapper` → pass pixel_values as list (wrapper handles stacking internally) → run all samples → unload

### 3. Pixel Values Format (critical detail)

| Stage | pixel_values format | Notes |
|-------|-------------------|-------|
| PyTorch (A, B) | `list of [C,H,W]` tensors | Eagle model expects list |
| ONNX (C) | `[N,C,H,W]` numpy | ONNX wrapper unstacks to list internally |
| TRT (D, E) | list or stacked tensor | `TensorRTBackboneWrapper` handles both |

### 4. Latency Benchmarking
- Pattern from `test_sdpa_backbone.py:80`: warmup → timed iterations → median/P90
- Run on first sample only (latency is shape-dependent, not data-dependent)
- `torch.cuda.synchronize()` barriers around each iteration

### 5. Memory Management
- Sequential stage execution (one TRT engine at a time)
- `gc.collect()` + `torch.cuda.empty_cache()` between stages
- `deepcopy` for eager backbone deleted immediately after use

### 6. TRT Availability
- Conditional import: `TRT_AVAILABLE = False` → try import tensorrt
- Stages D/E skip gracefully with info message when TRT unavailable
- Allows running PyTorch+ONNX stages on host without Docker

## Expected Output

```
================================================================================
BACKBONE PIPELINE BENCHMARK (10 samples, 5 trajectories x 2 steps)
================================================================================
Stage                           MSE          MAE    Cos Sim    Max Abs  Median ms   P90 ms
--------------------------------------------------------------------------------------------
A: PyTorch BF16 flash (ref)  baseline     baseline   baseline   baseline     93.2     95.1
B: PyTorch FP32 eager        0.000032     0.0038     0.99997    0.0412       95.8     97.3
C: ONNX Runtime              0.000045     0.0042     0.99996    0.0523      112.4    115.2
D: TRT FP16 engine           0.067000     0.1830     0.98200    2.3400      217.1    220.5
E: TRT INT8 engine           0.068000     0.1845     0.98150    2.4100      217.3    221.0

Incremental degradation (each stage vs previous):
  B vs A (flash→eager):    MSE +0.000032  (attention implementation)
  C vs B (eager→ONNX):     MSE +0.000013  (ONNX tracing)
  D vs C (ONNX→TRT FP16):  MSE +0.066955  ← BOTTLENECK
  E vs D (FP16→INT8):      MSE +0.001000  (INT8 quantization)
================================================================================
```

## CLI Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--model_path` | `alfie-gr00t/checkpoint-10000` | Checkpoint |
| `--dataset_path` | `alfiebot.CanDoChallenge` | Dataset |
| `--embodiment_tag` | `new_embodiment` | Embodiment |
| `--onnx_path` | `./groot_n1d6_onnx/backbone_model.onnx` | ONNX model |
| `--trt_fp16_path` | `./groot_n1d6_onnx/backbone_fp16_agx.trt` | FP16 engine |
| `--trt_int8_path` | `./groot_n1d6_onnx/backbone_int8_agx.trt` | INT8 engine |
| `--num_trajs` | `5` | Trajectories to sample |
| `--steps_per_traj` | `2` | Steps per trajectory |
| `--latency_iters` | `20` | Timed iterations |
| `--skip_latency` | flag | Skip latency benchmarks |
| `--skip_onnx` | flag | Skip ONNX stage |
| `--skip_trt` | flag | Skip TRT stages |

## Verification

Run inside Docker:
```bash
python scripts/deployment/benchmark_backbone_pipeline.py \
    --model_path /workspace/gr00t/alfie-gr00t/checkpoint-10000 \
    --dataset_path /workspace/gr00t/alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --onnx_path /workspace/gr00t/groot_n1d6_onnx/backbone_model.onnx \
    --trt_fp16_path /workspace/gr00t/groot_n1d6_onnx/backbone_fp16_agx.trt \
    --trt_int8_path /workspace/gr00t/groot_n1d6_onnx/backbone_int8_agx.trt
```

Expected: Summary table clearly showing where the 148x MSE regression enters the pipeline (likely stage C→D or B→C).
