# TensorRT Optimization: GR00T on Orin AGX

Complete guide to TensorRT optimization of the GR00T VLA model on Orin AGX,
with per-component accuracy validation.

**Platform:** NVIDIA Orin AGX, JetPack r36.5, CUDA 12.6, TensorRT 10.3
**Model:** GR00T N1.6-3B (EagleX backbone + 32-layer DiT)
**Checkpoint:** `alfie-gr00t/checkpoint-10000`
**Dataset:** `alfiebot.CanDoChallenge`
**Embodiment:** `new_embodiment`

---

## Final Results

| Mode | Backbone | DiT | BB ms | DiT ms | E2E ms | Hz | DiT MSE | DiT cos-sim | E2E MSE (test) |
|------|----------|-----|-------|--------|--------|----|---------|-------------|----------------|
| PyTorch BF16 | PyTorch BF16 | PyTorch BF16 | 157 | 207 | 364 | 2.7 | baseline | baseline | 0.087 |
| **TRT FP16 DiT** | PyTorch BF16 | TRT FP16 | 158 | 78 | 236 | **4.2** | 0.000020 | 0.999978 | 0.088 |
| **TRT INT8 DiT** | PyTorch BF16 | TRT INT8 | 157 | 77 | 234 | **4.3** | 0.000021 | 0.999976 | 0.088 |

### Per-Denoising-Step DiT Error

| Mode | Step 0 MSE | Step 1 MSE | Step 2 MSE | Step 3 MSE | Growth |
|------|-----------|-----------|-----------|-----------|--------|
| TRT FP16 | 0.000017 | 0.000019 | 0.000020 | 0.000024 | 1.42x |
| TRT INT8 | 0.000018 | 0.000020 | 0.000022 | 0.000025 | 1.39x |

### Key Findings

1. **DiT TRT is near-perfect:** Both FP16 and INT8 achieve cos_sim > 0.9999 with
   negligible E2E accuracy impact. INT8 offers no latency benefit over FP16 on
   Orin SM87 (77ms vs 78ms) because the DiT is memory-bound, not compute-bound.

2. **Backbone TRT is not viable** on Orin AGX:
   - FP16 TRT: catastrophically bad accuracy (MSE=25.6, cos_sim=0.30) due to
     the backbone's internal value range exceeding FP16 dynamic range
   - FP32 TRT: accurate but 2.2x slower than PyTorch BF16 (343ms vs 153ms)
     because flash attention cannot be exported to ONNX, and eager attention
     in FP32 is slower than flash attention in BF16
   - The ONNX trace itself is correct (validated via onnxruntime: MSE=0.000262
     vs PyTorch eager), but flash→eager attention swap causes inherent MSE=3.76

3. **Best configuration: PyTorch BF16 backbone + TRT FP16 DiT** at 4.2 Hz,
   a **1.54x speedup** over pure PyTorch (2.7 Hz) with negligible accuracy loss.

---

## Prerequisites

```bash
# Docker image must include: pycuda, onnxruntime-gpu, tensorrt
docker build -t gr00t-dev -f orin.Dockerfile .
```

## Launch Docker Container

All commands below run inside this container:

```bash
cd ~/Isaac-GR00T

docker run \
  -it --rm --runtime=nvidia \
  --network=host \
  -v $(pwd):/home/alfie/Isaac-GR00T \
  -w /home/alfie/Isaac-GR00T \
  gr00t-dev /bin/bash
```

---

## Quick Start (Recommended Path)

For the optimal PyTorch backbone + TRT FP16 DiT configuration:

```bash
# 1. Export DiT to FP32 ONNX
python3 scripts/deployment/export_onnx_n1d6.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --output_dir ./groot_n1d6_onnx \
  --video_backend torchcodec \
  --use_fp32

# 2. Build FP16 TRT engine (~2 minutes)
python3 scripts/deployment/build_tensorrt_engine.py \
  --onnx ./groot_n1d6_onnx/dit_model.onnx \
  --engine ./groot_n1d6_onnx/dit_fp16.trt \
  --precision fp16 \
  --model-type dit

# 3. Run inference
python3 scripts/deployment/standalone_inference_script.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --video_backend torchcodec \
  --inference_mode tensorrt \
  --trt_engine_path ./groot_n1d6_onnx/dit_fp16.trt
```

---

## Detailed Pipeline

### Phase 0: Precision Fixes (Already Applied)

Three code changes fix the ~65x MSE degradation observed with the original
BF16 ONNX → TRT path:

1. **FP32 output buffer** (`standalone_inference_script.py`):
   TRT output buffer uses `torch.float32` to prevent truncation before Euler integration.

2. **FP32 accumulation in denoising loop** (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`):
   Actions accumulated in FP32 regardless of model precision.

3. **FP32 ONNX export** (`export_onnx_n1d6.py --use_fp32`):
   Exports FP32 weights so TRT handles precision policy internally, avoiding
   BF16→ONNX→TRT conversion where precision semantics get lost.

### Phase 1: DiT TRT

#### 1.1 Export DiT ONNX (FP32)

```bash
python3 scripts/deployment/export_onnx_n1d6.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --output_dir ./groot_n1d6_onnx \
  --video_backend torchcodec \
  --use_fp32
```

**Output:** `./groot_n1d6_onnx/dit_model.onnx` (~0.9 MB with external data)

#### 1.2 Build FP16 TRT Engine

```bash
python3 scripts/deployment/build_tensorrt_engine.py \
  --onnx ./groot_n1d6_onnx/dit_model.onnx \
  --engine ./groot_n1d6_onnx/dit_fp16.trt \
  --precision fp16 \
  --model-type dit
```

**Build time:** ~2 minutes on Orin AGX.

#### 1.3 (Optional) Build INT8 TRT Engine

INT8 provides no latency benefit for DiT on Orin but can be tested for accuracy:

```bash
# Collect calibration data (5 trajectories = ~60 samples)
python3 scripts/deployment/calibrate_int8.py collect \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --output_dir ./groot_n1d6_onnx/calib_data/dit \
  --num_trajectories 5 \
  --component dit \
  --video_backend torchcodec

# Build INT8 engine
python3 scripts/deployment/build_tensorrt_engine.py \
  --onnx ./groot_n1d6_onnx/dit_model.onnx \
  --engine ./groot_n1d6_onnx/dit_int8.trt \
  --precision int8 \
  --model-type dit \
  --calib-data ./groot_n1d6_onnx/calib_data/dit
```

#### 1.4 Benchmark

```bash
python3 scripts/deployment/benchmark_quantization.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --video_backend torchcodec \
  --trt_dit_path ./groot_n1d6_onnx/dit_fp16.trt \
  --traj_ids 0 1 2 3 4 \
  --num_latency_iterations 30 \
  --latency_warmup 10
```

### Phase 2: Backbone TRT (Not Recommended)

The backbone ONNX export works correctly (validated with onnxruntime), but
TRT compilation is not viable:

| Config | MSE vs ref | Cos Sim | Latency | Verdict |
|--------|-----------|---------|---------|---------|
| PyTorch BF16 (flash attn) | baseline | baseline | 157ms | **Recommended** |
| ORT FP32 (ONNX, eager) | 3.764 | 0.931 | n/a | Trace correct |
| TRT FP32 (eager) | 3.749 | 0.931 | 343ms | 2.2x slower |
| TRT FP16 (eager) | 25.639 | 0.302 | ~124ms | Broken |

**Root causes:**
- Flash attention → eager swap (required for ONNX) introduces MSE=3.76
- FP16 precision catastrophically fails: output range compressed from [-342, 352]
  to [-101, 76] due to intermediate value overflow in Siglip2 vision encoder
- FP32 TRT is accurate but slower than PyTorch because it can't use flash attention

If you still want to experiment with backbone ONNX export:

```bash
python3 scripts/deployment/export_backbone_onnx.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --output_dir ./groot_n1d6_onnx \
  --video_backend torchcodec

# Validate with onnxruntime (should match PyTorch eager closely)
python3 scripts/deployment/validate_backbone_onnx.py \
  --model_path alfie-gr00t/checkpoint-10000 \
  --dataset_path alfiebot.CanDoChallenge \
  --embodiment_tag new_embodiment \
  --onnx_path ./groot_n1d6_onnx/backbone_model.onnx

# Build TRT (FP32 only — FP16 produces garbage)
python3 scripts/deployment/build_tensorrt_engine.py \
  --onnx ./groot_n1d6_onnx/backbone_model.onnx \
  --engine ./groot_n1d6_onnx/backbone_fp32.trt \
  --precision fp32 \
  --model-type backbone
```

---

## Troubleshooting

### ONNX export fails with "mat1 and mat2 must have same dtype"
Use `--use_fp32` flag to convert model to FP32 before export.

### TRT build fails with "CUDA initialization failure error 35"
Use the Docker container (TRT 10.3 matched to CUDA 12.6) instead of pip TRT.

### Flash attention ONNX export error
`export_backbone_onnx.py` automatically swaps `flash_attention_2` to `eager`.
Check logs for "Swapped N flash_attention_2 → eager".

### DiT TRT output NaN or terrible E2E MSE
Check that the TRT engine IO dtypes match the wrapper. Engines built from BF16
ONNX expect BF16 inputs; engines from FP32 ONNX expect FP32 inputs. Use
`--use_fp32` during ONNX export to ensure FP32 IO which the wrapper handles.

### Per-denoising-step error compounds exponentially
Verify FP32 accumulation is active:
- `gr00t_n1d6.py`: `dtype=torch.float32` in `torch.randn()` and `.float()` in Euler step
- `standalone_inference_script.py`: `dtype=torch.float32` for TRT output buffer

---

## File Reference

| File | Purpose |
|------|---------|
| `export_onnx_n1d6.py` | Export DiT to ONNX (FP32 or BF16) |
| `export_backbone_onnx.py` | Export EagleX backbone to ONNX (handles flash attn swap) |
| `build_tensorrt_engine.py` | Build TRT engine from ONNX (FP32/FP16/BF16/INT8) |
| `calibrate_int8.py` | Collect INT8 calibration data + TRT calibrator |
| `benchmark_quantization.py` | Per-component accuracy + latency benchmarking |
| `validate_backbone_onnx.py` | Validate backbone ONNX accuracy (ORT vs PyTorch) |
| `standalone_inference_script.py` | E2E inference with PyTorch / TRT modes |
| `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | Denoising loop (FP32 accumulation fix) |
