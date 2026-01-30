# INT8 Quantization for GR00T N1.6 DiT

TensorRT INT8 post-training quantization (PTQ) for the DiT diffusion transformer, using entropy calibration with FP16 fallback.

## Prerequisites

- Exported ONNX model at `groot_n1d6_onnx/dit_model.onnx` (see `export_onnx_n1d6.py`)
- A dataset for calibration (the same dataset used for training/inference)
- TensorRT with INT8 support (RTX 5090 / SM100+)

## Quick Start (All-in-One)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/build_int8_pipeline.py \
    --model-path cando/checkpoint-2000 \
    --dataset-path alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT
```

This runs all 4 steps automatically, skipping any that already have outputs.

## Step-by-Step

### 1. Export ONNX (if not already done)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/export_onnx_n1d6.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag NEW_EMBODIMENT \
    --output_dir ./groot_n1d6_onnx
```

### 2. Collect Calibration Data

Runs inference on dataset samples and captures the DiT input tensors:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/collect_calibration_data.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag NEW_EMBODIMENT \
    --output_dir ./calibration_data \
    --num_samples 500
```

Outputs `.npy` files in `calibration_data/`: `sa_embs.npy`, `vl_embs.npy`, `timesteps.npy`, plus optional mask files.

### 3. Build INT8 TensorRT Engine

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/dit_model.onnx \
    --engine ./groot_n1d6_onnx/dit_model_int8.trt \
    --precision int8 \
    --calib-dir ./calibration_data
```

The calibrator uses `IInt8EntropyCalibrator2` and caches results in `calibration_data/int8_calib.cache`. Subsequent builds with the same ONNX model reuse the cache.

### 4. Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/standalone_inference_script.py \
    --model-path cando/checkpoint-2000 \
    --dataset-path alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT \
    --traj-ids 0 1 2 \
    --inference-mode tensorrt \
    --trt-engine-path ./groot_n1d6_onnx/dit_model_int8.trt
```

## How It Works

1. **Calibration**: The `collect_calibration_data.py` script hooks into the DiT's forward pass during normal inference, capturing real input distributions across multiple trajectories and diffusion timesteps.

2. **Quantization**: TensorRT's `IInt8EntropyCalibrator2` computes optimal per-tensor scale factors by minimizing KL-divergence between the FP32 and INT8 activation distributions.

3. **Mixed precision**: The engine uses INT8 for quantizable operations (MatMul, Conv) with automatic FP16 fallback for operations that don't support INT8 (LayerNorm, SiLU, etc.).

4. **Dynamic shapes**: The engine supports variable sequence lengths (1-256 for sa_embs, 1-512 for vl_embs) with optimal shapes matching typical inference.

## Expected Performance

On RTX 5090:

| Precision | DiT Latency | Notes |
|-----------|------------|-------|
| PyTorch (BF16) | ~38 ms | torch.compile max-autotune |
| TensorRT BF16 | ~11 ms | 3.5x over PyTorch |
| TensorRT INT8 | ~6-8 ms | ~1.5-2x over TRT BF16 |

Actual speedup depends on whether TensorRT can fuse INT8 operations in the attention/FFN layers.

## Troubleshooting

**Calibration data collection is slow**: Reduce `--num_samples` to 100-200. More samples improve accuracy of scale factors but 100 is often sufficient.

**Engine build fails**: Ensure the ONNX model was exported correctly. Try rebuilding with `--force`. Check that TensorRT version supports INT8 on your GPU.

**Action quality degraded**: Compare INT8 vs BF16 outputs using the pipeline's validation step. If MSE is too high, you may need more calibration samples or should evaluate whether specific layers need FP16 precision (advanced: use TensorRT's layer-level precision API).

**Calibration cache**: Delete `calibration_data/int8_calib.cache` to force recalibration. The cache is tied to the specific ONNX model — if you retrain and re-export, delete the cache.
