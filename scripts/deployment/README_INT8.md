# INT8 Quantization for GR00T N1.6 Backbone and DiT

TensorRT INT8 post-training quantization (PTQ) for the DiT diffusion transformer, using entropy calibration with FP16 fallback.

## Prerequisites

- RTX 5090 for model fine tuning
- Orin AGX 64GB Devkit with Jetpack 6.2.1 for TensorRT engine compilation
    - gr00t n1.6 docker container for runtime
- Orin NX 16GB with Jetpack 6.2.1 for inference (alfie)
    - gr00t n1.6 docker container for runtime

## Quick Start (All-in-One)

```bash
python scripts/deployment/build_int8_pipeline.py \
    --model-path cando/checkpoint-2000 \
    --dataset-path alfiebot.CanDoChallenge \
#   --refresh
```

This runs all 6 steps automatically, skipping any that already have outputs.

## Step-by-Step

### 1. Export Backbone to ONNX (Eagle vision encoder + LLM)

```bash
python scripts/deployment/export_backbone_onnx.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./groot_n1d6_onnx \
    --attn_implementation eager \
    --export_dtype fp16
```

Outputs `backbone_model.onnx` (+ `backbone_model.onnx.data` external weights) in the `--output_dir`.

### 2. Export DiT to ONNX

```bash
python scripts/deployment/export_onnx_n1d6.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./groot_n1d6_onnx \
    --video_backend torchcodec
```

Outputs `dit_model.onnx` in the `--output_dir`. Uses BF16 precision by default.


### 3. Collect Backbone Calibration Data

```bash
python scripts/deployment/collect_calibration_data.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./calibration_data_backbone \
    --capture_target backbone \
    --attn_implementation eager \
    --num_samples 500
```

Outputs a single `calib_data.npz` file with all tensors and a `metadata.json` file.

### 4. Collect DiT Calibration Data

```bash
python scripts/deployment/collect_calibration_data.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./calibration_data_dit \
    --capture_target dit \
    --num_samples 500 \
    --video_backend torchcodec
```

Outputs a single `calib_data.npz` file with all tensors (`sa_embs`, `vl_embs`, `timestep`, `image_mask`, `backbone_attention_mask`) and a `metadata.json` file.

### 5. Build INT8 Backbone TensorRT Engine

```bash
python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/backbone_model.onnx \
    --engine ./groot_n1d6_onnx/backbone_int8_orin.trt \
    --precision int8 \
    --calib-data ./calibration_data_backbone/calib_data.npz \
    --calib-cache ./calibration_data_backbone/calibration.cache \
    --max-seq-len 512
```

### 6. Build INT8 DiT TensorRT Engine

```bash
python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/dit_model.onnx \
    --engine ./groot_n1d6_onnx/dit_model_int8_orin.trt \
    --precision int8 \
    --calib-data ./calibration_data_dit/calib_data.npz \
    --calib-cache ./calibration_data_dit/calibration.cache \
    --max-seq-len 512
```

The calibrator uses `IInt8EntropyCalibrator2` and caches results in the specified cache file. Subsequent builds with the same ONNX model reuse the cache. The script automatically detects whether you're building a DiT or backbone model based on the calibration data format.

### 7. Run Inference

```bash
python scripts/deployment/standalone_inference_script.py \
--model-path cando/checkpoint-2000 \
--dataset-path alfiebot.CanDoChallenge \
--embodiment-tag NEW_EMBODIMENT \
--traj-ids 0 1 \
--inference-mode tensorrt \
--trt-engine-path ./groot_n1d6_onnx/dit_model_int8_orin.trt \
--backbone-trt-engine-path ./groot_n1d6_onnx/backbone_int8_orin.trt \
--attn-implementation eager \
--action-horizon 4 \
--denoising_steps 2 \
2>&1 | tee inference_output_full_trt.txt
```

## How It Works

1. **Calibration**: The `collect_calibration_data.py` script hooks into the DiT's forward pass during normal inference, capturing real input distributions across multiple trajectories and diffusion timesteps.

2. **Quantization**: TensorRT's `IInt8EntropyCalibrator2` computes optimal per-tensor scale factors by minimizing KL-divergence between the FP32 and INT8 activation distributions.

3. **Mixed precision**: The engine uses INT8 for quantizable operations (MatMul, Conv) with automatic FP16 fallback for operations that don't support INT8 (LayerNorm, SiLU, etc.).

4. **Dynamic shapes**: The engine supports variable sequence lengths (1-256 for sa_embs, 1-512 for vl_embs) with optimal shapes matching typical inference.

## Expected Performance

On Orin NX 16GB:

| Precision | DiT Latency | Memory | Notes |
|-----------|-------------|--------|-------|
| PyTorch (BF16) | ~550 ms | 12.5GB | torch.compile max-autotune |
| TensorRT BF16 | ~550 ms | 12.5GB | similar to PyTorch |
| TensorRT INT8 | ~250 ms | 8GB | ~1.5-2x faster than TRT BF16 |

Actual speedup depends on whether TensorRT can fuse INT8 operations in the attention/FFN layers.

## Verify ONNX Export (Optional)

After exporting the backbone ONNX model (step 1), verify it produces the same outputs as the
PyTorch model. This catches graph conversion errors before building TensorRT engines.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/verify_backbone_onnx.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --onnx_path ./groot_n1d6_onnx/backbone_model.onnx \
    --device cpu
```

The script runs the same input through both the PyTorch backbone and the ONNX model,
then compares hidden states, attention masks, and image masks.

**`--device cpu` is recommended** because it forces both backends onto the same device,
isolating ONNX graph correctness from CPU-vs-GPU floating point differences. Without it,
FP16 numerical divergence between devices can produce misleadingly large errors.

Pass/fail criteria:
- **Cosine similarity** >= 0.999 (directional agreement)
- **p99 absolute error** < threshold (99% of elements are close; a few FP16 outliers are expected)
- **No NaN** values in ONNX output
- **Exact match** on boolean masks (attention mask, image mask)

### Expected results (FP16 export, `--device cpu`)

With `--device cpu`, PyTorch runs in FP32 and ONNX runs with FP16 weights, so all
differences come from FP16 weight quantization and graph conversion — not device numerics.

| Metric | Typical Value | Notes |
|--------|--------------|-------|
| Cosine similarity | 0.9997 | Well above 0.999 threshold |
| Mean absolute error | ~0.007 | Consistent with FP16 quantization noise |
| p99 absolute error | ~0.047 | 99% of elements within 0.05 |
| Max absolute error | ~66 | Single FP16 outlier in vision encoder (see below) |
| Text tokens mean error | ~0.001 | Text path is very tight |
| Image tokens mean error | ~0.008 | Slightly higher due to vision encoder complexity |
| Boolean masks | Exact match | attention_mask and image_mask are identical |

**About the max error outlier:** The max absolute error (~66) is concentrated in a single
image token position where FP16 dynamic range is insufficient for an intermediate value
in the vision encoder (likely an attention softmax edge case). This affects <0.1% of
elements and does not impact downstream action prediction — the DiT action head aggregates
over all 411 tokens, diluting the outlier.

**Without `--device cpu`** (ONNX on CPU, PyTorch on CUDA): expect cosine ~0.998 and max
error ~140 due to CPU-vs-GPU floating point divergence compounding through 24 transformer
layers. This does not indicate a graph conversion problem.

Additional options:
- `--atol 0.5` — p99 absolute error tolerance (default: 0.5)
- `--cosine_thresh 0.999` — minimum cosine similarity (default: 0.999)
- `--export_dtype fp32` — use FP32 for tighter comparison (must match what was used during export)

## Troubleshooting

**Calibration data collection is slow**: Reduce `--num_samples` to 100-200. More samples improve accuracy of scale factors but 100 is often sufficient.

**Engine build fails**: Ensure the ONNX model was exported correctly. Try rebuilding with `--force`. Check that TensorRT version supports INT8 on your GPU.

**Action quality degraded**: Compare INT8 vs BF16 outputs using the pipeline's validation step. If MSE is too high, you may need more calibration samples or should evaluate whether specific layers need FP16 precision (advanced: use TensorRT's layer-level precision API).

**Backbone TRT OOM on Orin NX 16GB**: The backbone INT8 engine needs ~3GB of contiguous GPU memory. On Jetson unified memory, the loading order matters — all TRT engines are loaded first (backbone, then DiT) while GPU is empty, before loading the PyTorch model. If you still hit OOM, check system memory usage with `tegrastats` and ensure no other processes are consuming GPU memory.

**Calibration cache**: Delete `calibration_data/int8_calib.cache` to force recalibration. The cache is tied to the specific ONNX model — if you retrain and re-export, delete the cache.
