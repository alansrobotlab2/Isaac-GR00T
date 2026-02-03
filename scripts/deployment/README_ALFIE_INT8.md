# INT8 Quantization for GR00T N1.6 DiT

TensorRT INT8 post-training quantization (PTQ) for the DiT diffusion transformer, using entropy calibration with FP16 fallback.

## Data Preparation

### Convert episodes to lerobot+ format
```bash
python3 rosbag_to_groot.py \
    --demos-dir /home/alansrobotlab/Projects/alfiebot_ws/data/demonstrations \
    --output-dir /home/alansrobotlab/Projects/alfiebot_ws/data/alfiebot.CanDoChallenge \
    --task-index 0 \
    --start-episode 0 \
    --num-threads 32
```

### Review Videos and delete demonstration folders that aren't suitable

### Run script again to update structure with remaining episodes
```bash
python3 rosbag_to_groot.py \
    --demos-dir /home/alansrobotlab/Projects/alfiebot_ws/data/demonstrations \
    --output-dir /home/alansrobotlab/Projects/alfiebot_ws/data/alfiebot.CanDoChallenge \
    --task-index 0 \
    --start-episode 0 \
    --num-threads 32
```

### Edit episodes.jsonl for correct task indexes
```bash
nano ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge/meta/episodes.jsonl
```

### Update parquet files with episode number details from episodes.jsonl
```bash
python3 update_task_index.py --data-dir ~/Projects/alfiebot_ws/data/alfiebot.CanDoChallenge
```

### generate stats file
```bash
uv run gr00t/data/stats.py \
  --dataset-path ~/Projects/alfiebot_ws/data/alfiebot.CanDoChallenge \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ~/Projects/alfiebot_ws/data/alfiebot.CanDoChallenge/alfiebot_config.py
```


## Fine tune
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 uv run \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge/alfiebot_config.py \
    --num-gpus 1 \
    --output-dir ./cando \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 2000 \
    --global-batch-size 16 \
    --gradient-accumulation-steps 32 \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

## Quick Start (All-in-One)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/build_int8_pipeline.py \
    --model-path cando/checkpoint-2000 \
    --dataset-path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment-tag NEW_EMBODIMENT
```

This runs all 4 steps automatically, skipping any that already have outputs.

## Step-by-Step

### 1. Export ONNX (if not already done)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/export_onnx_n1d6.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment_tag NEW_EMBODIMENT \
    --output_dir ./groot_n1d6_onnx
```

### 2. Collect Calibration Data

Runs inference on dataset samples and captures the DiT or backbone input tensors:

**For DiT (action head):**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/collect_calibration_data.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./calibration_data \
    --num_samples 500
```

**For Backbone (Eagle vision encoder + LLM):**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/collect_calibration_data.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./calibration_data_backbone \
    --capture_target backbone \
    --attn_implementation eager \
    --num_samples 500
```

Outputs a single `calib_data.npz` file with all tensors and a `metadata.json` file.

### 3. Build INT8 TensorRT Engine

**For DiT:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/dit_model.onnx \
    --engine ./groot_n1d6_onnx/dit_model_int8.trt \
    --precision int8 \
    --calib-data ./calibration_data/calib_data.npz \
    --calib-cache ./calibration_data/calibration.cache
```

**For Backbone:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/backbone_model.onnx \
    --engine ./groot_n1d6_onnx/backbone_int8_orin.trt \
    --precision int8 \
    --calib-data ./calibration_data_backbone/calib_data.npz \
    --calib-cache ./calibration_data_backbone/calibration.cache \
    --max-seq-len 512
```

**Note:** The backbone must be exported with FP16 (not BF16) for TensorRT compatibility:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/export_backbone_onnx.py \
    --model_path cando/checkpoint-2000 \
    --dataset_path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
    --embodiment_tag new_embodiment \
    --output_dir ./groot_n1d6_onnx \
    --attn_implementation eager \
    --export_dtype fp16
```

The calibrator uses `IInt8EntropyCalibrator2` and caches results in the specified cache file. Subsequent builds with the same ONNX model reuse the cache. The script automatically detects whether you're building a DiT or backbone model based on the calibration data format.

### 4. Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/deployment/standalone_inference_script.py \
    --model-path cando/checkpoint-2000 \
    --dataset-path ~/Projects/alfiebot_ws/alfiebot.CanDoChallenge \
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

### 5. Verify ONNX Export (Optional)

After exporting the backbone ONNX model, verify it produces the same outputs as the
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

#### Expected results (FP16 export, `--device cpu`)

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

**Calibration cache**: Delete `calibration_data/int8_calib.cache` to force recalibration. The cache is tied to the specific ONNX model — if you retrain and re-export, delete the cache.