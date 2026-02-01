# Worklog

## 2026-02-01: Wire up dead CLI args for inference latency reduction

**Problem:** Per-step inference latency was ~411ms on Orin NX 16GB with INT8 TRT. User specified `--denoising_steps 2` and `--action-horizon 4` but these CLI args were never wired to the model internals. The model config had `num_inference_timesteps=4` and `action_horizon=50`, so all 4 denoising steps ran with 50 action tokens regardless of CLI args.

**Changes:**
1. **Wired `--denoising_steps` to override `policy.model.action_head.num_inference_timesteps`** (standalone_inference_script.py) — After policy load, overrides the model's denoising step count with the CLI arg value. Reduces DiT forward passes from 4 to 2 (or 1).
2. **Cannot override model `action_horizon` at runtime** — The decode pipeline (delta_indices, norm params) is tightly coupled to the training action_horizon. Overriding causes shape mismatches in unnormalization. The CLI `--action-horizon` only controls eval loop stride (how many of the predicted actions to use). Added informational log when they differ.
3. **Event-based sync for backbone TRT** (standalone_inference_script.py) — Replaced blocking `stream.synchronize()` with event-based sync (matching DiT wrapper pattern), allowing CPU to proceed while GPU finishes.
4. **Guarded redundant dtype conversion** (gr00t_n1d6.py) — Added dtype check before `.to(self.dtype)` in denoising loop to skip no-op conversions.
5. **Added denoising steps to startup log** — Now logs the requested denoising steps value alongside other config.

## 2026-01-30: Episode memory reporting and inference memory reduction

**Problem:** TensorRT inference on Orin NX 16GB uses ~10GB, need to get under 8GB. Investigated where memory was going — the loaded episode DataFrame holds all video frames as PIL Images in memory for the entire inference run, even though only one step is accessed at a time and video data is never needed after inference.

**Changes (standalone_inference_script.py):**
1. **Added `measure_episode_memory()` helper** — When `--get-performance-stats` is enabled, measures and logs memory breakdown of each loaded episode (video vs state/action vs other). This identifies how much of the ~10GB is episode data vs model/TRT.
2. **Removed dead `obs` array construction** — After inference, the script was building an `obs` array from `parsed_obs` (lines 779-793) that was returned but never used by any caller. Removed entirely.
3. **Drop video columns after inference loop** — After all inference steps complete, video columns are dropped from the trajectory DataFrame and `gc.collect()` is called. `evaluate_predictions` only needs state/action columns. Frees ~500MB-1.5GB of PIL Images.
4. **Added gc.collect() between trajectories** — Explicitly frees trajectory data after each evaluation before loading the next episode.
5. **Episode memory summary in stats output** — The `--get-performance-stats` timing summary now includes per-trajectory memory breakdown and averages.

## 2026-01-31: Unified memory optimizations for Orin NX 16GB

**Problem:** Investigated whether the inference pipeline was making efficient use of Jetson's unified memory architecture (CPU and GPU share same 16GB DRAM). Found several areas where unnecessary memory copies and allocations wasted unified memory.

**Changes (standalone_inference_script.py):**
1. **Removed redundant `.to('cuda')` in TRT wrappers** — On unified memory, tensors are already GPU-accessible. The DiT wrapper was doing 5 device copies × 50 diffusion steps = ~250 unnecessary allocations per inference. Now only dtype casting is performed (no device movement), with `.contiguous()` guarded by `is_contiguous()` check.
2. **Pre-allocated reusable TRT output buffers** — DiT and backbone wrappers now cache output tensors and only reallocate on shape change, eliminating alloc/free churn during the diffusion loop.
3. **Configured `PYTORCH_CUDA_ALLOC_CONF`** — Set `expandable_segments:True` (reduces fragmentation) and `garbage_collection_threshold:0.6` (reclaims unused cache sooner than default 0.8).
4. **Capped PyTorch CUDA memory at 60%** — `set_per_process_memory_fraction(0.6)` prevents PyTorch's caching allocator from starving TensorRT/numpy/OS on the shared 16GB.
5. **Gated backbone debug logging to DEBUG level** — Per-call `logging.info` with f-string formatting was creating unnecessary overhead and temporary string allocations in the hot path.

## 2026-01-30: Fix backbone TRT OOM on Orin NX 16GB

**Problem:** Running full TRT inference (DiT + backbone) on Orin NX 16GB hit CUDA OOM when loading the backbone TRT engine (~3.08 GB). The loading order was: DiT TRT -> PyTorch model -> backbone TRT. By the time the backbone loaded, GPU memory was too fragmented/consumed.

**Root cause:** On Jetson unified memory, the `f.read()` of the 3.1 GB engine file creates a CPU buffer that competes with GPU allocations (since they share the same 16 GB). Additionally, loading the PyTorch model between the two TRT engines left less contiguous memory available.

**Fix (standalone_inference_script.py):**
1. Restructured loading order: backbone TRT (largest, ~3 GB) loads first while GPU is empty, then DiT TRT (~2.2 GB), then PyTorch model (~0.6 GB)
2. In both `TensorRTDiTWrapper` and `TensorRTBackboneWrapper`, split `f.read()` + `deserialize_cuda_engine()` into separate steps with explicit `del engine_data; gc.collect()` to free the file buffer before proceeding
3. Added OOM troubleshooting note to README_INT8.md

## 2026-01-31: Debug Cask/Myelin runtime error in backbone TRT (on-device build)

**Problem:** Backbone INT8 TRT engine was rebuilt on-device (Orin NX 16GB) without `--low-memory-mode`, but still fails at runtime with `Cask unknown internal error` from Myelin during `enqueueV3`.

**Action:** Added shape debug logging and profile validation to `TensorRTBackboneWrapper.__call__` in `standalone_inference_script.py`. On next run, the logs will print the exact input shapes/dtypes being sent to the engine, and flag any dimension that falls outside the engine's min/max optimization profile. This will identify whether the failure is a shape mismatch (e.g. pixel_values reshaping) or an INT8 calibration issue.

## 2026-01-31: Backbone TRT Cask/Myelin execution error on Orin NX

**Problem:** After fixing OOM, backbone TRT engine loaded successfully but failed at runtime with `Cask unknown internal error` from Myelin during `enqueueV3`. TRT also warned: "Using an engine plan file across different models of devices is not recommended."

**Root cause:** The backbone INT8 TRT engine was built on the Orin AGX 64GB (2048 CUDA cores) but executed on the Orin NX 16GB (1024 CUDA cores). TensorRT engines contain device-specific compiled tactics/fused kernels that are not portable across different GPU SKUs.

**Fix:** Rebuild the backbone (and DiT) TRT engines directly on the Orin NX using `--low-memory-mode`:
```
python scripts/deployment/build_tensorrt_engine.py \
    --onnx ./groot_n1d6_onnx/backbone_model.onnx \
    --engine ./groot_n1d6_onnx/backbone_int8_orin.trt \
    --precision int8 \
    --calib-data ./calibration_data_backbone/calib_data.npz \
    --calib-cache ./calibration_data_backbone/calibration.cache \
    --max-seq-len 512 \
    --low-memory-mode --prepare-system
```

## 2026-01-30: Research TensorRT cross-device engine portability (Orin AGX -> Orin NX)

**Problem:** Investigated whether TensorRT engines can be built on Orin AGX 64GB and deployed to Orin NX 16GB to avoid slow on-device builds.

**Findings:**
1. `HardwareCompatibilityLevel` (kAMPERE_PLUS, kSAME_COMPUTE_CAPABILITY) is **not supported on JetPack** per NVIDIA docs
2. Both devices share SM 8.7 but have different CUDA core counts (2048 vs 1024), so TRT selects different tactics/kernels
3. NVIDIA staff confirmed on forums: "TensorRT engine is not portable so it's required to be compiled on the exact target"
4. Workaround: Use Orin devkit emulation mode to emulate target module during build

**Action:** Must continue building engines on-device (Orin NX) or use devkit emulation. No cross-device shortcut available on JetPack.
