#!/usr/bin/env python3
"""
Build TensorRT Engine from ONNX Model

This script builds a TensorRT engine from the exported ONNX DiT model.
It's equivalent to using trtexec but works with pip-installed TensorRT.

Usage:
    python build_tensorrt_engine.py \
        --onnx ./groot_n1d6_onnx/dit_model.onnx \
        --engine ./groot_n1d6_onnx/dit_model_bf16.trt \
        --precision bf16

    # For INT8 with calibration:
    python build_tensorrt_engine.py \
        --onnx ./groot_n1d6_onnx/dit_model.onnx \
        --engine ./groot_n1d6_onnx/dit_model_int8.trt \
        --precision int8 \
        --calib-data ./groot_n1d6_onnx/calib_data.npz \
        --calib-cache ./groot_n1d6_onnx/int8_calib.cache

    # For Orin NX 16GB (low memory mode):
    python build_tensorrt_engine.py \
        --onnx ./groot_n1d6_onnx/dit_model.onnx \
        --engine ./groot_n1d6_onnx/dit_model_int8.trt \
        --precision int8 \
        --calib-data ./groot_n1d6_onnx/calib_data.npz \
        --low-memory-mode
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time

# Memory optimization environment variables - must be set BEFORE importing TensorRT
# These help reduce GPU memory during Myelin subgraph compilation
if os.environ.get('TRT_LOW_MEMORY_MODE') == '1' or '--low-memory-mode' in sys.argv:
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')  # Lazy load CUDA modules
    os.environ.setdefault('TRT_SINGLE_THREADED_BUILD', '1')  # Single-threaded reduces peak memory

import numpy as np
import tensorrt as trt

# cuda-python compatibility: >=12.6 uses cuda.bindings, <12.6 uses cuda.cuda/cuda.cudart
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    import cuda.cudart as cudart


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor GPU/CPU memory usage during TensorRT build on Jetson."""

    def __init__(self, interval_seconds: float = 5.0):
        self.running = False
        self.interval = interval_seconds
        self.peak_gpu_mb = 0
        self.peak_cpu_mb = 0
        self.thread = None

    def start(self):
        """Start background memory monitoring."""
        self.running = True
        self.peak_gpu_mb = 0
        self.peak_cpu_mb = 0
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Memory monitor started (interval: {self.interval}s)")

    def stop(self):
        """Stop monitoring and return peak memory usage."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.peak_gpu_mb, self.peak_cpu_mb

    def _get_memory_usage(self):
        """Get current memory usage on Jetson (unified memory architecture)."""
        gpu_mb = 0
        cpu_mb = 0

        # Try tegrastats first (Jetson-specific)
        try:
            # Use /sys/kernel/debug/nvmap for GPU memory on Jetson
            nvmap_path = '/sys/kernel/debug/nvmap/iovmm/clients'
            if os.path.exists(nvmap_path):
                result = subprocess.run(
                    ['sudo', 'cat', nvmap_path],
                    capture_output=True, text=True, timeout=1.0
                )
                # Parse total GPU memory from nvmap
                for line in result.stdout.split('\n'):
                    if 'total' in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                gpu_mb = int(part) // (1024 * 1024)
                                break
        except Exception:
            pass

        # Fallback: use /proc/meminfo for total memory pressure
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        available_kb = int(line.split()[1])
                    elif line.startswith('MemTotal:'):
                        total_kb = int(line.split()[1])
                cpu_mb = (total_kb - available_kb) // 1024
        except Exception:
            pass

        return gpu_mb, cpu_mb

    def _monitor_loop(self):
        """Background thread that monitors memory."""
        while self.running:
            gpu_mb, cpu_mb = self._get_memory_usage()
            if gpu_mb > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu_mb
            if cpu_mb > self.peak_cpu_mb:
                self.peak_cpu_mb = cpu_mb
                logger.debug(f"Memory: CPU used ~{cpu_mb}MB (peak: {self.peak_cpu_mb}MB)")
            time.sleep(self.interval)


def prepare_system_for_build():
    """
    Prepare system for memory-intensive TensorRT build.
    Drops caches and increases swappiness to free physical RAM for GPU.
    """
    logger.info("Preparing system for low-memory build...")

    # Drop filesystem caches to free physical RAM
    try:
        subprocess.run(['sudo', 'sync'], check=False, timeout=30)
        subprocess.run(
            ['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
            check=False, timeout=10
        )
        logger.info("Dropped filesystem caches")
    except Exception as e:
        logger.warning(f"Could not drop caches (may need sudo): {e}")

    # Increase swappiness to push CPU-only allocations to swap
    try:
        subprocess.run(
            ['sudo', 'sysctl', '-w', 'vm.swappiness=100'],
            check=False, timeout=10
        )
        logger.info("Set vm.swappiness=100")
    except Exception as e:
        logger.warning(f"Could not set swappiness: {e}")

    # Allow memory overcommit
    try:
        subprocess.run(
            ['sudo', 'sysctl', '-w', 'vm.overcommit_memory=1'],
            check=False, timeout=10
        )
        logger.info("Set vm.overcommit_memory=1")
    except Exception as e:
        logger.warning(f"Could not set overcommit: {e}")

    # Report available memory
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available_mb = int(line.split()[1]) // 1024
                    logger.info(f"Available memory after preparation: {available_mb}MB")
                    break
    except Exception:
        pass


def warn_if_low_memory(workspace_mb: int):
    """Warn if system memory is low for Jetson shared memory architecture."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available_mb = int(line.split()[1]) // 1024
                    if workspace_mb > available_mb * 0.4:
                        logger.warning(
                            f"Workspace ({workspace_mb}MB) may exceed safe limit for shared memory. "
                            f"Available system memory: {available_mb}MB. "
                            f"Consider using --workspace {max(256, available_mb // 4)}"
                        )
                    else:
                        logger.info(f"System memory available: {available_mb}MB")
                    break
    except Exception:
        pass  # Not critical if we can't check


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Calibrator for TensorRT DiT model using pre-captured calibration data.

    The calibration data should be a .npz file containing arrays for each input:
    - sa_embs: (N, seq_len, 1536) float16/float32
    - vl_embs: (N, seq_len, 2048) float16/float32
    - timestep: (N,) int64
    - image_mask: (N, seq_len) bool
    - backbone_attention_mask: (N, seq_len) bool

    Where N is the number of calibration samples (typically 100-500).
    """

    def __init__(self, calib_data_path: str, cache_file: str, batch_size: int = 1):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Load calibration data
        logger.info(f"Loading calibration data from {calib_data_path}...")
        data = np.load(calib_data_path)

        self.calib_data = {
            "sa_embs": data["sa_embs"].astype(np.float32),
            "vl_embs": data["vl_embs"].astype(np.float32),
            "timestep": data["timestep"].astype(np.int64),
            "image_mask": data["image_mask"].astype(np.bool_),
            "backbone_attention_mask": data["backbone_attention_mask"].astype(np.bool_),
        }

        self.num_samples = self.calib_data["sa_embs"].shape[0]
        logger.info(f"Loaded {self.num_samples} calibration samples")

        # Allocate device memory for each input

        self.device_inputs = {}
        self.input_sizes = {}

        for name, arr in self.calib_data.items():
            # Size of one batch
            single_sample_size = int(np.prod(arr.shape[1:])) * arr.itemsize
            self.input_sizes[name] = single_sample_size

            # Allocate GPU memory
            err, ptr = cudart.cudaMalloc(single_sample_size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to allocate CUDA memory for {name}")
            self.device_inputs[name] = ptr

        logger.info("DiT calibrator initialized")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_samples:
            return None


        # Copy data for each input
        bindings = []
        for name in names:
            if name not in self.calib_data:
                logger.warning(f"Unknown input name: {name}")
                return None

            # Get the current sample
            data = self.calib_data[name][self.current_index:self.current_index + 1]
            data = np.ascontiguousarray(data)

            # Copy to device
            ptr = self.device_inputs[name]
            cudart.cudaMemcpy(ptr, data.ctypes.data, data.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            bindings.append(int(ptr))

        self.current_index += self.batch_size

        if self.current_index % 50 == 0:
            logger.info(f"Calibration progress: {self.current_index}/{self.num_samples}")

        return bindings

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            logger.info(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        logger.info(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def __del__(self):
        # Free device memory
        for ptr in self.device_inputs.values():
            cudart.cudaFree(ptr)


class BackboneInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Calibrator for TensorRT Eagle backbone using pre-captured calibration data.

    The calibration data should be a .npz file containing arrays for each input:
    - input_ids: (N, seq_len) int64
    - attention_mask: (N, seq_len) int64
    - pixel_values: (total_frames, C, H, W) float32
      where total_frames = N * frames_per_sample

    Where N is the number of calibration samples (typically 100-500).
    """

    def __init__(self, calib_data_path: str, cache_file: str, batch_size: int = 1):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Load calibration data
        logger.info(f"Loading backbone calibration data from {calib_data_path}...")
        data = np.load(calib_data_path)

        input_ids = data["input_ids"].astype(np.int64)
        attention_mask = data["attention_mask"].astype(np.int64)
        pixel_values = data["pixel_values"].astype(np.float32)

        self.num_samples = input_ids.shape[0]

        # Calculate frames per sample
        total_frames = pixel_values.shape[0]
        self.frames_per_sample = total_frames // self.num_samples

        logger.info(f"Loaded {self.num_samples} backbone calibration samples")
        logger.info(f"  input_ids shape: {input_ids.shape}")
        logger.info(f"  attention_mask shape: {attention_mask.shape}")
        logger.info(f"  pixel_values shape: {pixel_values.shape}")
        logger.info(f"  Frames per sample: {self.frames_per_sample}")

        # Reshape pixel_values to [N, frames_per_sample, C, H, W]
        self.calib_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values.reshape(
                self.num_samples, self.frames_per_sample, *pixel_values.shape[1:]
            ),
        }

        logger.info(f"  Reshaped pixel_values: {self.calib_data['pixel_values'].shape}")

        # Allocate device memory for each input

        self.device_inputs = {}
        self.input_sizes = {}

        for name, arr in self.calib_data.items():
            # Size of one batch
            single_sample_size = int(np.prod(arr.shape[1:])) * arr.itemsize
            self.input_sizes[name] = single_sample_size

            # Allocate GPU memory
            err, ptr = cudart.cudaMalloc(single_sample_size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to allocate CUDA memory for {name}")
            self.device_inputs[name] = ptr

        logger.info("Backbone calibrator initialized")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_samples:
            return None


        # Copy data for each input
        bindings = []
        for name in names:
            if name not in self.calib_data:
                logger.warning(f"Unknown input name: {name}")
                return None

            # Get the current sample
            data = self.calib_data[name][self.current_index:self.current_index + 1]
            data = np.ascontiguousarray(data)

            # Copy to device
            ptr = self.device_inputs[name]
            cudart.cudaMemcpy(ptr, data.ctypes.data, data.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            bindings.append(int(ptr))

        self.current_index += self.batch_size

        if self.current_index % 50 == 0:
            logger.info(f"Calibration progress: {self.current_index}/{self.num_samples}")

        return bindings

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            logger.info(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        logger.info(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def __del__(self):
        # Free device memory
        for ptr in self.device_inputs.values():
            cudart.cudaFree(ptr)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 2048,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
    calibrator: trt.IInt8Calibrator = None,
    sparse_weights: bool = True,
    tactic_memory_mb: int = 4096,
    timing_cache_path: str = None,
    builder_opt_level: int = 3,
    low_memory_mode: bool = False,
    disable_external_tactics: bool = False,
    heuristic_build: bool = False,
    memory_monitor: bool = False,
    tactic_shared_memory_mb: int = 256,
    refittable: bool = False,
    strip_plan: bool = False,
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'bf16', 'fp8', 'int8')
        workspace_mb: Workspace size in MB
        min_shapes: Minimum input shapes (dict: name -> shape tuple)
        opt_shapes: Optimal input shapes (dict: name -> shape tuple)
        max_shapes: Maximum input shapes (dict: name -> shape tuple)
        calibrator: INT8 calibrator (required for int8 precision)
        sparse_weights: Enable sparse weights for memory efficiency
        tactic_memory_mb: Memory limit for tactic selection in MB
        timing_cache_path: Path to save/load timing cache
        builder_opt_level: Builder optimization level (0-5)
        low_memory_mode: Enable all memory-saving options for Orin NX 16GB
        disable_external_tactics: Disable cuDNN/cuBLAS tactic sources
        heuristic_build: Use heuristic tactic selection (faster, less memory)
        memory_monitor: Enable real-time memory monitoring during build
        tactic_shared_memory_mb: Limit for TACTIC_SHARED_MEMORY pool
    """
    logger.info("=" * 80)
    logger.info("TensorRT Engine Builder (Orin Optimized)")
    logger.info("=" * 80)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Engine output: {engine_path}")
    logger.info(f"Precision: {precision.upper()}")
    logger.info(f"Workspace: {workspace_mb} MB")
    logger.info(f"Tactic DRAM limit: {tactic_memory_mb} MB")
    logger.info(f"Tactic shared memory limit: {tactic_shared_memory_mb} MB")
    logger.info(f"Sparse weights: {sparse_weights}")
    logger.info(f"Builder optimization level: {builder_opt_level}")
    logger.info(f"Timing cache: {timing_cache_path or 'disabled'}")
    logger.info(f"Low memory mode: {low_memory_mode}")
    logger.info(f"Disable external tactics: {disable_external_tactics}")
    logger.info(f"Heuristic build: {heuristic_build}")
    logger.info("=" * 80)

    # Check memory before starting (Jetson shared memory warning)
    warn_if_low_memory(workspace_mb)

    # Start memory monitor if requested
    monitor = None
    if memory_monitor:
        monitor = MemoryMonitor(interval_seconds=5.0)
        monitor.start()

    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # Create builder and network
    logger.info("\n[Step 1/5] Creating TensorRT builder...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("\n[Step 2/5] Parsing ONNX model...")
    if not parser.parse_from_file(onnx_path):
        logger.error("Failed to parse ONNX file")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("ONNX parsing failed")

    # Parser successful. Network is loaded
    logger.info(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"  Input {i}: {inp.name} {inp.shape}")

    logger.info(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"  Output {i}: {out.name} {out.shape}")

    # Create builder config
    logger.info("\n[Step 3/5] Configuring builder...")
    config = builder.create_builder_config()

    # Enable detailed profiling for engine inspection
    # This allows get_layer_information() to return layer types, precisions, tactics, etc.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    logger.info("Enabled DETAILED profiling verbosity for engine inspection")

    # Set workspace and memory limits for Orin shared memory
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))

    # Limit memory used during tactic selection (critical for Orin OOM prevention)
    # TACTIC_DRAM limits the scratch memory TensorRT can allocate when evaluating tactics
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, tactic_memory_mb * (1024**2))
        logger.info(f"Set TACTIC_DRAM limit to {tactic_memory_mb} MB")
    except Exception as e:
        logger.warning(f"TACTIC_DRAM not available: {e}")

    # Also try to limit TACTIC_SHARED_MEMORY if available
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, tactic_shared_memory_mb * (1024**2))
        logger.info(f"Set TACTIC_SHARED_MEMORY limit to {tactic_shared_memory_mb} MB")
    except Exception:
        pass  # Not all TRT versions have this

    # Disable external tactic sources to reduce memory during build
    # This prevents TensorRT from loading cuDNN, cuBLAS, cuBLASLt tactics
    if disable_external_tactics or low_memory_mode:
        try:
            # Set tactic sources to 0 (disable all external sources)
            # This can significantly reduce memory during Myelin subgraph compilation
            config.set_tactic_sources(0)
            logger.info("Disabled all external tactic sources (cuDNN, cuBLAS, cuBLASLt)")
        except Exception as e:
            logger.warning(f"Could not disable external tactic sources: {e}")

    # Use heuristic tactic selection (skips expensive benchmarking)
    if heuristic_build or low_memory_mode:
        try:
            config.set_flag(trt.BuilderFlag.HEURISTIC)
            logger.info("Enabled HEURISTIC mode for reduced build memory")
        except Exception as e:
            logger.warning(f"HEURISTIC flag not available: {e}")

    # Disable expensive builder optimizations that consume memory
    try:
        # PREFER_PRECISION_CONSTRAINTS can reduce memory by avoiding precision exploration
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        logger.info("Enabled PREFER_PRECISION_CONSTRAINTS")
    except Exception:
        pass

    # Builder optimization level: lower = less memory, faster build, potentially slower engine
    # Level 0: Fastest build, least memory, basic optimizations only
    # Level 3: Default balance
    # Level 5: Most aggressive optimization, most memory
    try:
        config.builder_optimization_level = builder_opt_level
        logger.info(f"Builder optimization level: {builder_opt_level}")
    except Exception as e:
        logger.warning(f"Could not set builder optimization level: {e}")

    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode")
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
        logger.info("Enabled BF16 mode")
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
        logger.info("Enabled FP8 mode")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for layers that don't support INT8
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # Reduce memory from unnecessary precision upgrades
        if calibrator is not None:
            config.int8_calibrator = calibrator
            logger.info("Enabled INT8 mode with calibration data and precision constraints")
        else:
            raise ValueError("INT8 precision requires calibration data. Use --calib-data to provide calibration samples.")
    elif precision == "fp32":
        logger.info("Using FP32 (default precision)")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Memory optimization flags for Orin (shared CPU/GPU memory)
    if sparse_weights:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        logger.info("Enabled SPARSE_WEIGHTS for memory efficiency")

    # Refittable engine - weights can be updated without rebuilding
    # This can reduce peak memory during build on memory-constrained devices
    if refittable:
        try:
            config.set_flag(trt.BuilderFlag.REFIT)
            logger.info("Enabled REFIT mode (refittable engine)")
        except Exception as e:
            logger.warning(f"REFIT flag not available: {e}")

    # Strip plan - remove unnecessary data from serialized engine
    if strip_plan or low_memory_mode:
        try:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
            logger.info("Enabled STRIP_PLAN to reduce serialized engine size")
        except Exception as e:
            logger.warning(f"STRIP_PLAN flag not available: {e}")

    # Timing cache reduces memory churn during tactic search
    timing_cache = None
    if timing_cache_path:
        cache_data = b""
        if os.path.exists(timing_cache_path):
            logger.info(f"Loading timing cache from {timing_cache_path}")
            with open(timing_cache_path, "rb") as f:
                cache_data = f.read()
        timing_cache = config.create_timing_cache(cache_data)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
        logger.info("Timing cache enabled")

    # Set optimization profiles for dynamic shapes
    if min_shapes and opt_shapes and max_shapes:
        logger.info("\n[Step 4/5] Setting optimization profiles...")
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            input_name = inp.name

            if input_name in min_shapes:
                min_shape = min_shapes[input_name]
                opt_shape = opt_shapes[input_name]
                max_shape = max_shapes[input_name]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"  {input_name}:")
                logger.info(f"    min: {min_shape}")
                logger.info(f"    opt: {opt_shape}")
                logger.info(f"    max: {max_shape}")

        config.add_optimization_profile(profile)
    else:
        raise RuntimeError("Provide min/max and opt shapes for dynamic axes")

    # Build engine
    logger.info("\n[Step 5/5] Building TensorRT engine...")

    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    logger.info(f"Engine built in {build_time:.1f} seconds ({build_time / 60:.1f} minutes)")

    # Save timing cache for future builds
    if timing_cache_path and timing_cache:
        logger.info(f"Saving timing cache to {timing_cache_path}")
        with open(timing_cache_path, "wb") as f:
            f.write(timing_cache.serialize())

    # Save engine
    logger.info(f"\nSaving engine to {engine_path}...")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024**2)
    logger.info(f"Engine saved! Size: {engine_size_mb:.2f} MB")

    # Stop memory monitor and report
    if monitor:
        peak_gpu, peak_cpu = monitor.stop()
        logger.info(f"Peak memory during build: GPU ~{peak_gpu}MB, CPU ~{peak_cpu}MB")

    logger.info("\n" + "=" * 80)
    logger.info("ENGINE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Engine file: {engine_path}")
    logger.info(f"Size: {engine_size_mb:.2f} MB")
    logger.info(f"Build time: {build_time:.1f}s")
    logger.info(f"Precision: {precision.upper()}")
    if monitor:
        logger.info(f"Peak CPU memory: ~{peak_cpu}MB")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--engine", type=str, required=True, help="Path to save TensorRT engine")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "fp8", "int8"],
        help="Precision mode (default: bf16). Note: int8 uses fp16 fallback without calibration.",
    )
    parser.add_argument(
        "--workspace", type=int, default=2048, help="Workspace size in MB (default: 2048 for Orin)"
    )
    parser.add_argument(
        "--calib-data", type=str, default=None,
        help="Path to calibration data (.npz file) for INT8 quantization. "
             "Generate with: python scripts/deployment/generate_calib_data.py"
    )
    parser.add_argument(
        "--calib-cache", type=str, default="./int8_calib.cache",
        help="Path to save/load INT8 calibration cache (default: ./int8_calib.cache)"
    )
    parser.add_argument(
        "--sparse-weights", action="store_true", default=True,
        help="Enable sparse weights for memory efficiency (default: True)"
    )
    parser.add_argument(
        "--timing-cache", type=str, default=None,
        help="Path to save/load timing cache (reduces memory during tactic search)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=256,
        help="Maximum sequence length for dynamic shapes (default: 256, reduce to 128 if OOM)"
    )
    parser.add_argument(
        "--tactic-memory", type=int, default=4096,
        help="Memory limit for tactic selection in MB (default: 4096)"
    )
    parser.add_argument(
        "--builder-opt-level", type=int, default=3, choices=[0, 1, 2, 3, 4, 5],
        help="Builder optimization level 0-5 (default: 3). Lower = less memory, faster build, potentially slower engine"
    )
    parser.add_argument(
        "--low-memory-mode", action="store_true", default=False,
        help="Enable all memory-saving options for Orin NX 16GB. Sets env vars, disables external tactics, enables heuristics."
    )
    parser.add_argument(
        "--disable-external-tactics", action="store_true", default=False,
        help="Disable cuDNN/cuBLAS tactic sources to reduce memory during Myelin compilation"
    )
    parser.add_argument(
        "--heuristic-build", action="store_true", default=False,
        help="Use heuristic tactic selection (faster build, less memory, may be slower inference)"
    )
    parser.add_argument(
        "--prepare-system", action="store_true", default=False,
        help="Prepare system for build: drop caches, increase swappiness (requires sudo)"
    )
    parser.add_argument(
        "--memory-monitor", action="store_true", default=False,
        help="Enable real-time memory monitoring during build"
    )
    parser.add_argument(
        "--tactic-shared-memory", type=int, default=256,
        help="Memory limit for TACTIC_SHARED_MEMORY pool in MB (default: 256, try 128 for OOM)"
    )
    parser.add_argument(
        "--refittable", action="store_true", default=False,
        help="Build a refittable engine (weights stored separately, may reduce build memory)"
    )
    parser.add_argument(
        "--strip-plan", action="store_true", default=False,
        help="Strip unnecessary data from the serialized plan to reduce memory"
    )

    args = parser.parse_args()

    # Apply low memory mode defaults
    if args.low_memory_mode:
        logger.info("Low memory mode enabled - applying aggressive memory-saving settings")
        # Override defaults for low memory mode
        if args.workspace == 2048:  # Only override if using default
            args.workspace = 256
            logger.info(f"  --workspace reduced to {args.workspace} MB")
        if args.tactic_memory == 4096:  # Only override if using default
            args.tactic_memory = 512
            logger.info(f"  --tactic-memory reduced to {args.tactic_memory} MB")
        if args.builder_opt_level == 3:  # Only override if using default
            args.builder_opt_level = 0
            logger.info(f"  --builder-opt-level reduced to {args.builder_opt_level}")
        if args.tactic_shared_memory == 256:  # Only override if using default
            args.tactic_shared_memory = 128
            logger.info(f"  --tactic-shared-memory reduced to {args.tactic_shared_memory} MB")
        # Always enable these in low memory mode
        args.disable_external_tactics = True
        args.heuristic_build = True
        args.memory_monitor = True
        logger.info("  Enabled: --disable-external-tactics, --heuristic-build, --memory-monitor")

    # Prepare system if requested
    if args.prepare_system:
        prepare_system_for_build()

    # Define shapes for your specific model (from export)
    min_shapes = None
    opt_shapes = None
    max_shapes = None

    # Auto-detect model type from calibration data if provided
    model_type = None
    if args.calib_data:
        logger.info(f"Detecting model type from calibration data: {args.calib_data}")
        calib_data = np.load(args.calib_data)
        calib_keys = set(calib_data.keys())

        if "input_ids" in calib_keys and "pixel_values" in calib_keys:
            model_type = "backbone"
            logger.info("Detected backbone model (keys: input_ids, attention_mask, pixel_values)")
        elif "sa_embs" in calib_keys and "vl_embs" in calib_keys:
            model_type = "dit"
            logger.info("Detected DiT model (keys: sa_embs, vl_embs, timestep, etc.)")
        else:
            raise ValueError(
                f"Could not detect model type from calibration data keys: {calib_keys}\n"
                "Expected either backbone (input_ids, pixel_values) or DiT (sa_embs, vl_embs)"
            )

    # Establish Dynamic Shapes to handle variable seq lengths
    # Based on captured inputs but with ranges to handle variations
    # Note: Large max shapes cause massive memory allocation during tactic search
    max_seq = args.max_seq_len
    logger.info(f"Using max sequence length: {max_seq} (use --max-seq-len to adjust)")

    if model_type == "backbone":
        # Backbone model shapes
        # Get actual shapes from calibration data
        input_ids_shape = calib_data["input_ids"].shape
        pixel_values_shape = calib_data["pixel_values"].shape

        opt_seq_len = min(input_ids_shape[1], max_seq)

        # pixel_values is [total_frames, C, H, W]
        # Need to determine num_frames per sample from metadata
        import json
        metadata_path = os.path.join(os.path.dirname(args.calib_data), "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            num_samples = metadata["num_samples"]
            total_frames = pixel_values_shape[0]
            num_frames = total_frames // num_samples
            logger.info(f"From metadata: {num_samples} samples, {total_frames} total frames = {num_frames} frames/sample")
        else:
            # Fallback: assume standard 4-view setup
            num_frames = 4
            logger.warning(f"No metadata found, assuming {num_frames} frames per sample")

        img_channels = pixel_values_shape[1]  # Should be 3
        img_height = pixel_values_shape[2]
        img_width = pixel_values_shape[3]

        logger.info(f"Backbone model configuration:")
        logger.info(f"  Sequence length: min=1, opt={opt_seq_len}, max={max_seq}")
        logger.info(f"  Image: {num_frames} frames, {img_channels}x{img_height}x{img_width}")
        logger.info(f"  Note: ONNX model uses shape (num_frames, 1, C, H, W)")

        # ONNX model has shape (num_frames, 1, C, H, W) where the 1 is batch size per frame
        min_shapes = {
            "input_ids": (1, 1),
            "attention_mask": (1, 1),
            "pixel_values": (1, 1, img_channels, img_height, img_width),
        }
        opt_shapes = {
            "input_ids": (1, opt_seq_len),
            "attention_mask": (1, opt_seq_len),
            "pixel_values": (num_frames, 1, img_channels, img_height, img_width),
        }
        max_shapes = {
            "input_ids": (1, max_seq),
            "attention_mask": (1, max_seq),
            "pixel_values": (16, 1, img_channels, img_height, img_width),  # Max 16 frames
        }
    else:
        # DiT model shapes (original code)
        # Optimal shapes - clamp to max_seq to satisfy MIN <= OPT <= MAX constraint
        opt_sa_seq = min(51, max_seq)    # Typical: 51 tokens for sa_embs
        opt_vl_seq = min(122, max_seq)   # Typical: 122 tokens for vl_embs and masks

        min_shapes = {
            "sa_embs": (1, 1, 1536),  # Min: 1 token
            "vl_embs": (1, 1, 2048),  # Min: 1 token
            "timestep": (1,),
            "image_mask": (1, 1),  # Min: 1 token
            "backbone_attention_mask": (1, 1),  # Min: 1 token
        }
        opt_shapes = {
            "sa_embs": (1, opt_sa_seq, 1536),
            "vl_embs": (1, opt_vl_seq, 2048),
            "timestep": (1,),
            "image_mask": (1, opt_vl_seq),
            "backbone_attention_mask": (1, opt_vl_seq),
        }
        max_shapes = {
            "sa_embs": (1, max_seq, 1536),
            "vl_embs": (1, max_seq, 2048),
            "timestep": (1,),
            "image_mask": (1, max_seq),
            "backbone_attention_mask": (1, max_seq),
        }

    # Create calibrator for INT8
    calibrator = None
    if args.precision == "int8":
        if args.calib_data is None:
            raise ValueError(
                "INT8 precision requires calibration data. "
                "Use --calib-data to provide a .npz file with calibration samples. "
                "Generate with: python scripts/deployment/collect_calibration_data.py"
            )

        if model_type == "backbone":
            calibrator = BackboneInt8Calibrator(args.calib_data, args.calib_cache)
        else:
            calibrator = Int8Calibrator(args.calib_data, args.calib_cache)

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        calibrator=calibrator,
        sparse_weights=args.sparse_weights,
        tactic_memory_mb=args.tactic_memory,
        timing_cache_path=args.timing_cache,
        builder_opt_level=args.builder_opt_level,
        low_memory_mode=args.low_memory_mode,
        disable_external_tactics=args.disable_external_tactics,
        heuristic_build=args.heuristic_build,
        memory_monitor=args.memory_monitor,
        tactic_shared_memory_mb=args.tactic_shared_memory,
        refittable=args.refittable,
        strip_plan=args.strip_plan,
    )


if __name__ == "__main__":
    main()
