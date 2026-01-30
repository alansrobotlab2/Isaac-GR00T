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

    # For INT8 with calibration data:
    python build_tensorrt_engine.py \
        --onnx ./groot_n1d6_onnx/dit_model.onnx \
        --engine ./groot_n1d6_onnx/dit_model_int8.trt \
        --precision int8 \
        --calib-dir ./calibration_data \
        --calib-cache ./calibration_data/int8_calib.cache
"""

import argparse
import logging
import os
import time

import numpy as np
import tensorrt as trt


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator for TensorRT using pre-captured DiT input tensors.

    Expects a directory of .npy files produced by collect_calibration_data.py:
      - sa_embs.npy      (N, seq_len, 1536) float32
      - vl_embs.npy      (N, seq_len, 2048) float32
      - timesteps.npy    (N,) float32 or int64
      - image_masks.npy  (N, seq_len) bool  [optional]
      - backbone_attention_masks.npy (N, seq_len) bool [optional]
    """

    def __init__(self, calib_dir: str, cache_file: str, batch_size: int = 1):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        logger.info(f"Loading calibration data from {calib_dir}...")

        self.calib_data = {
            "sa_embs": np.load(os.path.join(calib_dir, "sa_embs.npy")).astype(np.float32),
            "vl_embs": np.load(os.path.join(calib_dir, "vl_embs.npy")).astype(np.float32),
            "timestep": np.load(os.path.join(calib_dir, "timesteps.npy")).astype(np.int64),
        }

        # Optional mask files
        image_masks_path = os.path.join(calib_dir, "image_masks.npy")
        if os.path.exists(image_masks_path):
            self.calib_data["image_mask"] = np.load(image_masks_path).astype(np.bool_)

        backbone_masks_path = os.path.join(calib_dir, "backbone_attention_masks.npy")
        if os.path.exists(backbone_masks_path):
            self.calib_data["backbone_attention_mask"] = np.load(backbone_masks_path).astype(np.bool_)

        self.num_samples = self.calib_data["sa_embs"].shape[0]
        logger.info(f"Loaded {self.num_samples} calibration samples")
        for name, arr in self.calib_data.items():
            logger.info(f"  {name}: {arr.shape} ({arr.dtype})")

        # Allocate device memory for each input
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            import cuda.cudart as cudart
        self._cudart = cudart

        self.device_inputs = {}
        for name, arr in self.calib_data.items():
            single_sample_bytes = int(np.prod(arr.shape[1:])) * arr.itemsize
            err, ptr = cudart.cudaMalloc(single_sample_bytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to allocate CUDA memory for {name}")
            self.device_inputs[name] = ptr

        logger.info("INT8 calibrator initialized")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_samples:
            return None

        cudart = self._cudart
        bindings = []
        for name in names:
            if name not in self.calib_data:
                logger.warning(f"Unknown input name requested by TRT: {name}")
                return None

            data = np.ascontiguousarray(
                self.calib_data[name][self.current_index : self.current_index + 1]
            )
            ptr = self.device_inputs[name]
            cudart.cudaMemcpy(
                ptr, data.ctypes.data, data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
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
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def __del__(self):
        if hasattr(self, "_cudart") and hasattr(self, "device_inputs"):
            for ptr in self.device_inputs.values():
                self._cudart.cudaFree(ptr)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
    calibrator: trt.IInt8Calibrator = None,
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
        calibrator: INT8 calibrator instance (required when precision='int8')
    """
    logger.info("=" * 80)
    logger.info("TensorRT Engine Builder")
    logger.info("=" * 80)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Engine output: {engine_path}")
    logger.info(f"Precision: {precision.upper()}")
    logger.info(f"Workspace: {workspace_mb} MB")
    logger.info("=" * 80)

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

    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))

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
        if calibrator is None:
            raise ValueError(
                "INT8 precision requires a calibrator. "
                "Use --calib-dir to provide calibration data from collect_calibration_data.py"
            )
        config.int8_calibrator = calibrator
        logger.info("Enabled INT8 mode with FP16 fallback and entropy calibration")
    elif precision == "fp32":
        logger.info("Using FP32 (default precision)")
    else:
        raise ValueError(f"Unknown precision: {precision}")

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

    # Save engine
    logger.info(f"\nSaving engine to {engine_path}...")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024**2)
    logger.info(f"Engine saved! Size: {engine_size_mb:.2f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("ENGINE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Engine file: {engine_path}")
    logger.info(f"Size: {engine_size_mb:.2f} MB")
    logger.info(f"Build time: {build_time:.1f}s")
    logger.info(f"Precision: {precision.upper()}")
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
        help="Precision mode (default: bf16)",
    )
    parser.add_argument(
        "--workspace", type=int, default=8192, help="Workspace size in MB (default: 8192)"
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Directory with calibration .npy files for INT8 (from collect_calibration_data.py)",
    )
    parser.add_argument(
        "--calib-cache",
        type=str,
        default=None,
        help="Path to save/load INT8 calibration cache (default: <calib-dir>/int8_calib.cache)",
    )

    args = parser.parse_args()

    # Dynamic shapes for the DiT model inputs
    min_shapes = {
        "sa_embs": (1, 1, 1536),  # Min: 1 token
        "vl_embs": (1, 1, 2048),  # Min: 1 token
        "timestep": (1,),
        "image_mask": (1, 1),  # Min: 1 token
        "backbone_attention_mask": (1, 1),  # Min: 1 token
    }
    opt_shapes = {
        "sa_embs": (1, 51, 1536),  # Typical: 51 tokens
        "vl_embs": (1, 122, 2048),  # Typical: 122 tokens
        "timestep": (1,),
        "image_mask": (1, 122),  # Typical: 122 tokens
        "backbone_attention_mask": (1, 122),  # Typical: 122 tokens
    }
    max_shapes = {
        "sa_embs": (1, 256, 1536),  # Max: 256 tokens (generous)
        "vl_embs": (1, 512, 2048),  # Max: 512 tokens (generous)
        "timestep": (1,),
        "image_mask": (1, 512),  # Max: 512 tokens
        "backbone_attention_mask": (1, 512),  # Max: 512 tokens
    }

    # Create INT8 calibrator if needed
    calibrator = None
    if args.precision == "int8":
        if args.calib_dir is None:
            raise ValueError(
                "INT8 precision requires calibration data. "
                "Use --calib-dir to provide the directory from collect_calibration_data.py"
            )
        cache_file = args.calib_cache or os.path.join(args.calib_dir, "int8_calib.cache")
        calibrator = Int8Calibrator(args.calib_dir, cache_file)

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        calibrator=calibrator,
    )


if __name__ == "__main__":
    main()
