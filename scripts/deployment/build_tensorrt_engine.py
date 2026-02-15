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
"""

import argparse
import logging
import os
import time

import tensorrt as trt


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
    calib_data_dir: str = None,
    calib_cache: str = None,
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
        calib_data_dir: Directory with calibration data (required for int8)
        calib_cache: Path to calibration cache file (optional, for int8)
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
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for unsupported layers
        logger.info("Enabled INT8 mode with FP16 fallback")

        if calib_data_dir is None:
            raise ValueError("--calib-data is required for INT8 precision")

        from calibrate_int8 import create_dit_calibrator

        cache_path = calib_cache or os.path.join(calib_data_dir, "int8_calib.cache")
        calibrator = create_dit_calibrator(calib_data_dir, cache_path)
        config.int8_calibrator = calibrator
        logger.info(f"INT8 calibrator loaded from {calib_data_dir}")
        logger.info(f"Calibration cache: {cache_path}")
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
    engine_dir = os.path.dirname(engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)
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
        "--calib-data",
        type=str,
        default=None,
        help="Directory with calibration data (required for int8 precision)",
    )
    parser.add_argument(
        "--calib-cache",
        type=str,
        default=None,
        help="Path to calibration cache file (optional, for int8)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="dit",
        choices=["dit", "backbone"],
        help="Model type for shape profiles (default: dit)",
    )

    args = parser.parse_args()

    if args.model_type == "backbone":
        # Backbone: pixel_values=[4,3,252,336] (fixed: ONNX trace unrolled 4 image splits),
        # input_ids=[1,seq], attention_mask=[1,seq]
        min_shapes = {
            "pixel_values": (4, 3, 252, 336),   # Fixed: 4 images (baked in trace)
            "input_ids": (1, 1),
            "attention_mask": (1, 1),
        }
        opt_shapes = {
            "pixel_values": (4, 3, 252, 336),   # Fixed: 4 images
            "input_ids": (1, 483),               # Typical: 483 tokens
            "attention_mask": (1, 483),
        }
        max_shapes = {
            "pixel_values": (4, 3, 252, 336),   # Fixed: 4 images
            "input_ids": (1, 1024),
            "attention_mask": (1, 1024),
        }
    else:
        # DiT: sa_embs=[1,51,1536], vl_embs=[1,483,2048], timestep=[1], masks
        min_shapes = {
            "sa_embs": (1, 1, 1536),
            "vl_embs": (1, 1, 2048),
            "timestep": (1,),
            "image_mask": (1, 1),
            "backbone_attention_mask": (1, 1),
        }
        opt_shapes = {
            "sa_embs": (1, 51, 1536),
            "vl_embs": (1, 483, 2048),
            "timestep": (1,),
            "image_mask": (1, 483),
            "backbone_attention_mask": (1, 483),
        }
        max_shapes = {
            "sa_embs": (1, 256, 1536),
            "vl_embs": (1, 1024, 2048),
            "timestep": (1,),
            "image_mask": (1, 1024),
            "backbone_attention_mask": (1, 1024),
        }

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        calib_data_dir=args.calib_data,
        calib_cache=args.calib_cache,
    )


if __name__ == "__main__":
    main()
