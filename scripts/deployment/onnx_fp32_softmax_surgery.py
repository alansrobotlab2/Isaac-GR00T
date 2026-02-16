#!/usr/bin/env python3
"""
ONNX Graph Surgery: Insert FP32 Casts Around Attention Softmax

Fixes TRT FP16 backbone quality destruction (cos_sim=0.354 → target >0.99) by forcing
attention score computation and softmax to remain in FP32, preventing TRT's fused MHA
kernel from computing Q@K^T and softmax in FP16 where values overflow (73K-121K > 65504).

Strategy:
  For each attention Softmax node in the ONNX graph:
    1. Insert Cast(FP32) BEFORE the MatMul(Q, K^T) that produces attention scores
    2. Insert Cast(FP32) on the attention mask input to the Add
    3. The Softmax naturally computes in FP32 since its inputs are FP32
    4. Insert Cast(FP16) AFTER the Softmax output (before attn_weights @ V MatMul)

  This breaks TRT's MHA fusion pattern by introducing dtype boundaries, forcing TRT to
  keep the attention score computation in FP32 while allowing Q/K/V projections and the
  attn@V matmul to use FP16 tensor cores.

Usage (inside Docker):
    cd /workspace/gr00t/Isaac-GR00T

    # Basic: patch all softmax nodes
    python scripts/deployment/onnx_fp32_softmax_surgery.py \
        --input groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
        --output groot_n1d6_onnx_sdpa_fp32/backbone_fp32_softmax.onnx

    # Only patch language model (Qwen2) layers where overflow actually occurs
    python scripts/deployment/onnx_fp32_softmax_surgery.py \
        --input groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
        --output groot_n1d6_onnx_sdpa_fp32/backbone_fp32_softmax_lang.onnx \
        --only-language

    # Dry run: show what would be patched without modifying
    python scripts/deployment/onnx_fp32_softmax_surgery.py \
        --input groot_n1d6_onnx_sdpa_fp32/backbone_model.onnx \
        --dry-run
"""

import argparse
import logging
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_attention_softmax_nodes(graph, only_language=False, only_vision=False):
    """Find all Softmax nodes that are part of attention patterns.

    Attention pattern in ONNX:
        MatMul(Q, K^T) → [Scale/Mul] → Add(mask) → Softmax → MatMul(attn, V)

    Returns list of dicts with the relevant nodes for each attention block.
    """
    softmax_nodes = [n for n in graph.nodes if n.op == "Softmax"]

    attention_patterns = []
    for sm_node in softmax_nodes:
        name = sm_node.name

        # Filter by model component
        if only_language and "language_model" not in name:
            continue
        if only_vision and "vision_model" not in name:
            continue

        # Verify this is an attention softmax by checking the pattern:
        # Input should come from an Add (attention scores + mask)
        # Output should go to a MatMul (attn_weights @ V)
        sm_input = sm_node.inputs[0]
        sm_output = sm_node.outputs[0]

        # Check output goes to MatMul
        output_consumers = sm_output.outputs  # nodes that consume this tensor
        if not any(c.op == "MatMul" for c in output_consumers):
            logger.warning(f"Skipping {name}: output doesn't go to MatMul")
            continue

        # Trace back to find the Add node (attention scores + mask)
        add_node = None
        if sm_input.inputs:  # producers of the softmax input tensor
            for producer in sm_input.inputs:
                if producer.op == "Add":
                    add_node = producer
                    break

        # Trace back further to find the MatMul(Q, K^T) that produces attention scores
        qk_matmul = None
        if add_node is not None:
            for add_inp in add_node.inputs:
                if add_inp.inputs:
                    for producer in add_inp.inputs:
                        if producer.op == "MatMul":
                            qk_matmul = producer
                            break
                    if qk_matmul:
                        break
                # Also check if input directly comes from MatMul (no intermediate ops)
                if not qk_matmul and add_inp.inputs:
                    for producer in add_inp.inputs:
                        if producer.op == "Mul":  # scale multiplication
                            for mul_inp in producer.inputs:
                                if mul_inp.inputs:
                                    for pp in mul_inp.inputs:
                                        if pp.op == "MatMul":
                                            qk_matmul = pp
                                            break

        attn_v_matmul = None
        for c in output_consumers:
            if c.op == "MatMul":
                attn_v_matmul = c
                break

        attention_patterns.append({
            "softmax": sm_node,
            "add": add_node,
            "qk_matmul": qk_matmul,
            "attn_v_matmul": attn_v_matmul,
            "name": name,
            "component": "vision" if "vision_model" in name else "language",
        })

    return attention_patterns


def insert_cast_before(graph, target_tensor, to_dtype, suffix):
    """Insert a Cast node before a tensor, returning the new casted tensor.

    Creates: original_producer → original_tensor → Cast → new_tensor
    Rewires consumers of original_tensor to use new_tensor instead.
    """
    cast_output = gs.Variable(
        name=f"{target_tensor.name}_cast_{suffix}",
        dtype=to_dtype,
    )

    cast_node = gs.Node(
        op="Cast",
        name=f"Cast_{suffix}_{target_tensor.name}",
        inputs=[target_tensor],
        outputs=[cast_output],
        attrs={"to": onnx.TensorProto.FLOAT if to_dtype == np.float32 else onnx.TensorProto.FLOAT16},
    )

    graph.nodes.append(cast_node)
    return cast_output


def insert_cast_after(graph, source_node, output_idx, to_dtype, suffix):
    """Insert a Cast node after a node's output.

    Rewires: source_node output[idx] → Cast → new_tensor, and all
    original consumers of output[idx] now consume new_tensor.
    """
    original_output = source_node.outputs[output_idx]

    # Create intermediate tensor (the cast input — keeps original dtype)
    intermediate = gs.Variable(
        name=f"{original_output.name}_pre_cast_{suffix}",
        dtype=None,  # inherits from producer
    )

    # Create cast output (replaces original output for downstream consumers)
    cast_output = gs.Variable(
        name=f"{original_output.name}_cast_{suffix}",
        dtype=to_dtype,
    )

    # Rewire: source_node → intermediate → Cast → cast_output
    # All original consumers of original_output now consume cast_output
    cast_node = gs.Node(
        op="Cast",
        name=f"Cast_{suffix}_{original_output.name}",
        inputs=[intermediate],
        outputs=[cast_output],
        attrs={"to": onnx.TensorProto.FLOAT if to_dtype == np.float32 else onnx.TensorProto.FLOAT16},
    )

    # Swap the original output with intermediate on the source node
    source_node.outputs[output_idx] = intermediate

    # Rewire all consumers of the original output to use cast_output
    for consumer in list(original_output.outputs):
        for i, inp in enumerate(consumer.inputs):
            if inp is original_output:
                consumer.inputs[i] = cast_output

    graph.nodes.append(cast_node)
    return cast_node


def patch_attention_fp32_softmax(graph, patterns, dry_run=False):
    """Insert FP32 casts around attention softmax for each pattern.

    For each attention block:
      1. Cast softmax input to FP32 (breaks TRT's FP16 MHA fusion)
      2. Cast softmax output back to original dtype (for attn@V matmul)
    """
    patched = 0
    for pat in patterns:
        sm_node = pat["softmax"]
        name = pat["name"]
        component = pat["component"]

        if dry_run:
            logger.info(f"  [DRY RUN] Would patch: {name} ({component})")
            patched += 1
            continue

        # --- Insert Cast(FP32) before Softmax input ---
        sm_input_tensor = sm_node.inputs[0]
        fp32_input = insert_cast_before(graph, sm_input_tensor, np.float32, "to_fp32")

        # Rewire softmax to use the FP32 input
        sm_node.inputs[0] = fp32_input

        # --- Insert Cast(FP16) after Softmax output ---
        # The softmax output (now FP32) needs to be cast back for the attn@V matmul
        sm_output_tensor = sm_node.outputs[0]

        # IMPORTANT: Capture consumers BEFORE creating the Cast node, because
        # creating a node with inputs=[sm_output_tensor] adds it to
        # sm_output_tensor.outputs, which would cause a cycle if we redirect it.
        original_consumers = list(sm_output_tensor.outputs)

        fp16_output = gs.Variable(
            name=f"{sm_output_tensor.name}_to_fp16",
            dtype=np.float16,
        )
        cast_back_node = gs.Node(
            op="Cast",
            name=f"Cast_fp16_after_{sm_node.name}",
            inputs=[sm_output_tensor],
            outputs=[fp16_output],
            attrs={"to": onnx.TensorProto.FLOAT16},
        )

        # Redirect only the ORIGINAL consumers (not the Cast node we just created)
        for consumer in original_consumers:
            for i, inp in enumerate(consumer.inputs):
                if inp is sm_output_tensor:
                    consumer.inputs[i] = fp16_output

        graph.nodes.append(cast_back_node)

        logger.info(f"  Patched: {name} ({component}) — Cast(FP32) before, Cast(FP16) after")
        patched += 1

    return patched


def patch_attention_full_fp32_chain(graph, patterns, dry_run=False):
    """More aggressive: Cast Q@K^T MatMul inputs to FP32 and softmax output back.

    This forces the entire attention score chain (Q@K^T → scale → mask → softmax)
    to compute in FP32, not just softmax. More likely to break TRT's MHA fusion.

    For each attention block:
      1. Cast Q@K^T MatMul output to FP32 (forces score computation in FP32)
      2. Cast softmax output back to FP16 (for attn@V matmul)
    """
    patched = 0
    for pat in patterns:
        sm_node = pat["softmax"]
        qk_matmul = pat["qk_matmul"]
        name = pat["name"]
        component = pat["component"]

        if dry_run:
            qk_name = qk_matmul.name if qk_matmul else "NOT FOUND"
            logger.info(f"  [DRY RUN] Would patch: {name} ({component}) QK_MatMul={qk_name}")
            patched += 1
            continue

        if qk_matmul is None:
            logger.warning(f"  Could not find Q@K^T MatMul for {name}, falling back to softmax-only patch")
            sm_input_tensor = sm_node.inputs[0]
            fp32_input = insert_cast_before(graph, sm_input_tensor, np.float32, "to_fp32")
            sm_node.inputs[0] = fp32_input
        else:
            # Cast Q@K^T MatMul output to FP32
            # This makes everything downstream (scale, mask add, softmax) compute in FP32
            qk_output = qk_matmul.outputs[0]
            original_qk_consumers = list(qk_output.outputs)  # Capture BEFORE creating Cast
            fp32_qk = gs.Variable(
                name=f"{qk_output.name}_to_fp32",
                dtype=np.float32,
            )
            cast_qk = gs.Node(
                op="Cast",
                name=f"Cast_fp32_qk_{qk_matmul.name}",
                inputs=[qk_output],
                outputs=[fp32_qk],
                attrs={"to": onnx.TensorProto.FLOAT},
            )
            for consumer in original_qk_consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp is qk_output:
                        consumer.inputs[i] = fp32_qk
            graph.nodes.append(cast_qk)

        # Cast softmax output back to FP16
        sm_output_tensor = sm_node.outputs[0]
        original_sm_consumers = list(sm_output_tensor.outputs)  # Capture BEFORE creating Cast
        fp16_output = gs.Variable(
            name=f"{sm_output_tensor.name}_to_fp16",
            dtype=np.float16,
        )
        cast_back = gs.Node(
            op="Cast",
            name=f"Cast_fp16_after_{sm_node.name}",
            inputs=[sm_output_tensor],
            outputs=[fp16_output],
            attrs={"to": onnx.TensorProto.FLOAT16},
        )
        for consumer in original_sm_consumers:
            for i, inp in enumerate(consumer.inputs):
                if inp is sm_output_tensor:
                    consumer.inputs[i] = fp16_output
        graph.nodes.append(cast_back)

        logger.info(f"  Patched: {name} ({component})")
        patched += 1

    return patched


def patch_decomposed_fp32_softmax(graph, patterns, dry_run=False):
    """Replace Softmax with decomposed FP32 ops to prevent TRT MHA fusion.

    Replaces each Softmax(x) with:
      x_fp32 = Cast(x, FP32)
      max_val = ReduceMax(x_fp32, axis=-1, keepdims=True)
      shifted = Sub(x_fp32, max_val)        # numerical stability
      exp_val = Exp(shifted)
      sum_val = ReduceSum(exp_val, axis=-1, keepdims=True)
      result_fp32 = Div(exp_val, sum_val)
      result = Cast(result_fp32, FP16)

    This is mathematically identical to Softmax but uses 7 primitive ops that
    TRT cannot fuse into its MHA kernel. The entire computation happens in FP32.
    """
    patched = 0
    for pat in patterns:
        sm_node = pat["softmax"]
        name = pat["name"]
        component = pat["component"]
        axis = sm_node.attrs.get("axis", -1)

        if dry_run:
            logger.info(f"  [DRY RUN] Would decompose: {name} ({component})")
            patched += 1
            continue

        sm_input = sm_node.inputs[0]
        sm_output = sm_node.outputs[0]
        prefix = sm_node.name.replace("/", "_")

        # Capture original consumers BEFORE any modifications
        original_consumers = list(sm_output.outputs)

        # 1. Cast input to FP32
        fp32_input = gs.Variable(name=f"{prefix}_cast_fp32", dtype=np.float32)
        cast_to_fp32 = gs.Node(
            op="Cast", name=f"{prefix}/Cast_to_fp32",
            inputs=[sm_input], outputs=[fp32_input],
            attrs={"to": onnx.TensorProto.FLOAT},
        )

        # 2. ReduceMax for numerical stability
        max_val = gs.Variable(name=f"{prefix}_max", dtype=np.float32)
        reduce_max = gs.Node(
            op="ReduceMax", name=f"{prefix}/ReduceMax",
            inputs=[fp32_input], outputs=[max_val],
            attrs={"axes": [axis], "keepdims": 1},
        )

        # 3. Subtract max (numerical stability)
        shifted = gs.Variable(name=f"{prefix}_shifted", dtype=np.float32)
        sub_max = gs.Node(
            op="Sub", name=f"{prefix}/Sub",
            inputs=[fp32_input, max_val], outputs=[shifted],
        )

        # 4. Exp
        exp_val = gs.Variable(name=f"{prefix}_exp", dtype=np.float32)
        exp_node = gs.Node(
            op="Exp", name=f"{prefix}/Exp",
            inputs=[shifted], outputs=[exp_val],
        )

        # 5. ReduceSum
        sum_val = gs.Variable(name=f"{prefix}_sum", dtype=np.float32)
        reduce_sum = gs.Node(
            op="ReduceSum", name=f"{prefix}/ReduceSum",
            inputs=[exp_val], outputs=[sum_val],
            attrs={"axes": [axis], "keepdims": 1},
        )

        # 6. Divide
        result_fp32 = gs.Variable(name=f"{prefix}_softmax_fp32", dtype=np.float32)
        div_node = gs.Node(
            op="Div", name=f"{prefix}/Div",
            inputs=[exp_val, sum_val], outputs=[result_fp32],
        )

        # 7. Cast back to FP16
        result_fp16 = gs.Variable(name=f"{prefix}_softmax_fp16", dtype=np.float16)
        cast_to_fp16 = gs.Node(
            op="Cast", name=f"{prefix}/Cast_to_fp16",
            inputs=[result_fp32], outputs=[result_fp16],
            attrs={"to": onnx.TensorProto.FLOAT16},
        )

        # Add all new nodes
        new_nodes = [cast_to_fp32, reduce_max, sub_max, exp_node, reduce_sum, div_node, cast_to_fp16]
        graph.nodes.extend(new_nodes)

        # Redirect consumers of original softmax output to use fp16 result
        for consumer in original_consumers:
            for i, inp in enumerate(consumer.inputs):
                if inp is sm_output:
                    consumer.inputs[i] = result_fp16

        # Remove original softmax node by disconnecting it
        sm_node.inputs.clear()
        sm_node.outputs.clear()

        logger.info(f"  Decomposed: {name} ({component}) — 7 FP32 ops replace Softmax")
        patched += 1

    return patched


def main():
    parser = argparse.ArgumentParser(
        description="ONNX graph surgery: insert FP32 casts around attention softmax"
    )
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", default=None, help="Output ONNX model path (default: input_fp32_softmax.onnx)")
    parser.add_argument("--only-language", action="store_true",
                        help="Only patch language model (Qwen2) layers")
    parser.add_argument("--only-vision", action="store_true",
                        help="Only patch vision model (SigLIP2) layers")
    parser.add_argument("--mode", choices=["cast", "aggressive", "decompose"], default="cast",
                        help="Surgery mode: cast (FP32 around softmax), aggressive (FP32 from Q@K^T), "
                             "decompose (replace softmax with FP32 primitive ops)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without modifying")
    # Keep --aggressive for backwards compat
    parser.add_argument("--aggressive", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Backwards compat
    if args.aggressive:
        args.mode = "aggressive"

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        suffix = f"_fp32_softmax_{args.mode}"
        if args.only_language:
            suffix += "_lang"
        if args.only_vision:
            suffix += "_vis"
        args.output = f"{base}{suffix}{ext}"

    logger.info(f"Loading ONNX model: {args.input}")
    model = onnx.load(args.input)
    graph = gs.import_onnx(model)

    logger.info(f"Graph: {len(graph.nodes)} nodes")

    # Find attention softmax patterns
    patterns = find_attention_softmax_nodes(
        graph,
        only_language=args.only_language,
        only_vision=args.only_vision,
    )
    logger.info(f"Found {len(patterns)} attention softmax patterns")

    vision_count = sum(1 for p in patterns if p["component"] == "vision")
    lang_count = sum(1 for p in patterns if p["component"] == "language")
    logger.info(f"  Vision: {vision_count}, Language: {lang_count}")

    # Apply patches
    if args.mode == "aggressive":
        logger.info("Using AGGRESSIVE mode: casting Q@K^T output to FP32")
        patched = patch_attention_full_fp32_chain(graph, patterns, dry_run=args.dry_run)
    elif args.mode == "decompose":
        logger.info("Using DECOMPOSE mode: replacing Softmax with FP32 primitive ops")
        patched = patch_decomposed_fp32_softmax(graph, patterns, dry_run=args.dry_run)
    else:
        logger.info("Using CAST mode: casting softmax input to FP32")
        patched = patch_attention_fp32_softmax(graph, patterns, dry_run=args.dry_run)

    logger.info(f"Patched {patched} attention blocks")

    if args.dry_run:
        logger.info("[DRY RUN] No changes written")
        return

    # Cleanup and export
    graph.cleanup().toposort()
    patched_model = gs.export_onnx(graph)

    logger.info(f"Saving patched model to: {args.output}")
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Save with external data to avoid the 2GB protobuf limit
    # (the original model uses external data files for weights)
    external_data_name = os.path.basename(args.output) + ".data"
    onnx.save_model(
        patched_model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,  # tensors > 1KB go to external file
    )

    # Stats
    onnx_size = os.path.getsize(args.output) / (1024 * 1024)
    ext_path = os.path.join(output_dir, external_data_name)
    ext_size = os.path.getsize(ext_path) / (1024 * 1024) if os.path.exists(ext_path) else 0
    logger.info(f"Patched model: {onnx_size:.1f}MB + {ext_size:.1f}MB external data")

    # Count Cast nodes added
    cast_count = sum(1 for n in graph.nodes if n.op == "Cast")
    logger.info(f"Total Cast nodes in patched model: {cast_count}")


if __name__ == "__main__":
    main()
