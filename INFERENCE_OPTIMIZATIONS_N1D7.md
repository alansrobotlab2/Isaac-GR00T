# GR00T N1.7 Inference Optimization on Jetson AGX Orin

Follow-up to `INFERENCE_OPTIMIZATIONS.md` (N1.6 / Eagle backbone). Same goal — control
rate on an AGX Orin 64GB — redone for N1.7 (Qwen3-VL / Cosmos-Reason2-2B backbone).

**Result: 480 ms → 304 ms per step (1.6x) at unchanged accuracy, 250 ms (1.9x) with
2-step denoising.** Measured on 2026-10-12.

**Round 2 (2026-10-13): 250 ms → 141 ms per step with 2-step denoising (7.1 Hz), and
304 ms → 176 ms with 4-step (5.7 Hz), again at unchanged accuracy.** The first sections
below are the round-1 record; round 2 has its own section, and the recommended
configuration at the end reflects both.

## Setup

| | |
|---|---|
| Device | Jetson AGX Orin Developer Kit, Jetson Linux R39.2, CUDA 13.2 |
| Stack | `scripts/deployment/orin/install_deps.sh`: PyTorch 2.13.0+cu132, TensorRT 10.16.1, flash-attn 2.8.3 |
| Model | `nvidia/GR00T-N1.7-3B` (base), BF16, action horizon 40 |
| Data | `demo_data/droid_sample`, `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, 3 trajectories |
| Workload | **4 images per step** (1,792 ViT patches → 448 visual tokens, 461 LLM tokens) |
| Harness | `scripts/deployment/standalone_inference_script.py --execution-horizon 8 --skip-timing-steps 3` |

Two things to keep in mind when comparing with other numbers:

- Upstream's Orin table (`scripts/deployment/README.md`) is LIBERO with **1 camera**; DROID
  feeds 4 images, so the backbone here does roughly 4x the vision work.
- **Use the Orin installer, not a plain `uv sync`.** The repo-root lockfile installs the
  desktop stack (PyTorch 2.9 + cu128, TensorRT 10.15). It runs, but the same baseline
  measured 644 ms on that stack vs 473 ms on the Orin stack.

## Results

End-to-end latency per inference step and action error vs ground truth, average of 3 trajectories.

| Configuration | Avg / step | Rate | MSE | Extra flags |
|---|---|---|---|---|
| PyTorch baseline (`origin/main` and this branch, identical) | 480 ms | 2.1 Hz | 0.017264 | — |
| torch.compile DiT + action head (flash attention) | 362 ms | 2.8 Hz | 0.017270 | `--compile-dit --compile-action-head` |
| torch.compile everything (SDPA) | 328 ms | 3.1 Hz | 0.017272 | `--attn-implementation sdpa --compile-backbone --compile-dit --compile-action-head` |
| TensorRT full pipeline, upstream default (FP32 ViT) | 389 ms | 2.6 Hz | 0.017284 | `--inference-mode trt_full_pipeline` |
| **TensorRT full pipeline, BF16 ViT** | **304 ms** | **3.3 Hz** | 0.017243 | engines built with `--vit-precision bf16` |
| PyTorch, 2-step denoising | 364 ms | 2.7 Hz | 0.016653 | `--denoising-steps 2` |
| torch.compile everything (SDPA), 2-step | 263 ms | 3.8 Hz | 0.016692 | as above + `--denoising-steps 2` |
| **TensorRT BF16 ViT, 2-step** | **250 ms** | **4.0 Hz** | 0.016678 | as above + `--denoising-steps 2` |

`gr00t/eval/open_loop_eval.py` on 2 trajectories: 439 ms → 283 ms with
`--compile-dit --compile-action-head --cudnn-benchmark --pipeline-backbone-dit --denoising-steps 2`
(MSE 0.00844 → 0.00801). Its baseline is lower than the standalone script's because the
standalone script forces deterministic CUDA algorithms in `set_seed`.

### Where the time goes

Per-component latency for one step (ms), model only:

| Component | PyTorch flash-attn | PyTorch SDPA | TRT (FP32 ViT) | TRT (BF16 ViT) |
|---|---|---|---|---|
| ViT | 109 | 117 | 145 | ~60 |
| LLM | 81 | 64 | 53 | 53 |
| VL self-attention | 8.6 | 8.6 | 8.8 | 8.8 |
| DiT (4 steps) | 200 | 198 | 87 | 87 |
| State/action encoders, decoder | 9 | 9 | 9 | 9 |

Outside the model, per step: ~29 ms CPU preprocessing (hidden by async prefetch), ~5 ms
input transfer, and 17–48 ms action decoding on the CPU. The decode cost is specific to
DROID: converting relative end-effector actions to absolute builds ~1,600 validated `Pose`
objects per chunk (`gr00t/data/state_action/pose.py`). A joint-space embodiment does not
pay it.

## Findings

1. **The ViT precision is the single biggest lever for TensorRT.** Upstream exports the ViT
   in FP32 "for accuracy". On Orin that engine (145 ms) is slower than the PyTorch BF16 ViT
   (109 ms), which is why upstream's full pipeline only reached 389 ms here. A BF16 ViT
   engine cuts the step to 304 ms and is at least as accurate end to end:

   | ViT engine | ViT cosine | Backbone cosine | Final action cosine | MSE (3 traj) |
   |---|---|---|---|---|
   | FP32 (upstream) | 0.99885 | 0.99989 | 0.999937 | 0.017284 |
   | BF16 | 0.99722 | 0.99984 | 0.999986 | 0.017243 |
   | FP16 (weakly typed) | 0.99826 | 0.99985 | 0.999878 | — (traj 0 only: same speed as BF16, slightly worse error) |

   Cosines are from `verify_n1d7_trt.py --mode n17_full_pipeline` against PyTorch.
   This was validated on the base model and DROID only — re-run the verify step on your own
   fine-tuned checkpoint before relying on it.

2. **TensorRT is a 2.3x win on the DiT** (50 → 22 ms per denoising step) and a modest win on
   the LLM (81 → 53 ms). Unlike N1.6, the backbone in TRT is no longer a dead end, as long as
   the ViT is not FP32.

3. **torch.compile alone gets most of the way** (328 ms) with no engine build:
   - DiT + action head: 480 → 362 ms.
   - Backbone: only works with SDPA. With flash attention, dynamo cannot trace
     `flash_attn._flash_attn_varlen_forward` and crashes on the first inference;
     `--compile-backbone` now skips with a warning in that case. Compiled SDPA beats
     uncompiled flash attention by ~35 ms on the Orin stack. (On the wrong desktop stack it
     was 170 ms *slower* — another reason to use the Orin installer.)

4. **2-step denoising** saves ~55 ms per removed pair of steps in TRT (~110 ms in PyTorch)
   and slightly *lowers* open-loop MSE on DROID, as it did on N1.6. Open-loop MSE is not a
   task-success metric; confirm on the robot or in sim.

5. **FP32 Euler accumulation and the policy/loader refactors are free and output-identical**:
   this branch with no flags reproduces `origin/main` to six decimals at the same speed.

### What did not help on N1.7

- **Backbone/DiT pipelining (`--pipeline-backbone-dit`)**: 304 → 300 ms with TRT, 473 → 472 ms
  in PyTorch. It lowers the best-case step (min 109 ms) but not the average. The flag
  remains; it was worth ~10% on N1.6 and may matter with a heavier CPU-side decode.
- **FP16 (`--use-fp16`)**: 489 ms vs 476 ms for BF16 in PyTorch. Slower, unlike N1.6.
- **Keeping the ViT in PyTorch with LLM + action head in TRT (`--trt-pytorch-vit`)**: 378 ms.
  Better than the FP32 ViT engine, worse than the BF16 one. Useful as a fallback if a BF16
  ViT engine fails verification on some checkpoint.
- **TF32**: within noise (~1%).

## Round 2: clocks, host overhead, and a fused action head

Round 1 left the step at 250 ms (2-step). A profile of that step showed the GPU was
waiting on the host for a large part of it, so round 2 went after everything that is not
kernel time, then after the action head. Same device, model, data and harness as above.

### Results

End-to-end per step, `standalone_inference_script.py`, 3 trajectories. Each row adds to
the one above it.

| Configuration | 2-step | 4-step | MSE (2 / 4-step) |
|---|---|---|---|
| Round-1 best (clocks not locked) | 250 ms | 304 ms | 0.016678 / 0.017243 |
| + `sudo jetson_clocks` (same code, same engines) | 193 ms | 241 ms | 0.016678 / 0.017243 |
| + rebuilt engines, no host syncs, layout cache, batched action decode | 156 ms | | 0.016713 |
| + CUDA graphs (default) | 150 ms | 193 ms | 0.016713 / 0.017289 |
| **+ fused action head (`--fused-denoising-steps`)** | **146 ms** | **176 ms** | 0.016717 / 0.017283 |
| **+ `--pipeline-backbone-dit`** | **141 ms** | | 0.016717 |
| Opt-in: INT4 DiT weights (see below) | 140 ms, 137 ms pipelined | 166 ms | 0.016688 / 0.017275 |

`open_loop_eval.py` in-process, 2-step, 2 trajectories: 246 → 145 ms. Through
`run_gr00t_server.py` from a local client: 280 → 164 ms.

Where the 2-step step goes now (GPU-only engine time from `trtexec`): ViT 58 ms, fused
action head 42 ms, LLM 32 ms, everything else (glue, input transfer, decode) about 9 ms.

### What changed

1. **Lock the clocks: 250 → 193 ms.** The power mode was already MAXN, but the GPU ran
   under the `nvhost_podgov` governor and the CPUs under `schedutil`. `sudo jetson_clocks`
   pins both at maximum. Besides the raw speed, it removes a penalty that only shows up
   on a robot: after a 100 ms idle gap (the robot executing a chunk) a DiT call took
   31.7 ms instead of 21.8 ms. It does not survive a reboot.

2. **The LLM engine synchronised with the host in the middle of every inference.** The
   export injected deepstack features with `masked_scatter`, which becomes `NonZero` in
   ONNX. Its output shape depends on data, so TensorRT stops, reads the result back and
   resumes (a "train station"). The engine cost 45 ms in-process against 37 ms of GPU
   work, and could not be captured in a CUDA graph. The export now uses CumSum + Gather +
   Where, which is bit-identical in PyTorch and has static shapes: 32 ms.

3. **No host sync between engines, and no per-step host work.** `trt_torch.Engine` used
   to block on the stream after every call and zero-fill its outputs. The engines now
   queue back to back (`sync=False`), like ordinary PyTorch modules. The transformers
   helpers that build the image mask and the 3D rope position ids run Python loops and
   read GPU tensors back; their results depend only on the prompt tokens and the image
   grid, so they are cached per token layout (`_StaticPlan`) and recomputed when the
   prompt changes. Timestep and position-embedding tensors are built once.

4. **CUDA graphs, on by default: about 5 ms.** Each engine is captured once and replayed,
   which replaces a few hundred kernel launches by one. `--no-trt-cuda-graph` turns it
   off. The graph is re-captured when the prompt's token count changes; an engine that
   cannot be captured falls back to plain execution with a warning. This is what the
   N1.6 notes called a dead end; it works here because the engines no longer contain
   host round-trips.

5. **Fused action head: 5 ms (2-step), 16 ms (4-step).** `--fused-denoising-steps N`
   exports VLLN, VL self-attention, the state encoder and N unrolled Euler steps as one
   graph (`action_head_fused_<N>step.engine`). One engine call replaces eight or
   fourteen. Nearly all of the gain comes from one thing: the DiT's 16 cross-attention
   blocks project the 461 vision-language tokens to keys and values on every denoising
   step, although nothing about them changes. In the fused graph the first step computes
   them and the rest reuse them. Fusing without that sharing saved only 1 ms. The
   runtime picks the fused engine when the policy runs that step count and otherwise
   uses the per-component engines, which are still built. Euler integration is FP32, as
   in PyTorch.

6. **Relative-to-absolute action conversion is batched: 18 → 1.7 ms** (48 ms before
   the clocks were locked). It built about 1,600 `Pose` objects per chunk; it now runs on
   whole arrays and matches the per-pose code to float rounding (tested for all three
   action formats). Only embodiments with relative end-effector actions paid this.

7. **`--pipeline-backbone-dit` is worth 5 ms now.** It was noise in round 1. With graphs
   the host is free early enough for the next backbone to overlap the CPU tail.

The no-sync, layout-cache and CUDA-graph modes are bit-identical to the old synchronous
path on the same engines (max difference 0 over 20 frames, also when switching between
prompts of 461, 464 and 479 tokens). Rebuilding engines moves the result within the usual
TensorRT build-to-build spread: worst-case cosine against PyTorch over 20 frames was
0.99996 for the round-1 engines and 0.99992 for the current ones.

### Opt-in: INT4 DiT weights

The DiT has 1.1 billion weights and 41 query tokens, so part of each denoising step is
just streaming 2.2 GB of weights. TensorRT can keep `MatMul` weights as INT4 and
dequantize inside the GEMM kernel. A synthetic stack of the same size went from 11.0 to
5.7 ms on this GPU, so the mechanism works on Orin. On the real model the gain is
smaller, because most layers are small:

| | BF16 | INT4 weights |
|---|---|---|
| DiT engine, per call | 19.3 ms | 17.1 ms |
| Fused 2-step action head | 42.0 ms | 36.9 ms |
| Step, 2-step / 4-step | 146 / 176 ms | 140 / 166 ms |
| Fused engine size | 3.1 GB | 1.8 GB |
| Open-loop MSE, 2-step / 4-step | 0.016717 / 0.017283 | 0.016688 / 0.017275 |
| Worst-case action cosine vs PyTorch (20 frames) | 0.99992 | 0.99951 |

Open-loop error is unchanged, but each action chunk is about six times further from the
PyTorch reference, so this is not the default. `scripts/deployment/quantize_onnx_int4.py`
writes a mirrored ONNX directory to build from. Two things TensorRT 10.16 is picky about:
only block size 128 has a fast kernel (block 32 was four times *slower* than BF16), and a
quantized weight must feed exactly one `MatMul`, so the tool gives each unrolled step its
own INT4 copy.

```bash
python scripts/deployment/quantize_onnx_int4.py \
    --onnx-dir gr00t_trt_deployment/onnx --output-dir gr00t_trt_deployment/onnx_int4
python scripts/deployment/build_tensorrt_engine.py --mode full_pipeline \
    --onnx-dir gr00t_trt_deployment/onnx_int4 --engine-dir gr00t_trt_deployment/engines_int4
```

### What did not help in round 2

- **Builder optimization level 5** (`--builder-optimization-level 5`): DiT 19.0 → 18.5 ms
  per call, LLM 30.9 → 30.7 ms, ViT unchanged, for 2.5x the build time. Available, not
  the default.
- **Static shape profiles** instead of the dynamic token-count range: DiT −0.3 ms, LLM
  −0.9 ms. Not adopted: a different prompt length would need a rebuild.
- **A dedicated CUDA stream** instead of the default one: no difference.
- **Server transport.** A request with 4 images is 0.7 MB and costs 0.3 ms to serialise
  and parse. The 19 ms the server adds over in-process is CPU preprocessing (about 15 ms)
  running before the GPU instead of beside it. Running the per-camera image transforms
  in threads saved 1.9 ms and was not worth touching the shared processor for. Round 1
  blamed the socket; that was wrong.
- **INT4 on a weight shared by several MatMuls**: TensorRT fails to build it.

## Recommended configuration

```bash
# Once per boot: pin CPU/GPU clocks at maximum.
sudo jetson_clocks

source scripts/activate_orin.sh

# One-time: export, build, and verify engines (~12 min on AGX Orin with one fused step
# count). Check the reported cosine. Use the step count you will run with.
python scripts/deployment/build_trt_pipeline.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --vit-precision bf16 --fused-denoising-steps 2 --steps export,build,verify

# Inference / benchmarking
python scripts/deployment/standalone_inference_script.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --inference-mode trt_full_pipeline --trt-engine-path ./gr00t_trt_deployment/engines \
    --execution-horizon 8 --denoising-steps 2 --pipeline-backbone-dit
```

The inference server and open-loop eval take the same engines:

```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path <checkpoint> --embodiment-tag <TAG> \
    --trt-engine-path ./gr00t_trt_deployment/engines --denoising-steps 2

python gr00t/eval/open_loop_eval.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --trt-engine-path ./gr00t_trt_deployment/engines --denoising-steps 2 --execution-horizon 8
```

Measured with these (2 DROID trajectories, 2-step): `open_loop_eval.py` in-process 145 ms per
step; through the server from a local client 164 ms. The server does not seed the sampling
noise, so its open-loop MSE varies run to run.

Notes:

- `--denoising-steps` must match a fused engine to use it. With any other value the
  per-component engines are used (correct, a few ms slower). `--fused-denoising-steps 2 4`
  builds both; each adds about 3 GB and 3-5 minutes.
- CUDA graphs are captured during the first inference, which is slower. A change of
  prompt length re-captures once (the first step after the change).
- With CUDA graphs an engine's outputs are fixed buffers that its next call overwrites.
  The shipped forward functions consume them in order; keep that in mind in custom code
  built on `trt_torch.Engine(cuda_graph=True)`.

No-engine alternative (longer warmup on first inference):

```bash
python scripts/deployment/standalone_inference_script.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --attn-implementation sdpa --compile-backbone --compile-dit --compile-action-head \
    --execution-horizon 8 --denoising-steps 2
```

Engines are tied to the GPU, the TensorRT version, and the image/camera configuration baked
in at export; rebuild after changing any of them.

## Not yet done

- Validation on a fine-tuned checkpoint and a real robot embodiment (everything above is the
  base model on DROID demo data). In particular, 2-step denoising and INT4 weights are
  only supported by open-loop MSE here.
- The ViT is now the largest block (58 of ~141 ms) and nothing in round 2 touched it. It
  is compute-bound (1,792 patches), so the levers are fewer or smaller camera images (a
  data/fine-tuning decision), INT8 with calibration, or skipping the ViT for cameras
  whose image has not changed.
- The server runs CPU preprocessing (about 15 ms) serially before the model. A client
  that sends the next observation early, or GPU-side image preprocessing, would hide it.
- The PyTorch wheel from the Orin installer has no sm_87 kernels (it warns at import and
  JIT-compiles). This only affects the PyTorch glue and the no-engine path.
