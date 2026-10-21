# GR00T N1.7 Inference Optimization on Jetson AGX Orin

Follow-up to `INFERENCE_OPTIMIZATIONS.md` (N1.6 / Eagle backbone). Same goal — control
rate on an AGX Orin 64GB — redone for N1.7 (Qwen3-VL / Cosmos-Reason2-2B backbone).

**Result: 480 ms → 304 ms per step (1.6x) at unchanged accuracy, 250 ms (1.9x) with
2-step denoising.** Measured on 2026-10-12.

**Round 2 (2026-10-13): 250 ms → 141 ms per step with 2-step denoising (7.1 Hz), and
304 ms → 176 ms with 4-step (5.7 Hz), again at unchanged accuracy.** The first sections
below are the round-1 record; round 2 has its own section, and the recommended
configuration at the end reflects both.

**Round 3 (2026-10-13): 141 ms → 106 ms per step with 2-step denoising (9.5 Hz), 176 ms →
131 ms with 4-step, and 164 ms → 118 ms through the inference server.** This uses opt-in
INT8 ViT and DiT engines; open-loop MSE moves by under 3%, and the actions are about twice
as far from PyTorch as with BF16 engines (2.9% vs 1.4% on average). Without INT8, GPU image
preprocessing alone takes the server from 158 to 151 ms. See the round 3 section.

**Round 4 (2026-10-14): 95 ms per step in-process (2-step), 104 ms through the server
(9.6 Hz), and a sustained 10 Hz with 111 ms latency through the new pipelined server**, with
the LLM in INT8 too. The cost is accuracy: actions are 4.2% from PyTorch on average (BF16
engines: 1.4%), with open-loop MSE unchanged. See the round 4 section.

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

## Round 3: INT8 ViT and DiT, GPU image preprocessing

Round 2 left the 2-step step at 141 ms and the server round trip at 164 ms. The target
for the robot is 10 Hz, a 100 ms budget through the server, with 4 cameras today and
possibly 6. Round 3 went after the ViT and the DiT, which round 2 left in BF16, and
after the CPU preprocessing the server does before the model. Same device, model, data
and harness as above.

### Results

End-to-end per step, `standalone_inference_script.py`, 3 trajectories. Action error is
the relative L2 distance of the unnormalized action chunk from the PyTorch model with the
same sampling noise, over 20 frames spread across the three DROID episodes.

| Configuration | 2-step | 4-step | MSE (2 / 4-step) | Action error vs PyTorch (mean / worst) |
|---|---|---|---|---|
| Round-2 engines (BF16) | 139.5 ms | 175.8 ms | 0.016717 / 0.017283 | 1.4% / 5.1% |
| INT8 ViT | 118.7 ms | 154.3 ms | 0.016701 / 0.017488 | 2.4% / 3.8% |
| INT8 DiT (action head) | 127.4 ms | 152.0 ms | 0.017029 / 0.017540 | 2.1% / 6.5% |
| **INT8 ViT + DiT** | **105.7 ms** | **130.9 ms** | 0.017017 / 0.017739 | 2.9% / 5.1% |

2-step rows use `--pipeline-backbone-dit`. Open-loop MSE moves by at most 3% (it went
down for the ViT alone). The PyTorch-relative error roughly doubles, but its worst case
stays where the BF16 engines already are. That is the trade to judge on the robot.

Through `run_gr00t_server.py` from a local client, one fixed prompt, 2-step, median of
63 requests (the 90th percentile is within 2 ms of the median):

| Configuration | Round trip |
|---|---|
| Round 2 (as measured in round 2) | 164 ms |
| Round-2 engines, threaded CPU preprocessing (`--no-gpu-image-preprocessing`) | 157.6 ms |
| + GPU image preprocessing (server default now) | 150.8 ms |
| **+ INT8 ViT + DiT engines** | **117.7 ms** |

### What changed

1. **INT8 needs quantized activations, not just weights.** Round 2's INT4 experiment
   only quantized weights, and TensorRT then runs BF16 kernels with the weights
   dequantized, which is why it gained little. With a QuantizeLinear/DequantizeLinear
   pair on the activation as well, TensorRT picks integer tensor-core GEMMs. On a
   synthetic stack of MLP blocks on this GPU that is 2.1x faster at the ViT's 1,792
   tokens, 2.1x at the LLM's 461 tokens, and 1.75x at the DiT's 41 tokens (there it
   halves the weight bytes read).

2. **New tool: `scripts/deployment/quantize_onnx_int8.py`.** It rewrites every linear
   layer of an exported graph as INT8 (weights per output channel, activations per
   tensor) and calibrates the activation ranges with TensorRT itself: the unmodified
   graph is built once with every linear input marked as an output, calibration samples
   run through it, and each range is the clip that minimizes squared error. No PyTorch
   model or onnxruntime is needed.

3. **Plain post-training INT8 breaks the ViT; SmoothQuant fixes most of it.** With every
   ViT linear layer in INT8 and nothing else, the image embeddings fell to cosine 0.87
   against an FP32 reference (BF16: 0.996) and the actions were 6.2% off on average,
   31% on the worst frame. A per-layer sweep found no small set of culprits: the MLP
   down-projections are the most sensitive, but even keeping the 24 worst layers in BF16
   only reached 0.97. Several layer inputs have a few channels 3-10x larger than the
   rest, which is what SmoothQuant addresses: divide each activation channel by
   `s = amax_x^a / amax_w^(1-a)` and multiply the weight's input channel by `s`. The
   tool inserts the division as a `Mul` that TensorRT fuses into the quantize kernel.
   Strength 0.7 was best for the ViT (sweep 0.3-0.85), 0.8 for the DiT (0.5, 0.8, off).

4. **INT8 ViT: 58 → 37 ms.** Keeping the MLP down-projections in BF16 (`--int8-skip
   'blocks\.\d+/mlp/linear_fc2'`) costs 5.5 ms and did not lower the end-to-end action
   error in the final measurement, so it is not the default.

5. **INT8 action head: 47 → 35 ms for the fused 2-step engine** (without CUDA graphs;
   engine alone). Only the DiT is quantized by default. Also quantizing the VL
   self-attention and the encoders (`--int8-action-head-all`) saves another 3.6 ms but
   doubles the head's error against the BF16 head (0.6% → 1.4%). The per-step DiT engine
   (used when the step count has no fused engine) is quantized too.

6. **Batched ViT attention: 1.3 ms, bit-identical.** Round 2's profile counted the ViT's
   QKV projections as attention and suggested 12 ms of unfused attention. That was
   wrong: TensorRT already used its fused attention kernel, once per image (96 calls).
   The export now runs one batched attention per block (24 calls), and the softmax no
   longer round-trips through FP32 in the graph. Outputs are bit-identical to the old
   engine; the ViT engine goes from 59.7 to 58.4 ms.

7. **Server preprocessing: 19.1 → 8.9 ms per observation.** Per observation with 4 images,
   including the copy to the GPU:

   | Preprocessing | Time |
   |---|---|
   | Round 2: sequential, all on the CPU | 19.1 ms |
   | Per-frame resize/crop in a thread pool (identical output) | 13.6 ms |
   | + Qwen image processor on the GPU | 8.9 ms |

   The thread pool scales with cameras: 12 frames take 4.8 ms instead of 17 ms. The GPU
   image processor (resize, normalize, patchify) copies uint8 frames instead of float32
   patches and runs on its own CUDA stream. Pixel values match the CPU path to within
   one 8-bit step (max difference 0.0098 in normalized units). It is on by default in
   `run_gr00t_server.py` and opt-in (`--gpu-image-preprocessing`) in
   `open_loop_eval.py`, `standalone_inference_script.py` and `Gr00tPolicy`.

### How to build INT8 engines

```bash
python scripts/deployment/build_trt_pipeline.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --vit-precision bf16 --fused-denoising-steps 2 \
    --int8 vit action_head --steps export,quantize,build,verify
```

The `quantize` step runs the PyTorch model on 64 frames spread over the dataset
(`--int8-calibration-frames`), records the inputs of the ViT, the fused heads and the
DiT, and writes `<output-dir>/onnx_int8` (quantized graphs plus links to the rest). The
build step then compiles that directory. On the AGX Orin the whole run took 31 minutes
with fused engines for 2 and 4 steps: export 3, quantize 9, build 17, verify 3.

Caveats:

- **Calibrate on your own robot's data.** Here calibration and evaluation frames come
  from the same three DROID episodes (different frames). A fine-tuned checkpoint has
  different activation ranges; rebuild with its dataset.
- **Engine size.** TensorRT needs a separate INT8 copy of a weight for each unrolled
  denoising step, so the fused 4-step INT8 engine is 5.2 GB (BF16: 3.3 GB). The 2-step
  one is the same size as BF16.
- The verify step reported cosine 0.9952 for the final action (4-step, one frame).

### What did not help in round 3

- **Fusing the ViT attention** was worth 1.3 ms, not the 6-9 ms estimated from the
  mislabelled profile (see above).
- **Keeping the ViT MLP down-projections in BF16** (see above).
- **Plain INT8 without smoothing** (see above).

## Round 4: INT8 LLM, prompt cache, pipelined server

Round 3 ended at 106 ms per step in-process and 118 ms through the server, against a
10 Hz target. Round 4 quantized the LLM, trimmed what the server does around the model,
and added an opt-in server mode that overlaps requests. Same device, model, data and
harness as above.

### Results

In-process, `standalone_inference_script.py`, 3 trajectories; action error as in round 3
(20 frames, 2-step, relative to PyTorch):

| Engines | 2-step | 4-step | MSE (2 / 4-step) | Action error (mean / worst) |
|---|---|---|---|---|
| BF16 (round 2) | 139.5 ms | 175.8 ms | 0.016717 / 0.017283 | 1.4% / 5.1% |
| INT8 ViT + DiT (round 3) | 105.7 ms | 130.9 ms | 0.017017 / 0.017739 | 2.9% / 5.1% |
| **INT8 ViT + LLM + DiT** | **95.2 ms** | **119.8 ms** | 0.016788 / 0.017686 | 4.2% / 10.1% |

Through `run_gr00t_server.py`, local client, one fixed prompt, 2-step, median of 63
requests:

| Engines | Server | Client | Result |
|---|---|---|---|
| INT8 ViT + DiT | default | one request at a time | 114.5 ms per action (round 3: 117.7 ms) |
| INT8 ViT + DiT | `--pipelined` | two in flight | an action every 110.7 ms, each 221 ms after its frame |
| INT8 ViT + LLM + DiT | default | one request at a time | **104.2 ms per action (9.6 Hz)** |
| INT8 ViT + LLM + DiT | `--pipelined` | two in flight | an action every 101.0 ms, each 202 ms after its frame |
| INT8 ViT + LLM + DiT | `--pipelined` | new frame every 100 ms | **keeps up at 10 Hz, each action 110.6 ms after its frame** (max 111.3) |

With every component in INT8, a client that sends each camera frame as it arrives gets
10 Hz with 111 ms latency. One request at a time gives 9.6 Hz with 104 ms latency. The
ViT + DiT engines cannot keep up with 10 Hz in any mode (a 10 Hz stream through the
pipelined server queues up).

### What changed

1. **INT8 LLM, keeping the MLP down-projections in BF16: 32 → about 21 ms.**
   `build_trt_pipeline.py --int8 vit llm action_head`. With every linear layer in INT8
   the step was 92 ms, but actions were 6.2% off on average and 12-20% on the worst
   frame, while open-loop MSE went *down*. Open-loop MSE alone would have hidden the
   damage. A per-layer study in PyTorch found the cause: layer 2's MLP down-projection
   creates the LLM's "massive activations" (a few channels in the thousands). Quantizing
   that single layer makes the backbone output 91% wrong; every other layer alone is
   under 8%. Keeping all 16 down-projections in BF16 (the default) costs 3 ms and gives
   4.0-4.2% mean action error. Keeping only layers 0-2's MLPs gave about the same error
   (4.3%), so the more conservative set, which does not depend on where this checkpoint
   happens to put its outliers, is the default. The LLM engine shrinks from 1.6 to 1.0 GB.

2. **Prompt cache: 1.5-2 ms per request, identical inputs.** The instruction and image
   layout repeat every step, so the chat template text and the tokenized prompt are
   cached (keyed by text and image grid); only the image processor runs per request.
   `Gr00tPolicy` turns it on; training is unaffected.

3. **No CPU wait after GPU image preprocessing.** On the server's main thread the model's
   stream now waits for the image kernels on the GPU instead of the CPU blocking on them,
   so the remaining CPU preprocessing overlaps them. (Background prefetch threads still
   block, so they cannot stall the frame the main thread is running.) Server preprocessing
   is now 6-7 ms per request with 4 images, most of it OpenCV resizing in the thread pool.

4. **Pipelined server (`run_gr00t_server.py --pipelined`).** A network thread receives and
   preprocesses request N+1 while the main thread runs the model for request N; replies
   go out in order. Clients use `gr00t.policy.server_client.PipelinedPolicyClient`
   (`submit()` / `receive()`); a plain `PolicyClient` still works, one request at a time.
   It raises how often actions can come out, not how fast one comes back: the model is
   GPU-bound, so the overlap only hides the CPU work around it (about 4 ms). Only worth
   using if the robot sends frames at a fixed rate instead of waiting for each action.

### What did not help in round 4

- **Plain INT8 on the whole LLM**: fastest (92 ms), but the action error doubled again.
- **INT8 attention only in the LLM**: 3.7% mean error, but only about 6 ms faster.
- **Moving the remaining preprocessing to the GPU**: what is left is OpenCV's
  `INTER_AREA` resize, which the model was trained with. A GPU copy would change pixels;
  not attempted. Its cost grows with camera resolution (measured here: 180x320 frames).

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

# Faster, opt-in: INT8 engines. 'vit action_head': 106 instead of 140 ms per step here;
# adding 'llm': 95 ms, at about 3x the PyTorch-relative action error of BF16 engines.
# Calibrates on --dataset-path, so use your robot's data, then check verify and an
# open-loop eval, and compare the actions against the BF16 engines on your task.
python scripts/deployment/build_trt_pipeline.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --vit-precision bf16 --fused-denoising-steps 2 --int8 vit llm action_head \
    --output-dir ./gr00t_trt_deployment_int8 --steps export,quantize,build,verify

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
step. Through the server from a local client, with round 3's GPU image preprocessing (on by
default; `--no-gpu-image-preprocessing` turns it off): 151 ms with BF16 engines, 115 ms with
INT8 ViT + DiT, 104 ms with INT8 ViT + LLM + DiT. For a camera that streams at a fixed rate,
start the server with `--pipelined` and send frames with
`gr00t.policy.server_client.PipelinedPolicyClient` (round 4). The server does not seed the sampling noise, so its open-loop MSE varies run
to run.

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
  INT8 engines change the actions more than BF16 ones (round 3) and need the same check.
- The pipelined server needs a client that sends frames without waiting for actions
  (`PipelinedPolicyClient`); the robot client has not been changed or tested with it.
- Remaining server time outside the model is about 8 ms (OpenCV resizing, action decode).
- 6 cameras: engines must be rebuilt for the new image layout. ViT, LLM and preprocessing
  cost grow with the image count; the action head does not.
- The PyTorch wheel from the Orin installer has no sm_87 kernels (it warns at import and
  JIT-compiles). This only affects the PyTorch glue and the no-engine path.

## Future areas of exploration

Ranked by expected gain against the 100 ms server-path budget. The 2-step step currently
splits as ViT 37 ms, fused action head 35 ms, LLM 21 ms, glue about 5 ms, plus about 9 ms
of server preprocessing and decode. Measurements below are from 2026-10-14 on this box.

1. **Action chunking on the robot.** If the robot executes one action per inference, running
   the model every N control steps (execution horizon > 1, or asynchronous chunk execution)
   decouples the 10 Hz control rate from the inference rate entirely. No model change; the
   largest lever if it is not already in use.

2. **Pre-sized frames from the camera pipeline.** Every frame currently goes through
   `SmallestMaxSize(256)` → 95% center crop → `SmallestMaxSize(256)` on the CPU (OpenCV
   `INTER_AREA`, thread pool), then the Qwen image processor rounds it to a multiple of 32
   (256x455 → 256x448 for 16:9, 256x341 → 256x352 for 4:3) with one more resize on the GPU.
   The crop and second resize run unconditionally, so a frame that already matches the
   output size still pays. Measured per request with 4 frames (6 frames cost about the same):

   | Source frame | CPU transform | GPU image step | On the critical path |
   |---|---|---|---|
   | 180x320 (DROID benchmark) | 2.5 ms | 2.7 ms | 5.2 ms |
   | 480x640 | 3.4 ms | ~2 ms | ~5.4 ms |
   | 720x1280 | 6.7 ms | 2.7 ms | 9.4 ms |
   | 1080x1920 | 11.2 ms | 2.7 ms | 13.9 ms |
   | 256x448, transform bypassed | 0 | 1.8 ms | 1.8 ms |

   **Implemented (2026-10-14):** `run_gr00t_server.py --images-pretransformed` /
   `Gr00tPolicy(images_pretransformed=True)` skips the eval transform and passes
   `do_resize=False` to the Qwen image processor (whose resize otherwise resamples even a
   frame already at the target size). Frames must have a short edge of 256 and both sides
   a multiple of 32 (352x256 for 4:3, 448x256 for 16:9); anything else raises `ValueError`.
   Measured on DROID (2 views, 180x320 source, 13 observations, 4 steps, PyTorch BF16): the
   policy's input preparation goes from 6.73 to 4.56 ms; actions with camera-side frames (95%
   center crop + one `INTER_AREA` resize) differ from the standard path by 3.1% mean / 10.4%
   worst, against 8.2% between two noise seeds of the standard path. This source is
   upsampled; a camera downsampling from native resolution should differ less.
   Original plan: an explicit opt-in that
   skips the eval transform and checks the shortest edge is 256; the client (or the camera
   pipeline in hardware: nvvidconv / Argus output resolution on Jetson) delivers the 95%
   center crop resized to shortest edge 256 with the long edge a multiple of 32. Cannot be
   inferred from the frame shape, because the crop is a field-of-view change. One resize in
   the client instead of two is not bit-identical: validate with the round-3 action-error
   check. Also shrinks the request (a 720p frame is 2.7 MB, 256x448 is 0.34 MB). In
   `--pipelined` mode the preprocessing already overlaps the GPU, so the gain there is a
   few ms of latency only. Sending the full field of view without the crop would be a 5%
   zoom-out relative to training; needs an open-loop check before use.

3. **1-step denoising.** About 17 ms (one DiT pass). Quality is untested; cheap to measure
   with the per-step engines (`--denoising-steps 1`) using open-loop MSE and action error.

4. **Image aspect ratio.** A 4:3 source becomes 256x352 (88 tokens per image) instead of
   256x448 (112 tokens): about 20% less ViT and LLM work, roughly 10 ms. Only applicable if
   the cameras are natively 4:3 or that field of view is acceptable; engines must be rebuilt
   for the new token range and the checkpoint re-validated.

5. **INT8 VL self-attention and encoders** (`--int8-action-head-all`): 3.6 ms, already
   built, but doubles the action head's error against BF16 (0.6% → 1.4%). Only after
   validation on the deployed checkpoint.

6. **Builder optimization level 5 plus static shape profiles:** about 2 ms combined (round 2
   measurements), at 2.5x the build time and a rebuild for any change in prompt length.

Checked and not a lever: the fused action head already shares the cross-attention
key/value projections of the vision-language tokens across denoising steps.

### Engine profile (2026-10-14) and the levers it exposes

Per-layer `IProfiler` timing of the INT8 engines, run standalone (no CUDA graph), 461 VL
tokens, 4 images. Engine sums match the 95 ms in-process step (glue is about 1 ms).

| Engine | Total | GEMM (incl. fused epilogues) | Fused attention (BF16) | Norm / quantize / elementwise | Data movement | Notable |
|---|---|---|---|---|---|---|
| ViT (24 blocks) | 37.0 ms | 22.6 ms | 6.3 ms (0.26 ms per block) | 2.9 ms | 3.3 ms | patch-embed `Conv3d` 1.7 ms; rotary kernel 0.13 ms per block (3.1 ms) |
| LLM (16 layers) | 25.8 ms | 17.3 ms | 4.5 ms (0.28 ms per layer) | 2.8 ms | 0.4 ms | BF16 `down_proj` 0.40 ms per layer (6.4 ms); standalone SiLU 1.0 ms |
| Fused action head, 2 steps | 32.4 ms | 26.2 ms | 0.9 ms | 1.8 ms | 2.6 ms | see below |

Action head breakdown: VL self-attention (4 blocks, BF16) 7.5 ms; cross-attention K/V
projections of the VL tokens, computed once for all 16 cross blocks (one batched kernel)
1.7 ms; adaLN linears 0.88 ms per step; embodiment weight gather 0.5 ms; encoders and
decoder about 0.5 ms; the 32 DiT blocks about 9.7 ms per step (about 0.30 ms per block,
of which the two MLP GEMMs are 0.14 ms and run near memory bandwidth).

The INT8 GEMMs with many tokens run at 50-65 TOPS, close to this GPU's dense INT8 peak,
so the remaining time is in things quantization does not touch:

1. **DiT cross-attention is unfused: about 4.5 ms over 2 steps.** The alternate-VL DiT
   gives every cross block a boolean mask (image tokens or text tokens), and TensorRT
   then runs QK^T, a masked softmax with NaN handling, and PV as separate kernels
   (about 165 µs per block plus mask preparation) while the self-attention blocks get
   the fused kernel (10-17 µs). Fix in the export: gather the K/V of the image tokens and
   of the text tokens once (index tensors from the host, where the policy already builds
   `image_mask`) and run unmasked attention on each subset. Same math; no fully-masked
   rows can occur at batch size 1.
2. **adaLN linears from a constant: 1.75 ms over 2 steps, plus 155 MB of weights per
   step.** In the fused graph the timestep is a constant per step, so the 32 blocks'
   `norm1.linear(silu(temb))` and `proj_out_1` are constants too. The quantizer's Q/DQ
   nodes stop TensorRT from folding them, and they run as one M=1, 33x1536x3072 INT8
   GEMM per step (0.88 ms, memory-bound). Precompute scale/shift per block per step in
   the export wrapper. Exact, and removes INT8 noise from these layers.
3. **Patch embedding as a Conv3d: 1.7 ms for a 5.6 GFLOP product (3 TFLOPS).** The
   kernel equals the stride, so it is a plain linear over the already-flattened 1536-wide
   patches. Export it as a MatMul and the quantizer and TensorRT treat it like every
   other linear (about 0.1-0.2 ms). Same math.
4. **ViT rotary embedding: 3.1 ms.** `_apply_rotary_real` upcasts to FP32 and builds
   `cat(-x2, x1)` with slices; Myelin emits a 130 µs kernel per block for a 3.7M-element
   elementwise op. Applying it to q and k in one op, or in BF16, should bring it to
   about 1 ms; BF16 changes numerics slightly (check cosine as in round 2).
5. **LLM `down_proj` in BF16 for all 16 layers: 6.4 ms.** Round 4 measured the same
   action error (4.3% vs 4.2%) with only layers 0-2 kept in BF16, which would save
   about 2.5 ms. Needs the user's checkpoint to decide.
6. **Embodiment weight gather: 0.5 ms.** The encoders and decoder gather the selected
   embodiment's weights from the 32-embodiment stack on every step. The export already
   takes the embodiment tag; selecting the weights at export turns them into plain
   linears (and lets INT8 apply to them without the gather).
7. **Fused attention kernels at about 6 TFLOPS** (ViT 6.3 ms, LLM 4.5 ms). The BF16
   fused MHA kernel TensorRT picks on sm_87 is far from peak at 448-461 tokens. Untested
   whether an INT8 fused MHA (needs Q/DQ on Q, K, V and the softmax) or another
   formulation is faster. Uncertain gain, up to several ms.
8. **Standalone SiLU in the LLM MLP: 1.0 ms.** Myelin fuses the gate*up product into
   the up-projection but leaves SiLU as its own kernel over 461x6144. A fusion choice,
   not directly controllable; worth a look if the MLP is re-exported.

Items 1-6 are export-level changes with no accuracy cost except item 5 and part of
item 4, and add up to roughly 12 ms per step: about 83 ms in-process and 92 ms through
the one-request-at-a-time server, before pre-sized frames.

Measured but ruled out: the cross-attention K/V projections are shared across steps as
intended (48 `to_k` MatMuls: 16 self blocks x 2 steps + 16 cross blocks x 1); the LLM's
deepstack injection runs once, not per layer; clocks were pinned during all measurements
(GPU 1.3 GHz, CPU 2.2 GHz, min = max).
