# GR00T N1.7 Inference Optimization on Jetson AGX Orin

Follow-up to `INFERENCE_OPTIMIZATIONS.md` (N1.6 / Eagle backbone). Same goal — control
rate on an AGX Orin 64GB — redone for N1.7 (Qwen3-VL / Cosmos-Reason2-2B backbone).

**Result: 480 ms → 304 ms per step (1.6x) at unchanged accuracy, 250 ms (1.9x) with
2-step denoising.** Measured on 2026-10-12.

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

## Recommended configuration

```bash
source scripts/activate_orin.sh

# One-time: export, build, and verify engines (~7 min on AGX Orin). Check the reported cosine.
python scripts/deployment/build_trt_pipeline.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --vit-precision bf16 --steps export,build,verify

# Inference / benchmarking
python scripts/deployment/standalone_inference_script.py \
    --model-path <checkpoint> --dataset-path <dataset> --embodiment-tag <TAG> \
    --inference-mode trt_full_pipeline --trt-engine-path ./gr00t_trt_deployment/engines \
    --execution-horizon 8 --denoising-steps 2
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

Measured with these (2 DROID trajectories, 2-step): `open_loop_eval.py` in-process 246 ms per
step; through the server from a local client 279–281 ms (the difference is shipping 4 images
per request over the socket). The server does not seed the sampling noise, so its open-loop
MSE varies run to run.

No-engine alternative (longer warmup on first inference, ~8% slower):

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
  base model on DROID demo data).
- Async prefetch and pipelining only exist in the two offline scripts; the server handles one
  request at a time.
- The DROID action-decode cost (17–48 ms of CPU per step) is untouched.
- Fewer cameras or lower image resolution would cut ViT time roughly proportionally; that is
  a data/fine-tuning decision, not an inference switch.
