# FPS, Action Horizon, and Temporal Alignment in GR00T Inference

## 1. What FPS Means During Finetuning

**FPS is a dataset property, not a training hyperparameter.** The finetune script (`examples/SO100/finetune_so100.sh`) has no `--fps` argument. FPS is baked into the dataset's `meta/info.json` at data collection time and is never read by the training loop.

The `alfiebot.CanDoChallenge` dataset is recorded at **15 FPS**.

The model has no concept of wall-clock time. It learns: "given this image + joint state, the next 16 frames of joint positions looked like this." Whether those 16 frames span 1.07 seconds (15 FPS) or 0.53 seconds (30 FPS) is entirely determined by the recording rate of the training data.

---

## 2. How Delta Indices Map to Frames

The action horizon is defined by `delta_indices` in the modality config (`examples/SO100/so100_config.py`):

```python
"action": ModalityConfig(
    delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
)
```

Each delta index = **one frame in the dataset**. During training, for a given observation at frame `N`:

- `action[0]` = joint positions at frame `N` (current)
- `action[1]` = frame `N+1` (66.7ms later at 15 FPS)
- `action[2]` = frame `N+2` (133ms later)
- ...
- `action[15]` = frame `N+15` (1000ms later)

Each "step" is literally one row in the parquet file — one observation/action sample from the robot at the data collection rate. The model learns to predict the next 16 consecutive frames of joint positions given the current image + state.

**Temporal spacing between action steps:**

```
time_per_step = 1 / dataset_fps
```

| Dataset FPS | Time per step | 16-step horizon | 8-step execution |
|-------------|---------------|-----------------|------------------|
| 30 FPS      | 33.3ms        | 0.53s           | 0.27s            |
| 15 FPS      | 66.7ms        | 1.07s           | 0.53s            |
| 10 FPS      | 100ms         | 1.60s           | 0.80s            |
| 5 FPS       | 200ms         | 3.20s           | 1.60s            |

---

## 3. The 16-Step Action Horizon

The model always produces 16 action steps in a single forward pass (`gr00t/configs/model/gr00t_n1d6.py:58`):

```python
action_horizon: int = 16
```

The generation process (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py:327-385`):
1. Initialize random noise: `(batch, 16, action_dim)` in FP32
2. Run 4 denoising (flow-matching) iterations via Euler integration
3. Each iteration updates **all 16 steps simultaneously**: `actions = actions + dt * pred_velocity`

The `dt = 0.25` in the denoising loop is the **diffusion timestep** (1/4 steps), NOT an action temporal spacing. This has nothing to do with the 66.7ms between action steps.

---

## 4. Inference + Execution Timeline

The SO100 eval loop (`gr00t/eval/real_robot/SO100/eval_so100.py:212-244`) is fully sequential — no pipelining:

```
[Observe] → [Inference (blocking)] → [Execute 0] → [Execute 1] → ... → [Execute 7] → [Observe] → ...
```

### On Orin AGX at ~4 Hz (250ms inference)

```
Time 0ms:       Capture observation
Time 0-250ms:   Model inference (robot is IDLE)
Time 250ms:     Execute action[0]  ← predicted for time 0ms, now 250ms stale
Time 317ms:     Execute action[1]  ← predicted for time 67ms
Time 383ms:     Execute action[2]  ← predicted for time 133ms
Time 450ms:     Execute action[3]  ← predicted for time 200ms
Time 517ms:     Capture next observation (if executing 4 steps)
```

### Three temporal mismatches

| Issue | Description | Magnitude |
|-------|-------------|-----------|
| **Inference delay** | action[0] executes 250ms after the observation was taken, but was predicted for "right now" | 250ms stale |
| **Execution rate** | SO100 eval hardcodes 30 Hz (`1.0/30`) but 15 FPS data needs 67ms/step | 2x too fast |
| **Observation staleness** | By action[7], the observation is 717ms old | 717ms stale |

**The execution rate must match the training FPS.** Change `1.0 / 30` → `1.0 / dataset_fps` in the eval script.

---

## 5. Do You Need to Match Data Collection FPS to Inference Speed?

**No.** Higher FPS data gives finer-grained action resolution (smoother movements). The inference rate only determines how often you refresh observations.

The key relationship: execute `ceil(inference_latency * fps)` steps per call to cover the inference time for pipelining, then discard the rest.

| Scenario | Steps to cover inference | Execution time | Observation rate |
|----------|------------------------|----------------|------------------|
| 15 FPS, 4 Hz inference (250ms) | ceil(250/67) = **4 steps** | 267ms | ~3.7 Hz (pipelined) |
| 15 FPS, 5.8 Hz inference (173ms) | ceil(173/67) = **3 steps** | 200ms | ~5 Hz (pipelined) |
| 4 FPS, 4 Hz inference (250ms) | ceil(250/250) = **1 step** | 250ms | ~4 Hz (pipelined) |

At 15 FPS with 4 Hz inference, you execute ~4 fine-grained steps while the next inference runs. At 4 FPS you'd only execute 1 coarse step — same throughput, but jerkier movement and no benefit from the 16-step planning horizon.

**Collect data as fast as your hardware allows for maximum action granularity.** The "waste" of unused later predictions isn't really waste — those later steps are increasingly unreliable anyway because they're conditioned on a single stale observation.

---

## 6. Strategies for Temporal Alignment

### Strategy A: Accept the latency gap (simplest)

Execute actions at training FPS (15 Hz = 67ms/step). Accept 250ms dead time during inference. action[0] will be slightly stale but temporal spacing is correct.

- Total cycle: 250ms inference + 4 × 67ms execution = **517ms** per refresh (~1.9 Hz obs rate)
- Pros: Temporally correct, simple
- Cons: 250ms of no motion during inference

### Strategy B: Skip early actions to compensate for latency

After inference completes (250ms), skip the first actions that are now in the past:

```
skip_count = floor(inference_latency * fps) = floor(0.250 * 15) = 3 actions
```

Execute `actions[3:7]` at 67ms spacing. action[3] was predicted for t=200ms, which is close to the actual elapsed time (250ms).

- Pros: Better temporal alignment
- Cons: Wastes some predictions

### Strategy C: Pipeline inference with execution (best)

The codebase already has infrastructure for this (`gr00t/policy/gr00t_policy.py:412-458`):

- `prepare_inputs()` — CPU-only preprocessing, safe for background thread
- `run_inference()` — GPU inference from pre-computed inputs

```
Thread 1 (execution):     Execute actions[0:4] at 67ms each (267ms total)
Thread 2 (inference):     prepare_inputs() + run_inference() for next observation (250ms)

Timeline:
[Infer N]──────────→ [Execute 4 steps from N] ──→ [Execute 4 steps from N+1] ──→
                      [Infer N+1]──────────────→   [Infer N+2]──────────────→
```

With 4 Hz inference and 4-step execution (267ms), inference (250ms) finishes before execution completes — continuous motion with no gaps.

- Observation rate: ~3.7 Hz
- Pros: No dead time, continuous motion
- Cons: Requires threading

### Strategy D: Adjust execution horizon dynamically

Execute fewer steps for fresher observations vs. more steps for smoother motion:

| Execution steps | Execution time (15 FPS) | Obs refresh (pipelined, 4 Hz) | Character |
|----------------|------------------------|-------------------------------|-----------|
| 2 steps        | 133ms                  | ~4 Hz (inference-bound)       | Most reactive, potentially choppy at chunk boundaries |
| 4 steps        | 267ms                  | ~3.7 Hz                       | Good balance for 4 Hz inference |
| 8 steps        | 533ms                  | ~1.9 Hz                       | Smooth within chunk, stale at end |
| 16 steps       | 1067ms                 | ~0.9 Hz                       | Full open-loop, very stale |

---

## 7. Key Code Locations

| Concept | File | Line |
|---------|------|------|
| Dataset FPS | `gr00t/data/dataset/lerobot_episode_loader.py` | 182 |
| Action delta_indices (SO100) | `examples/SO100/so100_config.py` | 25-42 |
| Model action_horizon (16) | `gr00t/configs/model/gr00t_n1d6.py` | 58 |
| Denoising loop (4 Euler steps) | `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | 349-385 |
| Execution rate (hardcoded 30Hz) | `gr00t/eval/real_robot/SO100/eval_so100.py` | 243 |
| Eval execution horizon (8) | `gr00t/eval/real_robot/SO100/eval_so100.py` | 170 |
| Pipeline-ready API | `gr00t/policy/gr00t_policy.py` | 412-458 |
| Training data extraction | `gr00t/data/dataset/sharded_single_step_dataset.py` | 13-63 |

---

## 8. Summary

1. **FPS is not a tuning knob** — it's fixed by how data was collected (15 FPS for this dataset)
2. **Each action step = one frame** at the dataset FPS (66.7ms at 15 FPS)
3. **Execution rate must match training FPS** — change `1.0/30` → `1.0/15` in the eval script
4. **Don't downsample data to match inference speed** — higher FPS gives smoother actions; execute ~4 steps per inference call and discard the rest
5. **Inference latency creates observation staleness** — inherent to the architecture; mitigate with pipelining or action skipping
6. **Execute fewer steps for fresher observations** — 4 steps at 15 FPS (267ms) covers a 250ms inference cycle perfectly for pipelining
7. **Pipelining infrastructure already exists** — `prepare_inputs()` + `run_inference()` in `gr00t_policy.py`
