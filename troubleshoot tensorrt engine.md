#### Pytorch Run
```bash
INFO:root:===============================================================
INFO:root:=== EVALUATION SUMMARY ===
INFO:root:===============================================================
INFO:root:
INFO:root:Metrics:
INFO:root: Average MSE across all trajs: 0.000516
INFO:root: Average MAE across all trajs: 0.008571
INFO:root:
INFO:root:===============================================================
INFO:root:=== DETAILED TIMING SUMMARY ===
INFO:root:===============================================================
INFO:root:
INFO:root:Initialization:
INFO:root: Model loading time: 17.8086s
INFO:root: Dataset loader creation: 0.0029s
INFO:root:
INFO:root:Per-Trajectory Timings (1 trajectories):
INFO:root: Total episode loading: 0.5854s (avg: 0.5854s)
INFO:root: Total data preparation: 0.0327s (avg: 0.0025s per step)
INFO:root: Total inference: 6.3709s (avg: 0.4901s per step)
INFO:root:
INFO:root:Inference Statistics:
INFO:root: Total inference steps: 13
INFO:root: Avg inference time per step: 0.4901s
INFO:root: Min inference time: 0.4218s
INFO:root: Max inference time: 1.0532s
INFO:root: P90 inference time: 0.4841s
INFO:root:
INFO:root:===============================================================
INFO:root:Plot saved to: /workspace/gr00t/episode000_output_pytorch.png
INFO:root:===============================================================
INFO:root:Done
```

#### TensorRT Run
```bash
INFO:root:===============================================================
INFO:root:=== EVALUATION SUMMARY ===
INFO:root:===============================================================
INFO:root:
INFO:root:Metrics:
INFO:root:  Average MSE across all trajs: 0.033599
INFO:root:  Average MAE across all trajs: 0.082764
INFO:root:
INFO:root:===============================================================
INFO:root:=== DETAILED TIMING SUMMARY ===
INFO:root:===============================================================
INFO:root:
INFO:root:Initialization:
INFO:root:  Model loading time:          19.8086s
INFO:root:  Dataset loader creation:     0.0031s
INFO:root:
INFO:root:Per-Trajectory Timings (1 trajectories):
INFO:root:  Total episode loading:       0.5546s  (avg: 0.5546s)
INFO:root:  Total data preparation:      0.0353s  (avg: 0.0027s per step)
INFO:root:  Total inference:             4.5405s  (avg: 0.3493s per step)
INFO:root:
INFO:root:Inference Statistics:
INFO:root:  Total inference steps:       13
INFO:root:  Avg inference time per step: 0.3493s
INFO:root:  Min inference time:          0.2743s
INFO:root:  Max inference time:          0.8617s
INFO:root:  P90 inference time:          0.3745s
INFO:root:
INFO:root:===============================================================
INFO:root:Plot saved to: /workspace/gr00t/episode000_output_tensorrt.png
INFO:root:===============================================================
INFO:root:Done
```


## Analysis: TensorRT Precision Degradation

The investigation reveals several compounding issues causing the ~65x MSE degradation. Let me look at the critical files to confirm the specific details.

Read [standalone_inference_script.py](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/index.html?id=d322687a-281d-4282-aa8d-d247245103f8&parentId=10&origin=3456213c-7244-4194-9344-7a6b4cc547c7&swVersion=4&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&remoteAuthority=ssh-remote%2B192.168.50.201&purpose=webviewView&session=a53f668e-afd7-4b85-883a-038e7a3f1159#)

Read [build_tensorrt_engine.py](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/index.html?id=d322687a-281d-4282-aa8d-d247245103f8&parentId=10&origin=3456213c-7244-4194-9344-7a6b4cc547c7&swVersion=4&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&remoteAuthority=ssh-remote%2B192.168.50.201&purpose=webviewView&session=a53f668e-afd7-4b85-883a-038e7a3f1159#)

Read [export_onnx_n1d6.py](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/index.html?id=d322687a-281d-4282-aa8d-d247245103f8&parentId=10&origin=3456213c-7244-4194-9344-7a6b4cc547c7&swVersion=4&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&remoteAuthority=ssh-remote%2B192.168.50.201&purpose=webviewView&session=a53f668e-afd7-4b85-883a-038e7a3f1159#)

Now let me check how the diffusion denoising loop works and how it interfaces with the DiT.

Read [gr00t_n1d6.py](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/index.html?id=d322687a-281d-4282-aa8d-d247245103f8&parentId=10&origin=3456213c-7244-4194-9344-7a6b4cc547c7&swVersion=4&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&remoteAuthority=ssh-remote%2B192.168.50.201&purpose=webviewView&session=a53f668e-afd7-4b85-883a-038e7a3f1159#) (lines 301-380)

Now I have the full picture. Here's the diagnosis:

## Root Cause: FP16 Precision in the Diffusion Denoising Loop

The core issue is that **BF16 → FP16 conversion loses significant precision**, and those errors **compound across 4 denoising steps** via Euler integration.

### How the precision flows:

1. **PyTorch path** (BF16 throughout):
    
    - DiT model weights are BF16 (from checkpoint)
    - All intermediate computations stay in BF16
    - BF16 has the same exponent range as FP32 (8 bits), just less mantissa precision (7 vs 23 bits)
2. **TensorRT path** (FP16):
    
    - ONNX export happens in BF16 ([export_onnx_n1d6.py:163-166](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/scripts/deployment/export_onnx_n1d6.py#L163-L166))
    - TensorRT engine is built with `--precision bf16` flag ([build_tensorrt_engine.py:102-104](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/scripts/deployment/build_tensorrt_engine.py#L102-L104))
    - **BUT** the output buffer is allocated as FP16 ([standalone_inference_script.py:136-140](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/scripts/deployment/standalone_inference_script.py#L136-L140)):
        
        ```python
        output = torch.empty(tuple(output_shape), dtype=torch.float16, ...)
        ```
        
    - On Orin (SM87), **BF16 tensor cores are not natively supported** — TensorRT may silently fall back to FP16 or FP32 internally
3. **Error accumulation** ([gr00t_n1d6.py:356-357](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/gr00t/model/gr00t_n1d6/gr00t_n1d6.py#L356-L357)):
    
    ```python
    actions = actions + dt * pred_velocity  # Euler integration, 4 iterations
    ```
    
    Each iteration feeds the (imprecise) output back into the DiT as input. Small FP16 errors compound across 4 denoising steps.
    

### Key difference: BF16 vs FP16

|Property|BF16|FP16|
|---|---|---|
|Exponent bits|8|5|
|Mantissa bits|7|10|
|Dynamic range|±3.4×10³⁸|±65504|

BF16 has **much wider dynamic range** — values that are representable in BF16 can overflow or underflow in FP16. The DiT's attention mechanism and layer norms produce values that benefit from that wider range.

### Recommended Fixes (in priority order):

1. **Try building the engine in FP32** — this tells you if the issue is purely precision:
    
    ```bash
    python build_tensorrt_engine.py --onnx ./groot_n1d6_onnx/dit_model.onnx \
        --engine ./groot_n1d6_onnx/dit_model_fp32.trt --precision fp32
    ```
    
    If MSE matches PyTorch, the problem is confirmed as precision loss.
    
2. **Fix the output buffer dtype** — even if the engine runs BF16 internally, outputting to FP16 truncates the result. Change [standalone_inference_script.py:139](vscode-webview://1s56912p2uq8gd5arsn8hsnh0bvtu68kp6td39fcuvdcns520i4e/scripts/deployment/standalone_inference_script.py#L139) to use `torch.float32` or `torch.bfloat16`.
    
3. **Export ONNX in FP32** — export with `use_bf16=False` so the ONNX graph has FP32 weights, then let TensorRT handle the precision policy. This avoids the BF16→ONNX→TensorRT conversion where precision semantics can get lost.
    
4. **Use mixed precision** — keep critical layers (LayerNorm, attention softmax) in FP32 while allowing linear layers to use FP16. This requires TensorRT's layer-precision API.
    

Want me to implement fix #2 (output buffer) and #3 (FP32 export) so you can test them?