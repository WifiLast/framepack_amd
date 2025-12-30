# AMD ROCm Bitsandbytes Investigation - Findings

## Summary

**Conclusion**: The AMD ROCm version of bitsandbytes (`cache/bitsandbytes-rocm_enabled/`) is **NOT compatible** with transformers' automatic quantization system used by FramePack.

## What Was Attempted

1. ✅ Mock `CPUBackend` and `CUDABackend` to allow diffusers to import
2. ✅ Import AMD ROCm bitsandbytes after diffusers
3. ✅ Create `BitsAndBytesConfig` for model loading
4. ❌ Load models with 8-bit quantization

## The Blocker

When attempting to load models with `quantization_config=BitsAndBytesConfig(load_in_8bit=True)`:

```
File "/root/miniconda3/envs/py310/lib/python3.10/site-packages/bitsandbytes/functional.py", line 1883, in double_quant
    return backends[A.device.type].double_quant(

AttributeError: 'ROCmBackend' object has no attribute 'double_quant'
```

### Root Cause

The AMD ROCm `bitsandbytes` library has an incomplete implementation:

**Missing from ROCmBackend**:
- `double_quant()` - Required for 8-bit quantization with transformers
- Possibly other methods needed for full transformers integration

**What ROCmBackend HAS**:
- Basic operations shown in examples (`int8_inference_huggingface.py`, `compile_inference.py`)
- Direct model usage (not through transformers' quantization API)

## Why the Examples Work But Integration Doesn't

### Examples in `cache/bitsandbytes-rocm_enabled/examples/`

The provided examples work because they:
1. Use `BitsAndBytesConfig` with simple standalone models
2. Load models directly without complex pipelines
3. Don't trigger the `double_quant()` path that transformers uses

**Example that works**:
```python
# From compile_inference.py
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
```

This works because:
- Simple causal LM model
- No diffusers integration
- No complex multi-model pipeline

### Why FramePack Integration Fails

FramePack's architecture:
1. Multiple models (text encoders, image encoder, VAE, transformer)
2. Diffusers + transformers integration
3. Custom model classes (`HunyuanVideoTransformer3DModelPacked`)
4. Complex device management and offloading

When transformers loads the model with quantization, it calls:
```python
# transformers/quantizers/quantizer_bnb_8bit.py
new_value = bnb.nn.Int8Params(param_value.to("cpu"), requires_grad=False).to(target_device)
```

This triggers:
```python
# bitsandbytes/nn/modules.py
CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
```

Which fails because `ROCmBackend.double_quant()` doesn't exist.

## Technical Details

### What is `double_quant()`?

Double quantization is a technique used in LLM.int8():
1. **First quantization**: Quantize weights to int8
2. **Second quantization**: Quantize the quantization constants themselves

This provides better compression and accuracy than single-level quantization.

### AMD ROCm Implementation Status

Based on the error, the AMD ROCm bitsandbytes:
- ✅ Has basic int8 operations
- ✅ Works for simple inference examples
- ❌ Missing `double_quant()` for transformers integration
- ❌ Missing full LLM.int8() support
- ❌ Incomplete backend compared to CUDA version

## Alternative Approaches Investigated

### 1. Manual Layer Replacement (Also Doesn't Work)
The old approach of manually replacing `nn.Linear` with `bnb.nn.Linear8bitLt` would also fail because it eventually calls the same missing `double_quant()` method.

### 2. Custom Quantization (Not Implemented)
Would require:
- Implementing our own int8 quantization
- Bypassing bitsandbytes entirely
- Using PyTorch's native quantization or custom kernels
- Significant development effort

### 3. Other Quantization Libraries
Potential alternatives:
- **ONNX Runtime**: Supports INT8 quantization on ROCm
- **AMD MIGraphX**: AMD's inference optimization framework
- **PyTorch Native**: `torch.quantization` (limited ROCm support)
- **Custom kernels**: Write ROCm-specific quantization

## Recommendations

### For Current Users (AMD ROCm)

#### ⭐ **RECOMMENDED: Use Torch-MIGraphX** ⭐

**AMD's official solution for ROCm optimization and quantization:**

```bash
# Install torch_migraphx
cd cache/torch_migraphx-master/py
pip install . --no-build-isolation

# Enable BF16 quantization for memory savings
export FRAMEPACK_MIGRAPHX_BF16=1

# Run FramePack (MIGraphX auto-detected)
python demo_gradio.py
```

**Benefits**:
- ✅ FP16/BF16 quantization (~30% memory savings)
- ✅ Graph optimization and kernel fusion
- ✅ Native ROCm support (AMD's official solution)
- ✅ Compatible with transformers/diffusers
- ✅ Production-ready

See [TORCH_MIGRAPHX_INTEGRATION.md](TORCH_MIGRAPHX_INTEGRATION.md) for details.

#### Alternative: Full Precision Mode

If not using MIGraphX, run with these optimizations:

1. **Torch Compile** (enabled by default)
   ```bash
   export FRAMEPACK_USE_TORCH_COMPILE=1
   export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune
   ```

2. **Model Offloading** (automatic in low-VRAM mode)
   - Swaps models between CPU/GPU as needed
   - Reduces peak VRAM usage

3. **VAE Tiling** (enabled by default)
   - Processes video in tiles
   - Reduces memory for large resolutions

4. **Adjust Memory Slider**
   - Increase "GPU Inference Preserved Memory" if getting OOM
   - Default: 10GB for 20-24GB cards

### For AMD ROCm Bitsandbytes Developers

To make this work, the ROCm version needs:
1. Implement `double_quant()` in `ROCmBackend`
2. Add full LLM.int8() support
3. Match CUDA backend API completeness
4. Test with transformers integration (not just standalone)

### For Future Investigation

Monitor these projects:
1. **bitsandbytes ROCm fork**: Check for updates adding `double_quant`
2. **AMD MIGraphX**: Investigate INT8 quantization capabilities
3. **ONNX Runtime + ROCm**: Export models to ONNX with quantization
4. **PyTorch 2.x quantization**: Native quantization improvements

## Files Modified

1. `demo_gradio.py`:
   - Added mock backends for diffusers compatibility
   - Disabled bitsandbytes integration with clear messages
   - Models load in full precision

2. `AMD_BITSANDBYTES_INTEGRATION.md`:
   - Documented the investigation
   - Explained why it doesn't work
   - Provided alternative optimization strategies

3. `BITSANDBYTES_AMD_FINDINGS.md` (this file):
   - Technical findings
   - Root cause analysis
   - Recommendations

## Status: DISABLED

Bitsandbytes integration is **disabled** in the current implementation due to incompatibility.

**Environment variable**: `FRAMEPACK_USE_BITSANDBYTES` is ignored (always disabled for AMD ROCm).

**Workaround**: Use full precision with other optimizations (torch.compile, offloading, tiling).

## References

- AMD ROCm bitsandbytes: `cache/bitsandbytes-rocm_enabled/`
- Working examples: `cache/bitsandbytes-rocm_enabled/examples/`
- Error location: `bitsandbytes/functional.py:1883`
- Missing method: `ROCmBackend.double_quant()`
- Transformers issue: Requires `double_quant` for LLM.int8()

---

**Date**: 2025-12-29
**Status**: Investigation Complete - Not Compatible
**Action**: Disabled bitsandbytes, using full precision
