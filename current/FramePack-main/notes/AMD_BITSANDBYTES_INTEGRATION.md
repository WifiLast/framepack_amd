# AMD ROCm Bitsandbytes Integration Guide

## ⚠️ IMPORTANT: AMD ROCm Bitsandbytes Limitations

**The AMD ROCm version of bitsandbytes is currently INCOMPATIBLE with transformers quantization.**

The ROCmBackend implementation lacks the `double_quant` method required by transformers' 8-bit quantization:
```
AttributeError: 'ROCmBackend' object has no attribute 'double_quant'
```

**Status**: Bitsandbytes integration is **DISABLED** in the current implementation.

Models load in **full precision** (fp16/bf16) for maximum compatibility with AMD ROCm GPUs.

---

## Overview (Historical - For Reference)

This document explains the attempted integration of AMD ROCm bitsandbytes and why it doesn't work with the current transformers library.

## The Problem

The AMD ROCm version of bitsandbytes has a different architecture than the standard CUDA version:
- **Missing**: `CPUBackend` and `CUDABackend` classes in `bitsandbytes.backends.*`
- **Impact**: Diffusers library expects these backends and crashes on import with:
  ```
  ImportError: cannot import name 'CPUBackend' from 'bitsandbytes.backends.cpu'
  ImportError: cannot import name 'CUDABackend' from 'bitsandbytes.backends.cuda'
  ```

## The Solution

### 1. Mock Backend Classes (Lines 82-96)

Before importing diffusers, we inject mock backend modules into `sys.modules`:

```python
import sys
import types

# Create mock bitsandbytes backend modules
mock_cpu_module = types.ModuleType('bitsandbytes.backends.cpu')
mock_cpu_module.CPUBackend = type('CPUBackend', (), {})
sys.modules['bitsandbytes.backends.cpu'] = mock_cpu_module

mock_cuda_module = types.ModuleType('bitsandbytes.backends.cuda')
mock_cuda_module.CUDABackend = type('CUDABackend', (), {})
sys.modules['bitsandbytes.backends.cuda'] = mock_cuda_module
```

**Why this works**: When diffusers tries to import these backends, it finds our mock modules instead of the missing AMD ROCm ones.

### 2. Import Real Bitsandbytes After Diffusers (Lines 97-117)

After diffusers is safely imported, we import the real AMD ROCm bitsandbytes:

```python
try:
    import importlib.util
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec is not None and bnb_spec.origin and 'bitsandbytes' in bnb_spec.origin:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        HAS_BITSANDBYTES = True
        print("Bitsandbytes loaded successfully (AMD ROCm version)")
except Exception as e:
    print(f"Warning: bitsandbytes import failed: {e}")
    HAS_BITSANDBYTES = False
```

### 3. Use BitsAndBytesConfig for Model Loading (Lines 142-162)

Instead of manually replacing layers, we use the transformers library's built-in quantization:

```python
def get_quantization_config():
    if not USE_BITSANDBYTES or not HAS_BITSANDBYTES or BitsAndBytesConfig is None:
        return None

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    return quantization_config
```

### 4. Apply to Transformers Models Only (Lines 371-414)

Only transformers-based models support `quantization_config`:

**Quantized Models** (8-bit):
- `text_encoder` (LlamaModel)
- `text_encoder_2` (CLIPTextModel)
- `image_encoder` (SiglipVisionModel)

**Full Precision Models**:
- `vae` (AutoencoderKLHunyuanVideo) - diffusers model, no quantization support
- `transformer` (HunyuanVideoTransformer3DModelPacked) - custom model, no quantization support

## Usage

### Enable/Disable Quantization

Set environment variable before running:
```bash
# Enable (default)
export FRAMEPACK_USE_BITSANDBYTES=1

# Disable
export FRAMEPACK_USE_BITSANDBYTES=0
```

Or in code:
```python
os.environ['FRAMEPACK_USE_BITSANDBYTES'] = '1'  # or '0' to disable
```

### Expected Output

When bitsandbytes is enabled:
```
Bitsandbytes loaded successfully (AMD ROCm version)

Loading models with 8-bit quantization (AMD ROCm)...
  Created BitsAndBytesConfig for 8-bit quantization (AMD ROCm)
Transformers models loaded with 8-bit quantization.
8-bit quantization applied to text encoders and image encoder.
VAE and transformer use full precision (quantization not supported).

Bitsandbytes 8-bit Optimization: Enabled (AMD ROCm)
  This will reduce memory usage and may improve performance
  Models are quantized during loading with BitsAndBytesConfig
```

## Key Differences from Original Implementation

### Before (Manual Layer Replacement)
```python
# Old approach - manually replaced nn.Linear with bnb.nn.Linear8bitLt
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Replace with 8-bit version
        int8_linear = bnb.nn.Linear8bitLt(...)
        setattr(parent, attr_name, int8_linear)
```

**Issues**:
- Required custom caching logic
- More complex code
- Needed to track quantization state

### After (BitsAndBytesConfig)
```python
# New approach - use transformers' built-in quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaModel.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Benefits**:
- ✅ Simpler, cleaner code
- ✅ Automatic device placement with `device_map="auto"`
- ✅ Better integration with transformers library
- ✅ Compatible with torch.compile (as shown in AMD examples)
- ✅ No manual caching needed

## Technical Details

### Why Mock Both CPU and CUDA Backends?

The AMD ROCm version of bitsandbytes uses different backend architecture:
- **AMD ROCm**: Uses HIP/ROCm-specific backends
- **Standard CUDA**: Uses `CPUBackend` and `CUDABackend`

Diffusers library checks for both backends during import, so we mock both to prevent import errors.

### Why device_map="auto"?

When using `quantization_config`, transformers requires `device_map` to be set:
- `device_map="auto"` - Automatically places model layers across available devices
- This is required for quantization to work properly
- The library handles moving quantized layers to appropriate devices

### Memory Savings

8-bit quantization provides:
- ~50% memory reduction for quantized models
- Minimal accuracy loss for inference
- Faster inference in some cases (depends on GPU)

**Example** for text_encoder (LlamaModel):
- Full precision (fp16): ~26GB
- 8-bit quantized: ~13GB
- **Savings**: ~13GB VRAM

## Troubleshooting

### If diffusers still fails to import:

Check if additional backends are required:
```python
# Add more mock backends as needed
mock_rocm_module = types.ModuleType('bitsandbytes.backends.rocm')
mock_rocm_module.ROCmBackend = type('ROCmBackend', (), {})
sys.modules['bitsandbytes.backends.rocm'] = mock_rocm_module
```

### If quantization fails:

1. Check bitsandbytes is installed: `pip list | grep bitsandbytes`
2. Disable quantization temporarily: `export FRAMEPACK_USE_BITSANDBYTES=0`
3. Check error messages during model loading
4. Verify AMD ROCm version compatibility

### If models run out of memory:

Even with quantization, you may need to:
1. Enable model offloading (already done in low-VRAM mode)
2. Reduce batch sizes
3. Enable VAE tiling (already enabled by default)
4. Increase GPU memory preservation parameter in the UI

## References

- AMD ROCm bitsandbytes examples: `cache/bitsandbytes-rocm_enabled/examples/`
- Transformers quantization docs: https://huggingface.co/docs/transformers/main/en/quantization
- Original implementation: See git history for old `apply_bitsandbytes_8bit()` function

## Compatibility

- ❌ AMD ROCm bitsandbytes (incompatible with transformers quantization)
- ✅ NVIDIA CUDA bitsandbytes (works with standard version)
- ✅ torch.compile integration
- ✅ Dynamic model offloading
- ✅ Full precision models on AMD ROCm

## Alternative Memory Optimization Strategies for AMD ROCm

Since bitsandbytes quantization is not available, use these alternatives:

### 1. **Model Offloading (Already Enabled)**
```python
# Low VRAM mode automatically uses DynamicSwapInstaller
if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
```

### 2. **VAE Tiling (Already Enabled)**
```python
vae.enable_tiling()  # Processes images in tiles to reduce memory
vae.enable_slicing()  # Processes slices separately
```

### 3. **Torch Compile Optimizations**
Already enabled with ROCm-specific settings:
```bash
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune
```

### 4. **Adjust GPU Memory Preservation**
In the Gradio UI, increase the "GPU Inference Preserved Memory" slider:
- Default: 10GB for 20-24GB VRAM cards
- Higher values = slower but prevents OOM errors
- Lower values = faster but may crash

### 5. **Use FP16 Instead of BF16**
Some models can use fp16 instead of bfloat16 to save memory (already done for most models).

### 6. **Future: ONNX/TensorRT Quantization**
Consider using ONNX Runtime or AMD's ROCm-optimized frameworks for quantization:
- ONNX Runtime with ROCm backend
- AMD's MIGraphX for INT8 quantization
- Custom quantization implementations

## Why AMD ROCm Bitsandbytes Doesn't Work

The AMD ROCm version of bitsandbytes in `cache/bitsandbytes-rocm_enabled/` is designed for:
1. **Direct usage** with custom models (as shown in examples)
2. **Manual layer replacement** (not transformers' automatic quantization)

It lacks:
- `double_quant()` method for dual quantization
- Full compatibility with transformers' quantization API
- CPU/CUDA backend classes that diffusers expects

The examples in `cache/bitsandbytes-rocm_enabled/examples/` show standalone inference, not integration with diffusers/transformers pipelines like FramePack uses.

## Conclusion

For now, **AMD ROCm users should run FramePack in full precision mode** with:
- ✅ Torch compile optimizations
- ✅ Model offloading
- ✅ VAE tiling/slicing
- ✅ ROCm-specific performance tuning

This provides good performance without quantization. Future AMD ROCm bitsandbytes updates may add the missing `double_quant` support.
