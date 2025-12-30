# Bitsandbytes Integration - Final Status

## Implementation: Graceful Fallback System

The current implementation uses a **try-with-fallback** approach that:

✅ Attempts 8-bit quantization if bitsandbytes is available
✅ Automatically falls back to full precision if quantization fails
✅ Only applies to compatible models (transformers models, not diffusers/custom)
✅ Provides clear logging of what succeeded and what failed

## How It Works

### 1. Mock Backend System ([demo_gradio.py:88-96](demo_gradio.py#L88-L96))

Prevents diffusers from crashing when AMD ROCm bitsandbytes is installed:

```python
# Create mock backends that diffusers expects
mock_cpu_module = types.ModuleType('bitsandbytes.backends.cpu')
mock_cpu_module.CPUBackend = type('CPUBackend', (), {})
sys.modules['bitsandbytes.backends.cpu'] = mock_cpu_module

mock_cuda_module = types.ModuleType('bitsandbytes.backends.cuda')
mock_cuda_module.CUDABackend = type('CUDABackend', (), {})
sys.modules['bitsandbytes.backends.cuda'] = mock_cuda_module
```

### 2. Import Real Bitsandbytes ([demo_gradio.py:107-122](demo_gradio.py#L107-L122))

After diffusers is safely imported, try to import the real bitsandbytes:

```python
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
    print("Bitsandbytes available - will attempt quantization with fallback...")
except Exception as e:
    print("Models will load in full precision.")
    HAS_BITSANDBYTES = False
```

### 3. Fallback Loading Function ([demo_gradio.py:271-308](demo_gradio.py#L271-L308))

```python
def load_model_with_fallback(model_class, model_name, subfolder=None,
                              dtype=torch.float16, quantization_config=None):
    """Try quantization, fall back to full precision if it fails."""

    if quantization_config is not None:
        try:
            # Try with 8-bit quantization
            model = model_class.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=dtype
            )
            print(f"✓ {model_class.__name__} loaded with 8-bit quantization")
            return model
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print(f"→ Falling back to full precision")

    # Fall back to full precision
    model = model_class.from_pretrained(model_name, torch_dtype=dtype).cpu()
    return model
```

### 4. Apply to Compatible Models Only

**Models that try quantization** (with automatic fallback):
- `text_encoder` - LlamaModel
- `text_encoder_2` - CLIPTextModel
- `image_encoder` - SiglipVisionModel

**Models that skip quantization** (not supported):
- `vae` - AutoencoderKLHunyuanVideo (diffusers model)
- `transformer` - HunyuanVideoTransformer3DModelPacked (custom model)

## Expected Behavior

### With AMD ROCm Bitsandbytes

When AMD ROCm bitsandbytes is installed:

```
Bitsandbytes available - will attempt quantization with fallback...

Attempting to load models with 8-bit quantization...
Note: Will fall back to full precision if quantization fails for any model.

  Attempting to load LlamaModel with 8-bit quantization...
  ⚠ Quantization failed for LlamaModel: 'ROCmBackend' object has no attribute 'double_quant'
  → Falling back to full precision for LlamaModel

  Attempting to load CLIPTextModel with 8-bit quantization...
  ⚠ Quantization failed for CLIPTextModel: 'ROCmBackend' object has no attribute 'double_quant'
  → Falling back to full precision for CLIPTextModel

  Attempting to load SiglipVisionModel with 8-bit quantization...
  ⚠ Quantization failed for SiglipVisionModel: 'ROCmBackend' object has no attribute 'double_quant'
  → Falling back to full precision for SiglipVisionModel

  Loading VAE (full precision, quantization not supported)...
  Loading Transformer (full precision, custom model)...

Model loading complete.
```

**Result**: All models load in full precision, no crashes, clear error messages.

### Without Bitsandbytes

When bitsandbytes is not installed:

```
Note: bitsandbytes not installed. Models will load in full precision.

Loading models in full precision...

  Loading LlamaModel...
  Loading CLIPTextModel...
  Loading SiglipVisionModel...
  Loading VAE (full precision, quantization not supported)...
  Loading Transformer (full precision, custom model)...

Model loading complete.
```

**Result**: Clean full precision loading, no error messages.

### With CUDA Bitsandbytes (Hypothetical)

If standard CUDA bitsandbytes with full `double_quant` support were installed:

```
Bitsandbytes available - will attempt quantization with fallback...

Attempting to load models with 8-bit quantization...

  Attempting to load LlamaModel with 8-bit quantization...
  ✓ LlamaModel loaded with 8-bit quantization

  Attempting to load CLIPTextModel with 8-bit quantization...
  ✓ CLIPTextModel loaded with 8-bit quantization

  Attempting to load SiglipVisionModel with 8-bit quantization...
  ✓ SiglipVisionModel loaded with 8-bit quantization

  Loading VAE (full precision, quantization not supported)...
  Loading Transformer (full precision, custom model)...

Model loading complete.
```

**Result**: Text/image encoders quantized, VAE/transformer in full precision.

## Environment Variables

- `FRAMEPACK_USE_BITSANDBYTES=1` - Try to use bitsandbytes (default: enabled)
- `FRAMEPACK_USE_BITSANDBYTES=0` - Skip bitsandbytes entirely

Even with `=1`, the system gracefully falls back if quantization fails.

## Advantages of This Approach

1. **✅ No Crashes**: Mock backends prevent diffusers import errors
2. **✅ Automatic Fallback**: Models load even if quantization fails
3. **✅ Clear Logging**: Users see exactly what succeeded/failed
4. **✅ Selective Application**: Only tries on compatible models
5. **✅ Future-Proof**: Will work if AMD fixes `double_quant()` in the future
6. **✅ Cross-Platform**: Same code works with CUDA or ROCm bitsandbytes

## Current AMD ROCm Limitations

As of the current AMD ROCm bitsandbytes version:

❌ Missing `double_quant()` method
❌ Incompatible with transformers' LLM.int8()
❌ Cannot quantize models through transformers API

**All models will fall back to full precision** when using AMD ROCm bitsandbytes.

## Alternative Optimizations (Already Active)

Since quantization doesn't work on AMD ROCm, these optimizations are active:

1. **Torch Compile** - ROCm-optimized JIT compilation
   - Mode: `max-autotune` for ROCm
   - Aggressive kernel optimization
   - Triton backend if available

2. **Model Offloading** - CPU↔GPU swapping in low-VRAM mode
   - DynamicSwapInstaller for fast swapping
   - Only loads active models to GPU
   - Automatic memory management

3. **VAE Tiling/Slicing** - Reduced memory for video processing
   - `vae.enable_tiling()` - Process in tiles
   - `vae.enable_slicing()` - Process slices separately
   - Configurable tile sizes

4. **Memory Preservation** - Adjustable GPU memory headroom
   - Default: 10GB for 20-24GB cards
   - User-adjustable in UI
   - Prevents OOM errors

## Future Improvements

If AMD adds `double_quant()` to ROCm bitsandbytes:

1. **No code changes needed** - Automatic detection and use
2. **Models will quantize successfully** - Fallback won't trigger
3. **Memory savings unlocked** - ~50% reduction for quantized models

Monitor these for updates:
- AMD ROCm bitsandbytes repository
- Transformers library compatibility
- AMD MIGraphX (alternative quantization)

## Conclusion

**Current Status**:
- ✅ Code is production-ready
- ✅ Handles AMD ROCm bitsandbytes gracefully
- ✅ Falls back to full precision automatically
- ✅ Works with or without bitsandbytes installed

**Recommendation**:
Use FramePack with AMD ROCm in full precision mode, leveraging torch.compile and model offloading for optimization. The bitsandbytes integration is ready for when AMD ROCm support improves.

---

**Last Updated**: 2025-12-29
**Status**: Production Ready with Graceful Fallback
**AMD ROCm Compatibility**: Full precision only (quantization falls back)
