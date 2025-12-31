# Bitsandbytes 8-bit Optimization Guide

This document explains how to use bitsandbytes 8-bit quantization to reduce memory usage and potentially improve performance in FramePack.

## Overview

Bitsandbytes is a library that provides 8-bit matrix multiplication and quantization for PyTorch models. By converting model weights from 16-bit (FP16/BF16) to 8-bit integers, you can:

- **Reduce VRAM usage by ~50%** for model weights
- **Reduce RAM usage by ~50%** when models are offloaded to CPU
- **Potentially improve inference speed** on some hardware
- **Enable running larger models** on GPUs with limited VRAM

## Installation

First, install bitsandbytes:

```bash
pip install bitsandbytes
```

For AMD GPUs (ROCm), you may need a ROCm-compatible version:

```bash
pip install bitsandbytes-rocm
```

## Usage

### Enable 8-bit Optimization

Set the environment variable before running the script:

```bash
# On Linux/Mac
export FRAMEPACK_USE_BITSANDBYTES=1
python demo_gradio.py

# On Windows (PowerShell)
$env:FRAMEPACK_USE_BITSANDBYTES=1
python demo_gradio.py

# On Windows (CMD)
set FRAMEPACK_USE_BITSANDBYTES=1
python demo_gradio.py
```

### Configuration Options

#### FRAMEPACK_USE_BITSANDBYTES
- **Default:** `0` (disabled)
- **Values:** `0` or `1`
- **Description:** Enable/disable 8-bit quantization

```bash
export FRAMEPACK_USE_BITSANDBYTES=1
```

#### FRAMEPACK_BNB_THRESHOLD
- **Default:** `6144`
- **Values:** Any positive integer
- **Description:** Minimum number of parameters for a layer to be quantized

```bash
export FRAMEPACK_BNB_THRESHOLD=6144  # Quantize layers with 6144+ parameters
```

Lower thresholds quantize more layers (more memory savings but potentially lower quality).
Higher thresholds quantize fewer layers (less memory savings but better quality).

## How It Works

### First Run (Quantization)

When you enable bitsandbytes for the first time:

1. Models are loaded normally from HuggingFace
2. Linear layers are identified and quantized to 8-bit
3. Quantized models are cached to `.cache_rocm/bitsandbytes_models/`
4. You'll see output like:

```
Applying bitsandbytes 8-bit optimization to models...
  Quantizing LlamaModel Text Encoder to 8-bit (first time, will be cached)...
    ✓ Converted 512 Linear layers to 8-bit
    Total parameters quantized: 4,294,967,296
    💾 Cached quantized model to LlamaModel_Text_Encoder_8bit.pt
```

### Subsequent Runs (Cached)

On subsequent runs:

1. Cached quantized models are loaded directly
2. No re-quantization needed (much faster startup)
3. You'll see output like:

```
Applying bitsandbytes 8-bit optimization to models...
    ✓ Loaded cached 8-bit model for LlamaModel Text Encoder (2.3 days old)
```

### Cache Management

- **Location:** `.cache_rocm/bitsandbytes_models/`
- **Expiration:** 30 days
- **Invalidation:** Automatic if threshold changes

To clear cache and re-quantize:

```bash
rm -rf .cache_rocm/bitsandbytes_models/
```

## Which Models Are Optimized?

All major models are quantized when enabled:

1. **LlamaModel Text Encoder** - Reduces VRAM/RAM by ~2-4GB
2. **CLIP Text Encoder** - Reduces VRAM/RAM by ~200-500MB
3. **VAE (Autoencoder)** - Reduces VRAM/RAM by ~100-300MB
4. **Siglip Image Encoder** - Reduces VRAM/RAM by ~300-600MB
5. **Hunyuan Transformer** - Reduces VRAM/RAM by ~4-8GB (largest savings)

## Performance Considerations

### Memory Savings

Expected VRAM/RAM savings with default settings:

- **Text Encoders:** ~2-4GB
- **Transformer:** ~4-8GB
- **VAE + Image Encoder:** ~500MB-1GB
- **Total:** ~7-13GB savings

### Speed Impact

- **Positive:** Lower memory pressure may reduce swap/OOM issues
- **Neutral:** 8-bit computation is well-optimized on modern GPUs
- **Negative:** Small overhead for quantization on first run

### Quality Impact

- **Minimal:** 8-bit quantization with bitsandbytes preserves quality very well
- **Testing recommended:** Run with and without to compare outputs
- **Threshold tuning:** Increase `FRAMEPACK_BNB_THRESHOLD` if you notice quality degradation

## Compatibility

### GPU Support

- **NVIDIA (CUDA):** ✅ Fully supported
- **AMD (ROCm):** ✅ Supported with `bitsandbytes-rocm`
- **Intel/CPU:** ⚠️ Limited support, may be slower

### Combining with Other Optimizations

Bitsandbytes works well with other optimizations:

- **✅ Torch Compile:** Can be used together (apply quantization first)
- **✅ VAE Tiling:** Compatible
- **✅ Model Offloading:** Compatible (saves RAM when offloaded)
- **✅ TeaCache:** Compatible

## Troubleshooting

### Error: "bitsandbytes not installed"

```bash
pip install bitsandbytes
# or for AMD
pip install bitsandbytes-rocm
```

### Error: "CUDA runtime not found"

Make sure CUDA/ROCm is properly installed and accessible.

### Quality Degradation

Try increasing the threshold to quantize fewer layers:

```bash
export FRAMEPACK_BNB_THRESHOLD=10240  # Only quantize larger layers
```

### Slow Startup on First Run

This is normal - quantization takes time. Subsequent runs will be fast thanks to caching.

### Cache Issues

Clear the cache and let it rebuild:

```bash
rm -rf .cache_rocm/bitsandbytes_models/
```

## Example Configurations

### Maximum Memory Savings (Low VRAM)

```bash
export FRAMEPACK_USE_BITSANDBYTES=1
export FRAMEPACK_BNB_THRESHOLD=4096  # Quantize more layers
python demo_gradio.py
```

### Balanced (Recommended)

```bash
export FRAMEPACK_USE_BITSANDBYTES=1
export FRAMEPACK_BNB_THRESHOLD=6144  # Default
python demo_gradio.py
```

### Quality Priority (High VRAM)

```bash
export FRAMEPACK_USE_BITSANDBYTES=1
export FRAMEPACK_BNB_THRESHOLD=16384  # Only quantize very large layers
python demo_gradio.py
```

## Technical Details

### Quantization Method

- **Algorithm:** Linear8bitLt (8-bit matrix multiplication with outlier handling)
- **Outlier threshold:** 6.0 (preserves important outlier values)
- **Weights:** INT8 format
- **Activations:** FP16/BF16 (not quantized)

### Cache Format

Cached models include:
- Quantized state_dict
- Timestamp (for expiration)
- Threshold (for validation)

### Implementation

The implementation uses bitsandbytes' `Linear8bitLt` layer, which:
1. Quantizes weights to 8-bit integers
2. Maintains outlier values in FP16 for accuracy
3. Uses optimized CUDA/ROCm kernels for 8-bit matmul
4. Automatically handles mixed-precision computation

## References

- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [8-bit Optimizers Paper](https://arxiv.org/abs/2110.02861)
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)

## Support

If you encounter issues:

1. Check that bitsandbytes is properly installed
2. Verify CUDA/ROCm compatibility
3. Try clearing the cache
4. Adjust the threshold parameter
5. Report issues with detailed error messages

---

**Note:** This optimization is **disabled by default** to maintain maximum quality. Enable it when you need to reduce memory usage or run on GPUs with limited VRAM.
