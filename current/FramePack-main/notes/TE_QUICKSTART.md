# Transformer Engine Quick Start Guide

Quick reference for using Transformer Engine optimization with FramePack on AMD ROCm GPUs.

## TL;DR - For RX 7900 / MI200 / MI300

```bash
# Install Transformer Engine (one-time)
pip install transformer_engine

# Enable TE optimization (works on RX 7900!)
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
python demo_gradio.py

# For MI300+ ONLY: Enable FP8 (RX 7900 doesn't support FP8)
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=1
python demo_gradio.py
```

## What You Get

### RX 7900 / MI200 (without FP8)
✅ **Optimized Linear layers** with better kernel fusion
✅ **Improved memory management**
✅ **Better performance** than standard PyTorch (10-20% faster encoding)
❌ **No FP8** (hardware doesn't support it)

### MI300+ (with FP8)
✅ **All the above PLUS**
✅ **50% less VRAM** for encoders (FP8 vs FP16)
✅ **1.5-2x faster** with native FP8 tensor cores
✅ **Better accuracy** than INT8 quantization

## Installation (Choose One)

### Option A: Quick Install (Recommended)
```bash
pip install transformer_engine
```

### Option B: AMD ROCm Wheels (Specific ROCm version)
```bash
# For ROCm 7.1.1
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine_rocm-2.2.0-py3-none-manylinux_2_28_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine-2.2.0-py3-none-any.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine_torch-2.2.0.tar.gz
pip install ./transformer_engine*.whl transformer_engine_torch-2.2.0.tar.gz --no-build-isolation
```

## Usage

### RX 7900 / MI200 (Optimized kernels, no FP8)

```bash
# Enable TE (FP8 disabled by default for compatibility)
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
python demo_gradio.py
```

### MI300+ (Optimized kernels + FP8)

```bash
# Enable TE with FP8
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=1
python demo_gradio.py
```

### Disable TE (use bitsandbytes or FP16)

```bash
export FRAMEPACK_USE_TRANSFORMER_ENGINE=0
python demo_gradio.py
```

## Check if Working

### RX 7900 / MI200 Output:
```
Transformer Engine: Enabled (AMD ROCm - RX 7900 compatible)
  Optimized kernels without FP8 (RX 7900 doesn't support FP8)
  Linear layers are converted to te.Linear for better performance
  Environment variable: FRAMEPACK_USE_TRANSFORMER_ENGINE=1

  Converting LlamaModel (text_encoder) to Transformer Engine...
  ✓ Converted LlamaModel (text_encoder): 123 Linear layers -> te.Linear
  ✓ Converted CLIPTextModel (text_encoder_2): 56 Linear layers -> te.Linear
  ✓ Converted SiglipVisionModel (image_encoder): 84 Linear layers -> te.Linear
```

### MI300+ with FP8 Output:
```
Transformer Engine: Enabled with FP8 (AMD MI300+)
  This will reduce memory usage and improve performance with native FP8
  Linear layers are converted to te.Linear with FP8 autocast
  Environment variables: FRAMEPACK_USE_TRANSFORMER_ENGINE=1, FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=1

  Created Transformer Engine FP8 recipe (AMD MI300)
  ✓ Converted LlamaModel (text_encoder): 123 Linear layers -> te.Linear
  ...
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FRAMEPACK_USE_TRANSFORMER_ENGINE` | `0` | Enable TE optimizations (works on RX 7900, MI200, MI300) |
| `FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8` | `0` | Enable FP8 mode (**MI300+ only**, RX 7900 doesn't support this) |
| `FRAMEPACK_USE_TRANSFORMER_ENGINE_CACHE` | `1` | Cache converted models for faster startup (recommended) |
| `FRAMEPACK_USE_BITSANDBYTES` | `1` | Automatically disabled when TE is enabled |

### Model Caching

When TE caching is enabled (default), converted models are saved to `.cache_rocm/te_models/`:
- **First run**: Converts models (~10-20 seconds) and saves to cache
- **Subsequent runs**: Loads from cache (~1-2 seconds) - **much faster!**
- Cache expires after 7 days or when models are updated

Disable caching if you want fresh conversion every time:
```bash
export FRAMEPACK_USE_TRANSFORMER_ENGINE_CACHE=0
```

## Compare Performance

Test all options on your hardware:

```bash
# Test 1: Full precision FP16 (baseline)
export FRAMEPACK_USE_BITSANDBYTES=0
export FRAMEPACK_USE_TRANSFORMER_ENGINE=0
python demo_gradio.py
# Generate a video, note: VRAM usage, speed, quality

# Test 2: Bitsandbytes INT8 (current default)
export FRAMEPACK_USE_BITSANDBYTES=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE=0
python demo_gradio.py
# Generate same video, compare VRAM, speed, quality

# Test 3: Transformer Engine (optimized, no FP8 - RX 7900 compatible)
export FRAMEPACK_USE_BITSANDBYTES=0
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0  # Explicitly disable FP8
python demo_gradio.py
# Generate same video, compare VRAM, speed, quality

# Test 4: Transformer Engine with FP8 (MI300+ only)
export FRAMEPACK_USE_BITSANDBYTES=0
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=1
python demo_gradio.py
# MI300+ only: compare with FP8 enabled
```

Monitor VRAM:
```bash
watch -n 1 rocm-smi
```

## Expected Results

### RX 7900 XTX (24GB)

| Configuration | Text Encoder VRAM | Speed | Quality | Notes |
|--------------|-------------------|-------|---------|-------|
| FP16 (baseline) | ~6 GB | 1.0x | 100% | Standard PyTorch |
| Bitsandbytes INT8 | ~3 GB | 0.9x | 98% | Quantized to INT8 |
| **TE (no FP8)** | **~6 GB** | **1.1-1.2x** | **100%** | Optimized kernels, FP16 |

### MI300X (192GB)

| Configuration | Text Encoder VRAM | Speed | Quality | Notes |
|--------------|-------------------|-------|---------|-------|
| FP16 (baseline) | ~6 GB | 1.0x | 100% | Standard PyTorch |
| Bitsandbytes INT8 | ~3 GB | 0.9x | 98% | Quantized to INT8 |
| TE (no FP8) | ~6 GB | 1.1-1.2x | 100% | Optimized kernels, FP16 |
| **TE + FP8** | **~3 GB** | **1.5-2.0x** | **99%** | Native FP8, best option |

*Results approximate, vary by model and settings*

## What Gets Optimized

✅ **LlamaModel** (text prompt encoding) - 100+ layers
✅ **CLIPTextModel** (text prompt encoding) - 50+ layers
✅ **SiglipVisionModel** (image encoding) - 80+ layers
❌ **VAE** (convolutions, not Linear layers)
❌ **Transformer** (custom architecture, may add later)

## When to Use TE

### RX 7900 / MI200 Users:
**Use TE (without FP8) when:**
- ✅ You want better performance without quality loss
- ✅ You prefer optimized kernels over quantization
- ✅ Same VRAM usage as FP16, but faster

**Use Bitsandbytes when:**
- ⚠️ You need to save VRAM (INT8 quantization)
- ⚠️ You're willing to trade speed for memory

### MI300+ Users:
**Use TE with FP8 when:**
- ✅ You want maximum performance
- ✅ You want VRAM savings with minimal quality loss
- ✅ Native FP8 support available

**Use TE without FP8 when:**
- ⚠️ You have plenty of VRAM
- ⚠️ You want maximum quality

## Troubleshooting

### TE not detected
```
Note: Transformer Engine not installed. Using bitsandbytes or full precision.
```
**Fix**: `pip install transformer_engine`

### TE Installed but Not Used
```
Transformer Engine: Requested but not available
```
**Fix**: Check ROCm installation and library paths:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
python -c "import transformer_engine.pytorch as te; print('OK')"
```

### FP8 Error on RX 7900
If you accidentally enable FP8 on RX 7900:
```
Error: FP8 not supported on this GPU
```
**Fix**: Disable FP8:
```bash
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0  # Explicitly disable
```

### Out of Memory
**Fix**: Increase GPU memory preservation in Gradio UI slider:
- Default: 8 GB
- Try: 10-12 GB for 24GB VRAM cards

## Hardware Requirements

### Minimum (TE without FP8):
- **GPU**: AMD RX 7900 XTX/XT, MI200, MI300 series
- **VRAM**: 24GB+ recommended
- **ROCm**: 6.0 or later
- **OS**: Linux (Ubuntu 22.04, RHEL 9, etc.)

### Recommended (TE with FP8):
- **GPU**: AMD MI300X/MI325X/MI350X (FP8 support required)
- **VRAM**: 24GB+ (192GB for full quality)
- **ROCm**: 7.0 or later
- **OS**: Linux

## More Information

See [TRANSFORMER_ENGINE_INTEGRATION.md](TRANSFORMER_ENGINE_INTEGRATION.md) for:
- Detailed technical architecture
- Complete installation instructions
- Benchmarking methodology
- Advanced configuration

---

**Quick Links:**
- [Full Documentation](TRANSFORMER_ENGINE_INTEGRATION.md)
- [TE for ROCm GitHub](https://github.com/ROCm/TransformerEngine)
- [AMD ROCm Docs](https://rocm.docs.amd.com/)
