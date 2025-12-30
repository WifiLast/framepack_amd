# Transformer Engine Integration for FramePack

This document describes the integration of AMD Transformer Engine (TE) with FramePack for FP8 optimization on AMD MI300 GPUs.

## Overview

Transformer Engine provides native FP8 (8-bit floating point) support on AMD MI300 GPUs, offering:
- **Memory savings**: ~50% reduction compared to FP16
- **Performance improvements**: Faster computation with FP8 tensor cores
- **Better accuracy**: Native FP8 vs. quantized INT8 (bitsandbytes)

The integration converts text encoders (LlamaModel, CLIPTextModel, SiglipVisionModel) to use `te.Linear` layers with FP8 autocast during inference.

## Installation

### Prerequisites
- AMD ROCm 7.0+ (for MI300 GPU support)
- PyTorch for ROCm
- AMD GPU with FP8 support (MI300, MI325, MI350)

### Install Transformer Engine

#### Option 1: From manylinux wheels (recommended)
```bash
# Download wheels for ROCm 7.1.1
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine_rocm-2.2.0-py3-none-manylinux_2_28_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine-2.2.0-py3-none-any.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/transformer_engine_torch-2.2.0.tar.gz

# Install (PyTorch extension only)
pip install ./transformer_engine_rocm-2.2.0-py3-none-manylinux_2_28_x86_64.whl
pip install ./transformer_engine-2.2.0-py3-none-any.whl
pip install ./transformer_engine_torch-2.2.0.tar.gz --no-build-isolation
```

#### Option 2: From source
```bash
# Clone the repository
git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine

# Set environment variables
export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH=gfx942  # for MI300/MI325
export NVTE_USE_ROCM=1

# Install
pip install . --no-build-isolation
```

### Verify Installation
```python
import transformer_engine.pytorch as te
print(f"Transformer Engine installed: {te.__version__}")
```

## Usage

### Enable Transformer Engine

Set the environment variable before running:
```bash
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
python demo_gradio.py
```

### Disable Bitsandbytes (automatic)

When Transformer Engine is enabled, bitsandbytes is automatically disabled as they are mutually exclusive:
```bash
# This combination will use TE and disable bitsandbytes
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_BITSANDBYTES=1  # Will be ignored
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEPACK_USE_TRANSFORMER_ENGINE` | `0` | Enable TE FP8 optimization |
| `FRAMEPACK_USE_BITSANDBYTES` | `1` | Enable bitsandbytes (disabled if TE is enabled) |

## What Gets Optimized

### Models Converted to TE
The following models have their `nn.Linear` layers replaced with `te.Linear`:

1. **LlamaModel** (text_encoder)
   - All Linear layers in the transformer blocks
   - ~100+ layers converted
   - Used during text prompt encoding

2. **CLIPTextModel** (text_encoder_2)
   - All Linear layers in CLIP attention and MLP blocks
   - ~50+ layers converted
   - Used during text prompt encoding

3. **SiglipVisionModel** (image_encoder)
   - All Linear layers in vision transformer
   - ~80+ layers converted
   - Used during CLIP vision encoding

### FP8 Autocast Applied

FP8 autocast is applied during:
- **Text encoding**: Prompt and negative prompt encoding (2x per generation)
- **CLIP vision encoding**: Input image encoding (1x per generation)

### Models NOT Optimized

These models are not converted to TE (limited benefit):
- **VAE** (AutoencoderKLHunyuanVideo): Uses convolutions, not Linear layers
- **Transformer** (HunyuanVideoTransformer3DModelPacked): Custom architecture, needs evaluation

## Performance Expectations

### Memory Savings
- **Text Encoders**: ~2-3 GB VRAM savings (FP8 vs FP16)
- **Total**: ~25-30% reduction in encoder memory footprint

### Speed Improvements
- **Text Encoding**: 1.2-1.5x faster (FP8 tensor cores)
- **CLIP Vision**: 1.2-1.3x faster
- **Overall**: Minimal impact on total generation time (encoders are <5% of total)

### Accuracy
- **FP8 precision**: Minimal quality degradation (E4M3 format for activations)
- **Better than INT8**: More dynamic range than bitsandbytes quantization

## Comparison: Transformer Engine vs Bitsandbytes

| Feature | Transformer Engine | Bitsandbytes |
|---------|-------------------|--------------|
| Precision | FP8 (E4M3/E5M2) | INT8 |
| GPU Support | AMD MI300+ only | AMD + NVIDIA |
| Memory Savings | ~50% (FP16→FP8) | ~50% (FP16→INT8) |
| Performance | Native FP8 ops | Quantized ops |
| Accuracy | Better (native FP8) | Good (quantized) |
| AMD ROCm | Native support | Limited support |

## Troubleshooting

### TE not detected
```
Note: Transformer Engine not installed. Using bitsandbytes or full precision.
```
**Solution**: Install Transformer Engine (see Installation section)

### TE requested but models not converted
Check the startup output:
```
Transformer Engine FP8 Optimization: Enabled (AMD MI300)
  Converting LlamaModel (text_encoder) to Transformer Engine...
  ✓ Converted LlamaModel (text_encoder): 123 Linear layers -> te.Linear
```

If you see `✓ Converted`, TE is working correctly.

### Import errors
```python
ImportError: libamdhip64.so.6: cannot open shared object file
```
**Solution**: Ensure ROCm 7.0+ is installed and `LD_LIBRARY_PATH` includes ROCm libraries:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Out of memory with TE
TE uses FP8, which should reduce memory. If OOM occurs:
1. Increase `gpu_memory_preservation` slider in Gradio UI
2. Check if other processes are using GPU memory
3. Verify GPU has sufficient memory (24GB+ recommended)

## Technical Details

### FP8 Recipe Configuration
```python
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,      # E4M3 for fwd, E5M2 for bwd
    amax_history_len=16,           # Shorter for inference
    amax_compute_algo="max",       # Stable scaling
    override_linear_precision=(False, False, False)  # Auto precision
)
```

### Linear Layer Conversion
```python
# Before (PyTorch)
nn.Linear(in_features=4096, out_features=4096, bias=True)

# After (Transformer Engine)
te.Linear(in_features=4096, out_features=4096, bias=True)
```

### FP8 Autocast Context
```python
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input)  # Runs in FP8
```

## Files Modified

1. **demo_gradio.py**:
   - Import TE and recipe classes
   - Add `USE_TRANSFORMER_ENGINE` env flag
   - Implement `convert_model_to_te()` function
   - Implement `get_te_fp8_recipe()` function
   - Convert text/image encoders after loading
   - Wrap encoding calls with `te.fp8_autocast()`

2. **te_text_encoder_wrapper.py** (standalone utility):
   - Reusable TE conversion functions
   - Can be used independently of demo_gradio.py

## Benchmarking

To compare TE vs bitsandbytes vs FP16:

```bash
# Full precision (FP16)
export FRAMEPACK_USE_BITSANDBYTES=0
export FRAMEPACK_USE_TRANSFORMER_ENGINE=0
python demo_gradio.py

# Bitsandbytes (INT8)
export FRAMEPACK_USE_BITSANDBYTES=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE=0
python demo_gradio.py

# Transformer Engine (FP8)
export FRAMEPACK_USE_BITSANDBYTES=0
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
python demo_gradio.py
```

Monitor:
- VRAM usage: `rocm-smi` or `watch -n 1 rocm-smi`
- Generation time: Gradio UI progress
- Quality: Visual comparison of outputs

## References

- [Transformer Engine for ROCm](https://github.com/ROCm/TransformerEngine)
- [TE Documentation](cache/TransformerEngine-dev/README.rst)
- [AMD MI300 FP8 Support](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [FP8 Format Primer](cache/TransformerEngine-dev/examples/README.md)

## Future Work

- [ ] Evaluate TE for HunyuanVideoTransformer (if compatible)
- [ ] Add TE support for attention mechanisms (Flash Attention on AMD)
- [ ] Benchmark TE vs MIGraphX torch.compile
- [ ] Support TE with model parallelism (multi-GPU)
- [ ] Create automated benchmark script

## Support

For issues related to:
- **TE installation**: See [cache/TransformerEngine-dev/README.rst](../../../cache/TransformerEngine-dev/README.rst)
- **FramePack integration**: Check this document
- **AMD ROCm**: Visit [ROCm documentation](https://rocm.docs.amd.com/)

---

**Last Updated**: 2025-12-29
**TE Version Tested**: 2.2.0
**ROCm Version**: 7.1.1
