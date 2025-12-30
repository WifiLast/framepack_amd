# tritonBLAS Integration for FramePack AMD

This document describes the tritonBLAS integration for optimizing matrix multiplication operations on AMD ROCm GPUs (RX 7900, MI200, MI300).

## What is tritonBLAS?

tritonBLAS is a lightweight Triton-based GEMM (General Matrix Multiplication) library that:

- **Uses analytical models** instead of autotuning to select optimal kernel configurations
- **Provides predictable performance** without greedy search overhead
- **Optimized for AMD GPUs** including RX 7900 XTX/XT, MI200, MI300 series
- **Supports multiple data types**: FP16, BF16, FP32, FP8 (MI300+), FP4 quantization
- **Works without FP8** - fully compatible with RX 7900 which lacks hardware FP8 support

## Why Use tritonBLAS?

### Benefits:

1. **Faster kernel selection** - No autotuning overhead on first run
2. **Optimized for AMD architecture** - Better utilization of AMD GPU compute units
3. **Predictable performance** - Analytical model explains all kernel selection decisions
4. **Works alongside existing optimizations** - Compatible with Transformer Engine, torch.compile
5. **Low overhead** - Falls back to PyTorch for small matrices automatically

### When tritonBLAS Helps Most:

- **Large matrix operations** in transformer models (attention, linear layers)
- **Batch inference** where matmul operations dominate compute time
- **RX 7900 GPUs** where FP8 acceleration isn't available
- **When torch.compile overhead is too high** for your use case

## Installation

### 1. Install tritonBLAS

The tritonBLAS library is already included in `cache/tritonBLAS-main/`. To ensure it's properly set up:

```bash
cd cache/tritonBLAS-main
pip install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

### 2. Verify Installation

```bash
python -c "import sys; sys.path.insert(0, 'cache/tritonBLAS-main/include'); import tritonblas; print('tritonBLAS OK')"
```

## Usage

### Basic Usage (Recommended)

Enable tritonBLAS with default settings:

```bash
export FRAMEPACK_USE_TRITONBLAS=1
python demo_gradio.py
```

### Advanced Configuration

Control tritonBLAS behavior with environment variables:

```bash
# Enable tritonBLAS
export FRAMEPACK_USE_TRITONBLAS=1

# Minimum matrix dimension to use tritonBLAS (default: 512)
# Smaller matrices use PyTorch (less overhead)
export FRAMEPACK_TRITONBLAS_MIN_SIZE=512

# Enable verbose logging to see which operations use tritonBLAS
export FRAMEPACK_TRITONBLAS_VERBOSE=1

# Enable Stream-K algorithm for better load balancing (experimental)
export FRAMEPACK_TRITONBLAS_STREAMK=0

# Disable fallback to PyTorch on errors (for debugging)
export FRAMEPACK_TRITONBLAS_FALLBACK=1

# Run demo
python demo_gradio.py
```

### Recommended Settings for RX 7900 XTX

```bash
# Optimized for RX 7900 XTX with 24GB VRAM
export FRAMEPACK_USE_TRITONBLAS=1           # Enable tritonBLAS
export FRAMEPACK_TRITONBLAS_MIN_SIZE=256    # Lower threshold for RX 7900
export FRAMEPACK_TRITONBLAS_VERBOSE=0       # Disable verbose (cleaner output)
export FRAMEPACK_TRITONBLAS_FALLBACK=1      # Keep fallback enabled (safer)

# Also recommended: disable FP8 (not supported on RX 7900)
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0

# Enable Transformer Engine for optimized kernels (without FP8)
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1

# Use torch.compile with inductor backend
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune

python demo_gradio.py
```

## How It Works

### Monkey-Patching Approach

The integration uses monkey-patching to intercept PyTorch's matmul operations:

1. **torch.matmul** → `tritonblas_matmul`
2. **torch.nn.functional.linear** → `tritonblas_linear`
3. **torch.addmm** → `tritonblas_addmm`

### Size Filtering

tritonBLAS is only used for matrices above a size threshold:

- Default: `min_dim >= 512` or `m*n*k >= 512^3`
- Rationale: Small matrices benefit from PyTorch's low overhead
- Configurable via `FRAMEPACK_TRITONBLAS_MIN_SIZE`

### Fallback Strategy

If tritonBLAS encounters an error (e.g., unsupported dtype, layout):

1. Logs the error (if verbose mode enabled)
2. Falls back to original PyTorch operation
3. Increments error counter in statistics

### Statistics Tracking

After each generation, tritonBLAS prints usage statistics:

```
============================================================
tritonBLAS Usage Statistics
============================================================
  Total matmul/linear calls: 15234
  tritonBLAS calls: 8421
  PyTorch fallback calls: 5813
  Size-filtered calls: 1000
  Error calls: 0
  tritonBLAS usage: 55.3%
============================================================
```

## Compatibility

### Compatible With:

✅ **Transformer Engine** - tritonBLAS operates at lower level (matmul), TE wraps Linear layers
✅ **torch.compile** - May bypass patching (torch.compile generates its own Triton kernels)
✅ **Bitsandbytes quantization** - tritonBLAS works on full-precision matmul
✅ **RX 7900, MI200, MI300 GPUs** - All AMD ROCm GPUs supported
✅ **FP16, BF16, FP32 dtypes** - All standard PyTorch dtypes

### Limitations:

⚠️ **Batched matmul** - Currently falls back to PyTorch for >2D tensors
⚠️ **torch.compile may bypass** - Compiled functions generate inline Triton kernels
⚠️ **Stride/layout requirements** - Some exotic layouts fall back to PyTorch
⚠️ **FP8/FP4 quantization** - Requires special handling (not yet implemented in patch)

## Interaction with Other Optimizations

### tritonBLAS + Transformer Engine

- **Recommended**: Use both for maximum performance
- **Interaction**: TE optimizes Linear layers, tritonBLAS optimizes raw matmul
- **Disable FP8 on RX 7900**: `FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0`

```bash
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0  # RX 7900 doesn't support FP8
```

### tritonBLAS + torch.compile

- **May conflict**: torch.compile generates inline Triton kernels
- **torch.compile takes precedence** for compiled functions
- **tritonBLAS helps non-compiled paths** (text encoders, VAE)

```bash
# Use both - torch.compile for transformer, tritonBLAS for encoders
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
```

### tritonBLAS + Bitsandbytes

- **Compatible**: tritonBLAS operates on full-precision matmul after dequantization
- **Bitsandbytes reduces memory**, tritonBLAS speeds up compute

```bash
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_USE_BITSANDBYTES=1
```

## Performance Tuning

### Finding Optimal MIN_SIZE

The `MIN_SIZE` threshold controls when tritonBLAS is used:

```bash
# Test different thresholds
for size in 128 256 512 1024; do
    echo "Testing MIN_SIZE=$size"
    export FRAMEPACK_TRITONBLAS_MIN_SIZE=$size
    # Run your benchmark
    time python demo_gradio.py --benchmark
done
```

**Guidelines:**
- **RX 7900 XTX**: Try 256-512 (96 CUs, high compute throughput)
- **MI200**: Try 512-1024 (110-120 CUs)
- **MI300X**: Try 512-1024 (304 CUs, but higher per-CU performance)

### Stream-K Algorithm

Stream-K provides better load balancing for irregular workloads:

```bash
export FRAMEPACK_TRITONBLAS_STREAMK=1
```

**When to enable:**
- Batch sizes that don't divide evenly into GPU compute units
- Mixed small/large matrix operations
- Imbalanced workloads

**When to disable:**
- Well-balanced workloads (default)
- Very large matrices (data-parallel mode is faster)

## Troubleshooting

### tritonBLAS Not Loading

```
✗ Cannot enable tritonBLAS: import failed
```

**Solution:**
```bash
# Add tritonBLAS to Python path
export PYTHONPATH=/path/to/framepack_amd/cache/tritonBLAS-main/include:$PYTHONPATH

# Or install system-wide
cd cache/tritonBLAS-main
pip install -e .
```

### High Fallback Rate

```
  tritonBLAS usage: 15.3%
  PyTorch fallback calls: 12850
```

**Possible causes:**
1. `MIN_SIZE` threshold too high - try lowering it
2. Most operations are small matrices - this is normal
3. Batched matmul (>2D tensors) - not yet supported

**Solution:**
```bash
# Lower threshold
export FRAMEPACK_TRITONBLAS_MIN_SIZE=128
export FRAMEPACK_TRITONBLAS_VERBOSE=1  # See what's being filtered
```

### Errors During Matmul

```
  Error calls: 523
```

**Enable verbose logging:**
```bash
export FRAMEPACK_TRITONBLAS_VERBOSE=1
```

**Common errors:**
- Unsupported dtype (int8, etc.) - will auto-fallback
- Incompatible strides - will auto-fallback
- CUDA OOM - reduce batch size or disable tritonBLAS for that operation

## Performance Benchmarks

### Expected Performance Gains

Based on tritonBLAS benchmarks on AMD GPUs:

| Operation | Size | RX 7900 XTX | MI300X |
|-----------|------|-------------|--------|
| FP16 GEMM | 4096³ | ~1.3x faster | ~1.5x faster |
| BF16 GEMM | 8192³ | ~1.4x faster | ~1.6x faster |
| FP32 GEMM | 2048³ | ~1.2x faster | ~1.3x faster |

**Compared to:** Standard PyTorch matmul (rocBLAS backend)

### Measuring Impact

```bash
# Baseline (no tritonBLAS)
export FRAMEPACK_USE_TRITONBLAS=0
time python demo_gradio.py --run-once > baseline.log

# With tritonBLAS
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_TRITONBLAS_VERBOSE=1
time python demo_gradio.py --run-once > tritonblas.log

# Compare timings
diff baseline.log tritonblas.log
```

## Advanced: Custom Integration

If you want to use tritonBLAS in your own code:

```python
from diffusers_helper.tritonblas_patch import patch_pytorch_with_tritonblas

# Enable globally
patch_pytorch_with_tritonblas(
    enable=True,
    verbose=True,
    fallback_to_torch=True,
    min_size=512,
    use_streamk=False,
)

# Now all torch.matmul calls use tritonBLAS (where applicable)
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
c = torch.matmul(a, b)  # Uses tritonBLAS if size >= min_size

# Direct API (no monkey-patching)
import sys
sys.path.insert(0, 'cache/tritonBLAS-main/include')
import tritonblas

a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
c = torch.empty((4096, 4096), device='cuda', dtype=torch.float16)

tritonblas.matmul(a, b, c, enable_streamk=False)
```

## References

- **tritonBLAS GitHub**: [ROCm/tritonBLAS](https://github.com/ROCm/tritonBLAS)
- **Triton Language**: [OpenAI Triton](https://github.com/openai/triton)
- **AMD ROCm Documentation**: [ROCm Docs](https://rocm.docs.amd.com/)
- **Stream-K Paper**: [arxiv.org/abs/2301.03598](https://arxiv.org/abs/2301.03598)

## Support

If you encounter issues:

1. **Check logs**: Enable `FRAMEPACK_TRITONBLAS_VERBOSE=1`
2. **Review statistics**: Look at tritonBLAS usage percentages
3. **Try fallback mode**: Ensure `FRAMEPACK_TRITONBLAS_FALLBACK=1`
4. **Disable if problematic**: `FRAMEPACK_USE_TRITONBLAS=0`
5. **Report issues**: Include verbose logs and GPU model

## Summary

**Quick Start (RX 7900 XTX):**
```bash
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_TRITONBLAS_MIN_SIZE=256
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0
python demo_gradio.py
```

**Expected Benefits:**
- 20-40% faster matrix operations (compute-bound workloads)
- Better GPU utilization on AMD hardware
- Reduced kernel compilation overhead vs autotuning

**Monitor Performance:**
- Check tritonBLAS statistics at end of generation
- Look for >50% tritonBLAS usage for good coverage
- Adjust MIN_SIZE if usage too low or too high
