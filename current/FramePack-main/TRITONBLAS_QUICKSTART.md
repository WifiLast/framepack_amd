# tritonBLAS Quick Start Guide

## TL;DR - Enable tritonBLAS for RX 7900

```bash
export FRAMEPACK_USE_TRITONBLAS=1
python demo_gradio.py
```

That's it! tritonBLAS is now accelerating matrix operations.

---

## What You Just Enabled

✅ **Optimized GEMM kernels** for AMD GPUs
✅ **Analytical model selection** (no autotuning overhead)
✅ **Automatic fallback** to PyTorch when needed
✅ **Works with RX 7900** (no FP8 required)

---

## Recommended Settings

### For RX 7900 XTX (24GB VRAM)

```bash
# Core optimizations
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_TRITONBLAS_MIN_SIZE=256
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0  # RX 7900 doesn't support FP8

# Run
python demo_gradio.py
```

### For RX 7900 XT (20GB VRAM)

```bash
# Same as XTX + memory optimizations
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_TRITONBLAS_MIN_SIZE=256
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0
export FRAMEPACK_USE_BITSANDBYTES=0  # Disable to avoid conflicts

# Run
python demo_gradio.py
```

---

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEPACK_USE_TRITONBLAS` | `0` | Enable tritonBLAS (set to `1`) |
| `FRAMEPACK_TRITONBLAS_MIN_SIZE` | `512` | Min matrix size for tritonBLAS |
| `FRAMEPACK_TRITONBLAS_VERBOSE` | `0` | Show debug logs (set to `1`) |
| `FRAMEPACK_TRITONBLAS_STREAMK` | `0` | Enable Stream-K (experimental) |
| `FRAMEPACK_TRITONBLAS_FALLBACK` | `1` | Auto-fallback on errors |

---

## Checking If It's Working

After running a generation, you'll see statistics:

```
============================================================
tritonBLAS Usage Statistics
============================================================
  Total matmul/linear calls: 15234
  tritonBLAS calls: 8421        ← Should be >50% for good coverage
  PyTorch fallback calls: 5813
  Size-filtered calls: 1000
  Error calls: 0                ← Should be 0 or very low
  tritonBLAS usage: 55.3%       ← Healthy usage percentage
============================================================
```

**Good:** 40-70% tritonBLAS usage
**Low:** <20% usage → Lower `MIN_SIZE`
**High:** >90% usage → May be using tritonBLAS for small matrices (overhead)

---

## Troubleshooting

### "tritonBLAS not found"

```bash
# Add to Python path
export PYTHONPATH=/path/to/framepack_amd/cache/tritonBLAS-main/include:$PYTHONPATH
```

### Low usage percentage (<20%)

```bash
# Lower the threshold
export FRAMEPACK_TRITONBLAS_MIN_SIZE=128
```

### Performance not improving

```bash
# Enable verbose to see what's happening
export FRAMEPACK_TRITONBLAS_VERBOSE=1
python demo_gradio.py
```

---

## Optimization Stack (Recommended)

For **maximum performance** on RX 7900, use this combination:

```bash
# Layer 1: tritonBLAS for matmul acceleration
export FRAMEPACK_USE_TRITONBLAS=1
export FRAMEPACK_TRITONBLAS_MIN_SIZE=256

# Layer 2: Transformer Engine for Linear layers (no FP8)
export FRAMEPACK_USE_TRANSFORMER_ENGINE=1
export FRAMEPACK_USE_TRANSFORMER_ENGINE_FP8=0

# Layer 3: torch.compile for graph optimization
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune

# Layer 4: Memory optimizations
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64

# Run
python demo_gradio.py
```

**Expected speedup:** 30-50% faster inference vs baseline PyTorch

---

## When NOT to Use tritonBLAS

❌ **Don't enable if:**
- You're using NVIDIA GPUs (tritonBLAS is AMD-optimized)
- Your workload is memory-bound (not compute-bound)
- torch.compile already gives you good performance (may conflict)
- You're debugging other issues (simplify the stack first)

✅ **Do enable if:**
- You have RX 7900, MI200, or MI300 GPU
- Matrix operations are bottleneck (transformer inference)
- You want faster kernel selection vs autotuning
- You're not using FP8 (RX 7900 doesn't support it)

---

## More Information

See [TRITONBLAS_INTEGRATION.md](TRITONBLAS_INTEGRATION.md) for:
- Detailed architecture explanation
- Performance tuning guide
- Interaction with other optimizations
- Custom integration examples

---

## Quick Comparison

### Without tritonBLAS (PyTorch default)
- Uses rocBLAS for matmul (AMD's default BLAS library)
- Autotuning overhead on first run
- Good general performance

### With tritonBLAS
- Uses Triton kernels optimized for AMD architecture
- Analytical model → instant kernel selection
- Better performance for transformer workloads

**Bottom line:** tritonBLAS is faster for AI workloads on AMD GPUs.

---

## Contact / Support

- **Issues?** Check verbose logs: `FRAMEPACK_TRITONBLAS_VERBOSE=1`
- **Errors?** Fallback is automatic (enabled by default)
- **Questions?** See full docs: [TRITONBLAS_INTEGRATION.md](TRITONBLAS_INTEGRATION.md)

Happy optimizing! 🚀
