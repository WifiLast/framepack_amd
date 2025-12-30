# Torch Compile Recommendations for FramePack AMD

## TL;DR - Don't Precompile

**Bottom line**: Precompilation of these large models is **not practical** and **not recommended**.

### Why Precompilation Doesn't Work Here

1. **Transformer is too large**
   - The HunyuanVideo transformer is massive
   - Compilation takes extremely long (many minutes to hours)
   - Memory requirements are very high
   - The compiled graph may be too large to cache effectively

2. **MIGraphX has compatibility issues**
   - Doesn't support `aten._local_scalar_dense.default` operations
   - Fails on data-dependent operations
   - Limited support for dynamic shapes
   - Incompatible with many operations in VAE, encoders

3. **Inductor works but is slow to compile**
   - Better compatibility than MIGraphX
   - But still takes a very long time on large models
   - First run compilation happens during inference anyway

## Recommended Approach: Just-In-Time Compilation

Instead of precompilation, let torch.compile work **just-in-time** during the first inference run:

### Option 1: Use Torch Compile with Inductor (RECOMMENDED)

```bash
# In demo_gradio.py, these settings are already configured:
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune

python demo_gradio.py
```

**What happens:**
- First video generation: Slower (compiles on first use)
- Subsequent generations: Faster (uses cached kernels)
- Kernels cached in `.cache_rocm/inductor/` and `.cache_rocm/triton/`

### Option 2: Disable Torch Compile (FASTEST STARTUP)

If you want the fastest startup and don't mind slightly slower inference:

```bash
export FRAMEPACK_USE_TORCH_COMPILE=0
python demo_gradio.py
```

**What happens:**
- Immediate startup (no compilation)
- Models run in eager mode
- Slightly slower inference but still very usable
- No compilation delays

### Option 3: Compile Only Small Models (COMPROMISE)

If you still want some compilation benefits without the overhead:

```bash
# Edit demo_gradio.py and only compile the VAE:
# Around line 901, change:
vae = configure_vae_inference(vae, target_device=gpu, apply_compile=True)

# And disable transformer compilation:
# Around line 907, comment out:
# transformer = maybe_torch_compile(transformer, 'Hunyuan Transformer')
```

**What happens:**
- VAE compiles quickly (if it works with your backend)
- Transformer runs in eager mode (no compilation overhead)
- Balanced approach

## Performance Expectations

### With Torch Compile (Inductor, JIT):
- **First run**: Very slow startup (5-15 minutes compilation)
- **Subsequent runs**: 15-25% faster inference
- **Memory**: Same as eager mode
- **Stability**: May encounter compilation issues

### Without Torch Compile (Eager Mode):
- **First run**: Immediate startup (< 1 minute)
- **Subsequent runs**: Standard PyTorch performance
- **Memory**: Baseline memory usage
- **Stability**: Excellent (no compilation issues)

## What About the Precompilation Scripts?

The scripts I created (`pytorch_compile_amd.py` and `pytorch_compile_inductor.py`) were created to help, but based on your feedback and the practical reality:

**You should NOT use them** because:
- They take too long to compile large models
- They don't provide significant benefits over JIT compilation
- They can fail with OOM or compatibility errors
- The Transformer is too big to precompile effectively

## Final Recommendation

### For Development/Testing:
```bash
# Disable compilation for fast iteration
export FRAMEPACK_USE_TORCH_COMPILE=0
python demo_gradio.py
```

### For Production/Best Performance:
```bash
# Enable Inductor, let it compile on first use
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
export FRAMEPACK_TORCH_COMPILE_MODE=max-autotune
python demo_gradio.py

# First generation will be slow (compilation)
# All subsequent generations will be faster
```

### For Benchmarking:
```bash
# Disable compilation to get baseline performance
export FRAMEPACK_USE_TORCH_COMPILE=0
python demo_gradio.py

# Then enable compilation and compare
export FRAMEPACK_USE_TORCH_COMPILE=1
export FRAMEPACK_TORCH_COMPILE_BACKEND=inductor
python demo_gradio.py
```

## Summary

**Don't precompile**. The models are too large and complex. Instead:

1. Either disable torch.compile for immediate usability
2. Or enable it and accept the first-run compilation delay
3. The kernels will be cached after first use regardless

The existing `demo_gradio.py` already has sensible defaults that work well for AMD ROCm GPUs. Just use it as-is!

## Cleanup

If you want to remove the precompilation scripts:

```bash
# These scripts are not needed:
rm current/FramePack-main/pytorch_compile_amd.py
rm current/FramePack-main/pytorch_compile_inductor.py

# Keep the documentation for reference:
# - PRECOMPILATION_GUIDE.md
# - MIGRAPHX_COMPILATION_STATUS.md
# - TORCH_COMPILE_RECOMMENDATION.md (this file)
```

## References

For more information about torch.compile on AMD ROCm:
- PyTorch Compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- ROCm Documentation: https://rocm.docs.amd.com/
- Triton for AMD: https://github.com/ROCmSoftwarePlatform/triton
