# Troubleshooting: AMD ROCm → NVIDIA CUDA Workflow

## Problem: demo_gradio.py (AMD) is Slow After Running process_saved_latents.py (CUDA)

### Symptoms
- `process_saved_latents.py` runs fine on NVIDIA/CUDA
- After it completes, starting a new generation with `demo_gradio.py` on AMD/ROCm takes significantly longer
- The delay happens during model initialization or first inference

### Root Causes

This happens because both environments may share certain cached resources:

1. **PyTorch Compiled Kernels Cache**
   - `torch.compile` caches compiled kernels in `~/.triton/cache` or similar
   - CUDA-compiled kernels can interfere with ROCm kernels

2. **HuggingFace Model Cache**
   - Model weights are cached in `~/.cache/huggingface/`
   - Loading from cache can trigger re-initialization

3. **Shared Memory/GPU State**
   - If both GPUs are in the same system, there may be shared driver state
   - PCIe resource allocation conflicts

### Solutions

#### Solution 1: Clear Triton Cache Between Runs (Recommended)

Add cache clearing to your workflow:

```bash
# After running process_saved_latents.py, before running demo_gradio.py
rm -rf ~/.triton/cache/*
rm -rf /tmp/__pycache__/*

# Or add to your workflow script
python auto_process_latents.py --mode watch --clear-cache
```

#### Solution 2: Use Separate Conda Environments

Keep AMD and NVIDIA environments completely isolated:

```bash
# AMD ROCm environment (current)
conda activate base  # or your ROCm environment
python demo_gradio.py

# NVIDIA CUDA environment (separate)
conda activate hunyuan3d_21
python process_saved_latents.py --latents outputs/xxx_latents.pt
conda deactivate

# Clear cache before returning to AMD
rm -rf ~/.triton/cache/*
```

#### Solution 3: Disable torch.compile for process_saved_latents.py

The VAE in `process_saved_latents.py` doesn't use `torch.compile` by default, but if you've enabled it, disable it:

```python
# In process_saved_latents.py, the load_vae function doesn't use torch.compile
# This should already be the case, but verify no compilation is happening
```

#### Solution 4: Reset GPU State After CUDA Processing

Add explicit GPU reset after `process_saved_latents.py` completes:

```bash
# After CUDA processing
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()"

# Or use nvidia-smi to reset GPU
nvidia-smi --gpu-reset

# For AMD GPU, reset driver state
rocm-smi --resetclocks
```

#### Solution 5: Increase MIOpen Find Timeout

The slowdown might be MIOpen re-running algorithm search. Increase the cache validity:

```bash
# In demo_gradio.py, already set:
export MIOPEN_FIND_MODE=FAST
export MIOPEN_FIND_TIME_LIMIT=30

# Try increasing the timeout:
export MIOPEN_FIND_TIME_LIMIT=60  # 60 seconds instead of 30
```

#### Solution 6: Use Persistent MIOpen Database

Enable MIOpen's persistent database to avoid re-searching:

```bash
# Set MIOpen database location
export MIOPEN_USER_DB_PATH=~/.config/miopen/

# Ensure database persists between runs
export MIOPEN_DEBUG_DISABLE_FIND_DB=0  # Already set in demo_gradio.py
```

### Recommended Workflow

**Option A: Automatic Cache Clearing**

```bash
# Terminal 1: Start demo_gradio.py (AMD)
python demo_gradio.py

# Terminal 2: Auto-process with cache clearing
python auto_process_latents.py --mode watch --clear-cache
```

**Option B: Manual Sequential Processing**

```bash
# Step 1: Generate latents (AMD)
python demo_gradio.py
# Wait for completion...

# Step 2: Clear caches
rm -rf ~/.triton/cache/* /tmp/__pycache__/*

# Step 3: Process on CUDA
conda activate hunyuan3d_21
python process_saved_latents.py --latents outputs/xxx_latents.pt
conda deactivate

# Step 4: Clear caches again
rm -rf ~/.triton/cache/*

# Step 5: Return to demo_gradio.py (AMD)
# Should start quickly now
```

**Option C: Separate Physical Machines**

The cleanest solution: Run AMD and NVIDIA processing on separate machines:
- Machine 1 (AMD): Generate latents, save to shared storage
- Machine 2 (NVIDIA): Process latents, save videos to shared storage

### Diagnostic Commands

Check what's consuming time during startup:

```bash
# Profile demo_gradio.py startup
python -m cProfile -o profile.stats demo_gradio.py

# Check GPU memory state
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Check cache sizes
du -sh ~/.triton/cache
du -sh ~/.cache/huggingface

# Monitor startup in real-time
python demo_gradio.py 2>&1 | ts  # requires 'moreutils' package
```

### Prevention: Run on Different GPUs

If you have multiple GPUs, use different ones:

```bash
# AMD GPU 0 for demo_gradio.py
export HIP_VISIBLE_DEVICES=0
python demo_gradio.py

# NVIDIA GPU 1 for process_saved_latents.py
python process_saved_latents.py --device cuda:1 --latents outputs/xxx_latents.pt
```

This avoids GPU state interference entirely.

### Advanced: Pre-warm AMD Environment

Pre-load models before your workflow starts:

```bash
# Create a warmup script
cat > warmup_amd.py << 'EOF'
import torch
from diffusers import AutoencoderKLHunyuanVideo

print("Pre-warming AMD environment...")
vae = AutoencoderKLHunyuanVideo.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='vae',
    torch_dtype=torch.float16,
)
vae.to('cuda')
print("Warmup complete")
del vae
torch.cuda.empty_cache()
EOF

# Run before starting workflow
python warmup_amd.py
python demo_gradio.py
```

## Other Common Issues

### Issue: "CUDA out of memory"

**Solution**: Use `--enable-slicing` flag:
```bash
python process_saved_latents.py --latents outputs/xxx_latents.pt --enable-slicing
```

### Issue: "MIOpen Find hanging" in demo_gradio.py

**Solution**: Already configured in demo_gradio.py, but verify:
```bash
export MIOPEN_FIND_MODE=FAST
export MIOPEN_FIND_TIME_LIMIT=30
python demo_gradio.py
```

### Issue: Video duration mismatch

**Solution**: This was fixed in the latest process_saved_latents.py. Make sure you're using the updated version.

### Issue: "Cannot find conda environment"

**Solution**:
```bash
# Check available environments
conda env list

# Update script with correct environment name
python auto_process_latents.py --mode watch --conda-env YOUR_ENV_NAME
```
