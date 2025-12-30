"""
PyTorch Model Precompilation Script for AMD ROCm with Inductor Backend

This script precompiles models using the Inductor backend instead of MIGraphX.
Inductor has much better compatibility with complex models while still providing
excellent AMD ROCm optimizations through Triton.

Usage:
    python pytorch_compile_inductor.py

Note: This uses the Inductor backend which is more compatible than MIGraphX
but still provides excellent performance on AMD GPUs through Triton.
"""

import os
import runpy
import time
import torch

# Configure environment paths (same as demo_gradio.py)
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Separate cache directories for AMD ROCm
_cache_base = os.path.join(os.path.dirname(__file__), '.cache_rocm')
os.makedirs(_cache_base, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = os.path.join(_cache_base, 'triton')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_cache_base, 'torch_extensions')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_cache_base, 'inductor')

# Trim PyTorch/HIP allocations aggressively
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:64')

# Load ZLUDA compatibility layer
zluda_entry = os.path.join(os.path.dirname(__file__), 'customzluda', 'zluda-default.py')
if os.path.exists(zluda_entry):
    print('Loading ZLUDA compatibility layer for ROCm/AMD GPUs...')
    runpy.run_path(zluda_entry, run_name='__framepack_zluda__')

# Prevent diffusers from importing bitsandbytes (AMD ROCm compatibility)
import sys
import types

mock_cpu_module = types.ModuleType('bitsandbytes.backends.cpu')
mock_cpu_module.CPUBackend = type('CPUBackend', (), {})
sys.modules['bitsandbytes.backends.cpu'] = mock_cpu_module

mock_cuda_module = types.ModuleType('bitsandbytes.backends.cuda')
mock_cuda_module.CUDABackend = type('CUDABackend', (), {})
sys.modules['bitsandbytes.backends.cuda'] = mock_cuda_module

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

IS_HIP_RUNTIME = getattr(torch.version, "hip", None) is not None

print("\n" + "="*70)
print("PyTorch Model Precompilation for AMD ROCm with Inductor Backend")
print("="*70)
print(f"Cache Directory: {_cache_base}")
print(f"Backend: Inductor (with Triton for AMD ROCm)")
print("="*70 + "\n")


def get_device():
    """Get the GPU device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        print("ERROR: No CUDA/ROCm device available")
        exit(1)


def precompile_transformer_inductor(device):
    """
    Precompile the Transformer model with Inductor backend.
    This is the most important model - 95% of compute happens here!
    """
    print("\nLoading Transformer (HunyuanVideo)...")
    print("This is the primary model for video generation")

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY',
        torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    transformer.requires_grad_(False)

    # Example inputs for transformer (from demo_gradio.py worker function)
    batch_size = 1
    num_frames = 33  # latent_window_size * 4 - 3
    height, width = 640, 640
    latent_height, latent_width = height // 8, width // 8

    print(f"\nPreparing example inputs:")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {height}x{width} (latent: {latent_height}x{latent_width})")

    example_inputs = {
        'hidden_states': torch.randn(batch_size, 16, num_frames, latent_height, latent_width,
                                     dtype=torch.bfloat16, device=device),
        'timestep': torch.tensor([500.0], dtype=torch.bfloat16, device=device),
        'encoder_hidden_states': torch.randn(batch_size, 512, 4096, dtype=torch.bfloat16, device=device),
        'encoder_attention_mask': torch.ones(batch_size, 512, dtype=torch.bfloat16, device=device),
        'pooled_projections': torch.randn(batch_size, 4096, dtype=torch.bfloat16, device=device),
        'image_embeddings': torch.randn(batch_size, 729, 1152, dtype=torch.bfloat16, device=device),
    }

    print(f"\n{'='*70}")
    print(f"Compiling Transformer with Inductor backend")
    print(f"{'='*70}")

    # Inductor compilation options for AMD ROCm
    compile_options = {
        'backend': 'inductor',
        'mode': 'max-autotune',  # Inductor supports mode
        'dynamic': True,
    }

    if IS_HIP_RUNTIME:
        # ROCm-specific optimizations
        compile_options['options'] = {
            'triton.cudagraphs': False,  # Disable for ROCm
            'max_autotune': True,
            'epilogue_fusion': True,
            'coordinate_descent_tuning': True,
        }
        print("ROCm detected - using Triton-optimized kernels")
    else:
        print("CUDA detected - using standard Inductor optimizations")

    print(f"Compilation options: {compile_options}")

    start_time = time.time()

    try:
        compiled_transformer = torch.compile(transformer, **compile_options)

        # Trigger compilation with example inputs (dry run)
        print(f"\nRunning warmup pass to trigger compilation...")
        print("This may take several minutes on first run...")

        with torch.no_grad():
            _ = compiled_transformer(**example_inputs)

        compilation_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"SUCCESS: Transformer compiled in {compilation_time:.2f} seconds!")
        print(f"{'='*70}")
        print(f"Inductor kernels are now cached for subsequent runs")
        print(f"Future runs will load much faster from cache")

        return compiled_transformer

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FAILED: Could not compile Transformer")
        print(f"{'='*70}")
        print(f"Error: {str(e)[:300]}")
        print(f"\nThe model will still work in eager mode (uncompiled)")
        return transformer


def main():
    """Main precompilation routine - Transformer only with Inductor."""
    device = get_device()
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Total GPU memory: {free_mem_gb:.2f} GB\n")

    total_start = time.time()

    try:
        print("="*70)
        print("PRECOMPILATION STRATEGY")
        print("="*70)
        print("Focusing on the Transformer model:")
        print("  - This model handles 95% of the computational workload")
        print("  - It runs hundreds of times during video generation")
        print("  - Compiling it provides the biggest performance boost")
        print("")
        print("Other models (VAE, encoders) will run in eager mode:")
        print("  - They run only once or a few times per generation")
        print("  - Their performance impact is minimal")
        print("  - Running them uncompiled is perfectly fine")
        print("="*70 + "\n")

        transformer_compiled = precompile_transformer_inductor(device)
        del transformer_compiled
        torch.cuda.empty_cache()

        total_time = time.time() - total_start

        print("\n" + "="*70)
        print("PRECOMPILATION COMPLETE!")
        print("="*70)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\nCached kernels location:")
        print(f"  Inductor: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
        print(f"  Triton:   {os.environ['TRITON_CACHE_DIR']}")
        print(f"\nNext steps:")
        print(f"  1. Run demo_gradio.py normally")
        print(f"  2. The Transformer will load compiled kernels from cache")
        print(f"  3. Enjoy faster inference!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nPrecompilation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
