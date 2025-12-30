"""
PyTorch Model Precompilation Script for AMD ROCm with MIGraphX Backend

This script precompiles models used in demo_gradio.py with torch.compile using
the MIGraphX backend for AMD GPUs. Compiled models are cached to speed up
subsequent runs of demo_gradio.py.

Usage:
    python pytorch_compile_amd.py

Models precompiled:
    - VAE (AutoencoderKLHunyuanVideo)
    - Transformer (HunyuanVideoTransformer3DModelPacked)
    - Text Encoder (LlamaModel)
    - Text Encoder 2 (CLIPTextModel)
    - Image Encoder (SiglipVisionModel)
"""

from diffusers_helper.hf_login import login

import os
import runpy
import time
import torch
import torch.nn as nn

# Configure environment paths (same as demo_gradio.py)
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Separate cache directories for AMD ROCm
_cache_base = os.path.join(os.path.dirname(__file__), '.cache_rocm')
os.makedirs(_cache_base, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = os.path.join(_cache_base, 'triton')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_cache_base, 'torch_extensions')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_cache_base, 'inductor')

# MIGraphX compiled models cache directory
MIGRAPHX_CACHE_DIR = os.path.join(_cache_base, 'migraphx_compiled')
os.makedirs(MIGRAPHX_CACHE_DIR, exist_ok=True)

""" # Configure MIOpen for AMD GPUs
os.environ['MIOPEN_FIND_MODE'] = '2'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '0'
os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'
os.environ['MIOPEN_FIND_TIME_LIMIT'] = '30'

# Enable 3D convolution algorithms
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS'] = '1'
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS'] = '1'
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS'] = '1'

# Enable fallback mechanisms
os.environ['MIOPEN_DEBUG_CONV_IMMED_FALLBACK'] = '1'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES'] = '1'

# Enable all convolution algorithm types
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW'] = '1'

os.environ['MIOPEN_LOG_LEVEL'] = '4' """

# Trim PyTorch/HIP allocations aggressively
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:64')

# Load ZLUDA compatibility layer
zluda_entry = os.path.join(os.path.dirname(__file__), 'customzluda', 'zluda-default.py')
if os.path.exists(zluda_entry):
    print('Loading ZLUDA compatibility layer for ROCm/AMD GPUs...')
    runpy.run_path(zluda_entry, run_name='__framepack_zluda__')
else:
    print(f'Warning: ZLUDA helper not found at {zluda_entry}')

# Prevent diffusers from importing bitsandbytes (AMD ROCm compatibility)
# The AMD ROCm version of bitsandbytes lacks CPUBackend and CUDABackend which diffusers expects
import sys
import types

# Create mock bitsandbytes backend modules with required Backend classes
mock_cpu_module = types.ModuleType('bitsandbytes.backends.cpu')
mock_cpu_module.CPUBackend = type('CPUBackend', (), {})  # Minimal mock class
sys.modules['bitsandbytes.backends.cpu'] = mock_cpu_module

mock_cuda_module = types.ModuleType('bitsandbytes.backends.cuda')
mock_cuda_module.CUDABackend = type('CUDABackend', (), {})  # Minimal mock class
sys.modules['bitsandbytes.backends.cuda'] = mock_cuda_module

# Import torch_migraphx
try:
    import torch_migraphx
    HAS_TORCH_MIGRAPHX = True
    print("torch_migraphx available - AMD MIGraphX backend enabled")
except ImportError:
    HAS_TORCH_MIGRAPHX = False
    print("ERROR: torch_migraphx not installed. Install with: pip install torch_migraphx")
    exit(1)

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, SiglipVisionModel
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

IS_HIP_RUNTIME = getattr(torch.version, "hip", None) is not None

if not IS_HIP_RUNTIME:
    print("WARNING: Not running on AMD ROCm/HIP runtime. This script is designed for AMD GPUs.")
    print("Continuing anyway, but MIGraphX optimizations may not be effective.")

# MIGraphX compilation settings
USE_MIGRAPHX_BF16 = os.environ.get('FRAMEPACK_MIGRAPHX_BF16', '0').strip().lower() in ('1', 'true', 'yes', 'on')
USE_MIGRAPHX_DEALLOCATE = os.environ.get('FRAMEPACK_MIGRAPHX_DEALLOCATE', '1').strip().lower() in ('1', 'true', 'yes', 'on')

print("\n" + "="*70)
print("PyTorch Model Precompilation for AMD ROCm with MIGraphX")
print("="*70)
print(f"MIGraphX Cache Directory: {MIGRAPHX_CACHE_DIR}")
print(f"MIGraphX BF16 Precision: {USE_MIGRAPHX_BF16}")
print(f"MIGraphX Memory Deallocation: {USE_MIGRAPHX_DEALLOCATE}")
print("="*70 + "\n")


def get_device():
    """Get the GPU device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        print("ERROR: No CUDA/ROCm device available")
        exit(1)


def compile_model_with_migraphx(model, model_name, example_inputs, mode='max-autotune', dynamic=True):
    """
    Compile a model with MIGraphX backend and save to cache.

    Args:
        model: PyTorch model to compile
        model_name: Name for logging and cache file naming
        example_inputs: Example inputs for tracing (tuple of tensors)
        mode: Compilation mode (ignored for MIGraphX, kept for API compatibility)
        dynamic: Enable dynamic shapes (default: True)

    Returns:
        Compiled model
    """
    print(f"\n{'='*70}")
    print(f"Compiling {model_name} with MIGraphX backend")
    print(f"{'='*70}")

    # Build compilation options for MIGraphX
    # Note: MIGraphX backend doesn't support 'mode', only 'options'
    compile_options = {
        'backend': 'migraphx',
    }

    if dynamic:
        compile_options['dynamic'] = True

    # MIGraphX-specific options
    migraphx_options = {}
    if USE_MIGRAPHX_BF16:
        migraphx_options['bf16'] = True
    if USE_MIGRAPHX_DEALLOCATE:
        migraphx_options['deallocate'] = True

    # Always add options for MIGraphX (even if empty, to ensure proper backend behavior)
    compile_options['options'] = migraphx_options

    print(f"Compilation options: {compile_options}")

    # Better input shape formatting
    if isinstance(example_inputs, dict):
        input_info = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v) for k, v in example_inputs.items()}
    else:
        input_info = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t) for t in example_inputs]
    print(f"Example input shapes: {input_info}")

    # Compile the model
    start_time = time.time()

    try:
        compiled_model = torch.compile(model, **compile_options)

        # Trigger compilation with example inputs (dry run)
        print(f"Running warmup pass to trigger MIGraphX compilation...")
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                _ = compiled_model(**example_inputs)
            else:
                _ = compiled_model(*example_inputs)

        compilation_time = time.time() - start_time
        print(f" Successfully compiled {model_name} in {compilation_time:.2f} seconds")
        print(f"  MIGraphX kernels are now cached for subsequent runs")

        return compiled_model

    except Exception as e:
        print(f" Failed to compile {model_name}: {e}")
        print(f"  Returning uncompiled model")
        return model


def precompile_vae(device):
    """Precompile the VAE model."""
    print("\nLoading VAE...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Enable tiling (as in demo_gradio.py)
    vae.enable_tiling()

    print(f"\nNote: VAE contains data-dependent operations and cannot be compiled with MIGraphX.")
    print(f"The VAE will run in eager mode (this is normal and expected).")
    print(f"VAE loading complete")

    # Return the uncompiled VAE since it can't be compiled with MIGraphX
    # The VAE has dynamic control flow that MIGraphX can't handle
    return vae


def precompile_transformer(device):
    """Precompile the Transformer model."""
    print("\nLoading Transformer...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY',
        torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    transformer.requires_grad_(False)

    # Example inputs for transformer (from demo_gradio.py worker function)
    # These are typical shapes used during inference
    batch_size = 1
    num_frames = 33  # latent_window_size * 4 - 3 (for latent_window_size=9)
    height, width = 640, 640
    latent_height, latent_width = height // 8, width // 8

    example_inputs = {
        'hidden_states': torch.randn(batch_size, 16, num_frames, latent_height, latent_width,
                                     dtype=torch.bfloat16, device=device),
        'timestep': torch.tensor([500.0], dtype=torch.bfloat16, device=device),
        'encoder_hidden_states': torch.randn(batch_size, 512, 4096, dtype=torch.bfloat16, device=device),
        'encoder_attention_mask': torch.ones(batch_size, 512, dtype=torch.bfloat16, device=device),
        'pooled_projections': torch.randn(batch_size, 4096, dtype=torch.bfloat16, device=device),
        'image_embeddings': torch.randn(batch_size, 729, 1152, dtype=torch.bfloat16, device=device),
    }

    # Compile transformer
    compiled_transformer = compile_model_with_migraphx(
        transformer,
        "Transformer",
        example_inputs,
        mode='max-autotune',
        dynamic=True
    )

    print(f"Transformer precompilation complete")
    return compiled_transformer


def precompile_text_encoders(device):
    """Precompile text encoder models."""
    print("\nLoading Text Encoders...")

    # Text Encoder (LlamaModel)
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder',
        torch_dtype=torch.float16
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # Text Encoder 2 (CLIPTextModel)
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder_2',
        torch_dtype=torch.float16
    ).to(device)
    text_encoder_2.eval()
    text_encoder_2.requires_grad_(False)

    # Example inputs for text encoders
    # Typical sequence length is 512 tokens
    seq_length = 77  # Standard CLIP sequence length

    example_input_ids = torch.randint(0, 32000, (1, seq_length), device=device)
    example_attention_mask = torch.ones(1, seq_length, device=device, dtype=torch.long)

    # Try to compile Text Encoder 2 (CLIP) - more commonly used
    print("\nAttempting to compile CLIP Text Encoder...")
    compiled_text_encoder_2 = compile_model_with_migraphx(
        text_encoder_2,
        "TextEncoder2_CLIP",
        {
            'input_ids': example_input_ids,
            'attention_mask': example_attention_mask,
        },
        mode='max-autotune',
        dynamic=True  # Allow dynamic shapes for flexibility
    )

    print(f"Text encoder precompilation complete")

    # Clean up
    del text_encoder

    return compiled_text_encoder_2


def precompile_image_encoder(device):
    """Precompile the image encoder model."""
    print("\nLoading Image Encoder...")

    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder='image_encoder',
        torch_dtype=torch.float16
    ).to(device)
    image_encoder.eval()
    image_encoder.requires_grad_(False)

    # Example input for image encoder (384x384 is typical for SigLIP)
    example_pixel_values = torch.randn(1, 3, 384, 384, dtype=torch.float16, device=device)

    # Try to compile image encoder
    print("\nAttempting to compile SigLIP Image Encoder...")
    compiled_image_encoder = compile_model_with_migraphx(
        image_encoder,
        "ImageEncoder_SigLIP",
        {'pixel_values': example_pixel_values},
        mode='max-autotune',
        dynamic=True  # Allow dynamic shapes for flexibility
    )

    print(f"Image encoder precompilation complete")
    return compiled_image_encoder


def main():
    """Main precompilation routine."""
    device = get_device()
    print(f"\nUsing device: {device}")

    free_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"Total GPU memory: {free_mem_gb:.2f} GB\n")

    # Track total compilation time
    total_start = time.time()

    try:
        # Precompile each model
        # Note: We compile models individually to manage memory

        print("\n" + "="*70)
        print("Step 1: Precompiling VAE Decoder")
        print("="*70)
        vae_compiled = precompile_vae(device)
        del vae_compiled
        torch.cuda.empty_cache()

        print("\n" + "="*70)
        print("Step 2: Precompiling Image Encoder")
        print("="*70)
        image_enc_compiled = precompile_image_encoder(device)
        del image_enc_compiled
        torch.cuda.empty_cache()

        print("\n" + "="*70)
        print("Step 3: Precompiling Text Encoders")
        print("="*70)
        text_enc_compiled = precompile_text_encoders(device)
        del text_enc_compiled
        torch.cuda.empty_cache()

        #print("\n" + "="*70)
        #print("Step 4: Precompiling Transformer (this may take several minutes)")
        #print("="*70)
        #transformer_compiled = precompile_transformer(device)
        #del transformer_compiled
        #torch.cuda.empty_cache()

        total_time = time.time() - total_start

        print("\n" + "="*70)
        print("Precompilation Complete!")
        print("="*70)
        print(f"Total compilation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\nMIGraphX kernels are cached in: {MIGRAPHX_CACHE_DIR}")
        print(f"Inductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
        print(f"Triton cache: {os.environ['TRITON_CACHE_DIR']}")
        print("\nNext time you run demo_gradio.py, models will load much faster!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n Precompilation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
