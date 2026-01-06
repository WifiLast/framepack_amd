from diffusers_helper.hf_login import login

import os
import runpy
import ctypes
import hashlib
import time
from collections import OrderedDict
import threading

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Separate cache directories for AMD ROCm to prevent CUDA/ROCm interference
_cache_base = os.path.join(os.path.dirname(__file__), '.cache_rocm')
os.makedirs(_cache_base, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = os.path.join(_cache_base, 'triton')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_cache_base, 'torch_extensions')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_cache_base, 'inductor')

# Configure MIOpen for AMD GPUs to prevent convolution errors
# Find mode and database configuration
# CRITICAL: Use FAST mode to prevent hanging in Find phase (VAE decoder issue)
os.environ['MIOPEN_FIND_MODE'] = '2'  # Skip exhaustive search to prevent hangs
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '0'  # Enable find database
os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'  # Don't enforce Find if it's slow/hanging

# CRITICAL: Set timeout for Find operations (prevents infinite hanging)
os.environ['MIOPEN_FIND_TIME_LIMIT'] = '30'  # 30 second timeout for algorithm search

# Critical: Enable 3D convolution algorithms (for VAE decoder)
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS'] = '1'
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS'] = '1'
os.environ['MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS'] = '1'

# Enable fallback mechanisms (CRITICAL for preventing hangs)
os.environ['MIOPEN_DEBUG_CONV_IMMED_FALLBACK'] = '1'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'

# Force use of immediate mode kernels when Find times out
os.environ['MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES'] = '1'

# Enable all convolution algorithm types including fallback algorithms
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD'] = '1'  # Add naive to search space
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD'] = '1'
os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW'] = '1'
# Note: Naive algorithms are in the search space but MIOpen will prefer optimized ones
# Only if optimized algorithms fail will MIOpen select naive (automatic fallback)

# Logging (set to 4 for warnings, 5 for debug if issues persist)
os.environ['MIOPEN_LOG_LEVEL'] = '4'

# Trim PyTorch/HIP allocations aggressively to keep individual BlockAllocator requests small.
# The BlockAllocator documented in cache/BlockAllocator.txt works on 2MB blocks, so we keep
# the caching allocator from hanging onto large chunks that would otherwise fragment the pool.
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:64')

# Load ZLUDA compatibility layer for AMD GPUs by default
zluda_entry = os.path.join(os.path.dirname(__file__), 'customzluda', 'zluda-default.py')
if os.path.exists(zluda_entry):
    print('Loading ZLUDA compatibility layer for ROCm/AMD GPUs...')
    runpy.run_path(zluda_entry, run_name='__framepack_zluda__')
else:
    print(f'Warning: ZLUDA helper not found at {zluda_entry}')
    print('Continuing without ZLUDA - NVIDIA GPU will be used if available')

import gradio as gr
import torch
import torch.nn as nn
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
from typing import Optional
import itertools


# last set if errors occurs this is the reason 
# Mixed Precision Optimization
torch.set_float32_matmul_precision('medium')  # Use TF32/FP16 where beneficial
# not available on AMD
torch.backends.cudnn.allow_tf32 = False
print("✓ Mixed precision mode: medium (5-10% gain)")


# Initialize MIOpen fallback system for AMD GPUs (before any torch operations)
from diffusers_helper.miopen_fallback import initialize_miopen_fallback, MIOpenFallbackHandler
initialize_miopen_fallback(use_monkey_patch=True, verbose=True)

# Apply AMD TransformerEngine optimizations for better performance on ROCm
try:
    from diffusers_helper.amd_te_monkey_patch import (
        apply_amd_te_optimizations,
        convert_model_to_te,
        get_fp8_context,
        HAS_AMD_TE
    )
    _te_optimization_results = apply_amd_te_optimizations(
        verbose=True,
        enable_fp8=False,
        patch_layers=True,  # Don't patch globally, we'll convert models explicitly
    )
    _fp8_recipe = _te_optimization_results.get('fp8_recipe', None)
except ImportError as e:
    print(f"AMD TransformerEngine optimizations not available: {e}")
    HAS_AMD_TE = False
    _fp8_recipe = None
    convert_model_to_te = lambda model, **kwargs: model  # No-op fallback

    # Create dummy context manager
    from contextlib import nullcontext
    get_fp8_context = lambda **kwargs: nullcontext()

# Prevent diffusers from importing bitsandbytes (AMD ROCm compatibility)
# The AMD ROCm version of bitsandbytes lacks CPUBackend and CUDABackend which diffusers expects
# We'll handle quantization manually after model loading instead
import sys
import types

# Create mock bitsandbytes backend modules with required Backend classes
# This allows diffusers to import without errors while we use the real bitsandbytes separately
mock_cpu_module = types.ModuleType('bitsandbytes.backends.cpu')
mock_cpu_module.CPUBackend = type('CPUBackend', (), {})  # Minimal mock class
sys.modules['bitsandbytes.backends.cpu'] = mock_cpu_module

mock_cuda_module = types.ModuleType('bitsandbytes.backends.cuda')
mock_cuda_module.CUDABackend = type('CUDABackend', (), {})  # Minimal mock class
sys.modules['bitsandbytes.backends.cuda'] = mock_cuda_module

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer

# Try to import torch_migraphx for AMD ROCm graph optimization and quantization
# This is AMD's recommended solution for model optimization on ROCm
HAS_TORCH_MIGRAPHX = False
try:
    import torch_migraphx
    HAS_TORCH_MIGRAPHX = True
    print("torch_migraphx available - AMD MIGraphX backend enabled for torch.compile")
except ImportError:
    print("Note: torch_migraphx not installed. Using standard torch backends.")
    print("Install with: pip install torch_migraphx for AMD ROCm optimizations.")

# Try to import the real bitsandbytes for our use (AMD ROCm version has limitations)
# IMPORTANT: Import AFTER diffusers to avoid CPUBackend compatibility issues during diffusers init
HAS_BITSANDBYTES = False
BitsAndBytesConfig = None

try:
    import importlib.util
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec is not None and bnb_spec.origin and 'bitsandbytes' in bnb_spec.origin:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        HAS_BITSANDBYTES = True
        print("Bitsandbytes available - will attempt quantization with fallback to full precision if it fails.")
        print("Note: AMD ROCm bitsandbytes has limitations. Consider using torch_migraphx instead.")
    else:
        print("Note: bitsandbytes not installed. Models will load in full precision.")
except ImportError as e:
    print(f"Note: bitsandbytes not available: {e}")
    print("Models will load in full precision.")
except Exception as e:
    print(f"Note: bitsandbytes import failed: {e}")
    print("Models will load in full precision.")

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.fbcache import FirstBlockCache, FirstBlockCacheConfig
from diffusers_helper.magcache_video import MagCacheConfig, MagCacheVideo
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
# Try to import log_memory_status (only available in AMD version)
try:
    from diffusers_helper.memory import log_memory_status
except ImportError:
    # Fallback for original CUDA version without log_memory_status
    def log_memory_status(device=None, prefix=""):
        pass  # No-op for compatibility

# Try to import psutil for RAM monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with 'pip install psutil' for RAM monitoring.")

from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

def _env_flag(name: str, default: str = '0') -> bool:
    """Return True if the env var is a truthy value (1/true/on)."""
    value = os.environ.get(name, default)
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

# Bitsandbytes 8-bit optimization configuration (AMD ROCm compatible)
USE_BITSANDBYTES = _env_flag('FRAMEPACK_USE_BITSANDBYTES', '1')  # Enabled by default

# AMD TransformerEngine optimization configuration
USE_AMD_TE = _env_flag('FRAMEPACK_USE_AMD_TE', '0')  # Disabled by default (experimental)


def get_quantization_config():
    """
    Create BitsAndBytesConfig for 8-bit quantization (AMD ROCm compatible).

    Returns:
        BitsAndBytesConfig for model loading or None if disabled
    """
    if not USE_BITSANDBYTES or not HAS_BITSANDBYTES or BitsAndBytesConfig is None:
        return None

    try:
        # AMD ROCm compatible configuration
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_threshold=6.0,  # Optional: threshold for outlier detection
        )
        print("  Created BitsAndBytesConfig for 8-bit quantization (AMD ROCm)")
        return quantization_config
    except Exception as e:
        print(f"  ⚠ Failed to create quantization config: {e}")
        return None


free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# RAM monitoring and pinning configuration
def get_ram_info():
    """Get current RAM usage information."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        used_gb = mem.used / (1024**3)
        available_gb = mem.available / (1024**3)
        percent = mem.percent
        return total_gb, used_gb, available_gb, percent
    else:
        # Fallback: assume 32GB total, can't measure actual usage
        return 32.0, 22.0, 10.0, 68.75

total_ram_gb, used_ram_gb, available_ram_gb, ram_percent = get_ram_info()
print(f'Total RAM: {total_ram_gb:.1f} GB')
print(f'Used RAM: {used_ram_gb:.1f} GB ({ram_percent:.1f}%)')
print(f'Available RAM: {available_ram_gb:.1f} GB')

# Configure RAM pinning - use up to 90% of total RAM
MAX_RAM_USAGE_PERCENT = 90.0
target_ram_gb = (MAX_RAM_USAGE_PERCENT / 100.0) * total_ram_gb
ram_headroom_gb = target_ram_gb - used_ram_gb
print(f'Target RAM usage: {MAX_RAM_USAGE_PERCENT}% ({target_ram_gb:.1f} GB)')
print(f'RAM headroom for pinning: {ram_headroom_gb:.1f} GB')

# Enable pinned memory for faster CPU-GPU transfers
#ENABLE_PINNED_MEMORY = ram_headroom_gb > 2.0  # Only enable if we have >2GB headroom
#if ENABLE_PINNED_MEMORY:
#    print(f'Enabling pinned memory for model tensors (improves CPU-GPU transfer speed)')
#else:
#    print(f'Pinned memory disabled (insufficient RAM headroom)')
ENABLE_PINNED_MEMORY = False

def pin_model_to_memory(model: torch.nn.Module, verbose: bool = True):
    """Pin model parameters and buffers to CPU memory for faster GPU transfers."""
    if not ENABLE_PINNED_MEMORY:
        return

    pinned_count = 0
    pinned_size_mb = 0.0

    for name, param in model.named_parameters():
        if param is not None and not param.is_cuda:
            try:
                # Pin memory only for CPU tensors
                if not param.is_pinned():
                    param.data = param.data.pin_memory()
                    pinned_count += 1
                    pinned_size_mb += param.numel() * param.element_size() / (1024**2)
            except:
                pass  # Some tensors may not support pinning

    for name, buffer in model.named_buffers():
        if buffer is not None and not buffer.is_cuda:
            try:
                if not buffer.is_pinned():
                    pinned_buffer = buffer.pin_memory()
                    # Replace the buffer in the model
                    model._buffers[name] = pinned_buffer
                    pinned_count += 1
                    pinned_size_mb += buffer.numel() * buffer.element_size() / (1024**2)
            except:
                pass

    if verbose and pinned_count > 0:
        print(f'  Pinned {pinned_count} tensors ({pinned_size_mb:.1f} MB) in {model.__class__.__name__}')

# Helper function to load model with optional quantization and fallback
def load_model_with_fallback(model_class, model_name, subfolder=None, dtype=torch.float16, quantization_config=None):
    """
    Try to load model with quantization, fall back to full precision if it fails.

    Args:
        model_class: The model class to instantiate
        model_name: Model name/path for from_pretrained
        subfolder: Optional subfolder in the model repository
        dtype: Torch dtype for the model
        quantization_config: BitsAndBytesConfig or None

    Returns:
        Loaded model instance
    """
    load_kwargs = {
        "torch_dtype": dtype,
    }

    if subfolder:
        load_kwargs["subfolder"] = subfolder

    # Try with quantization first if available
    if quantization_config is not None:
        try:
            print(f"  Attempting to load {model_class.__name__} with 8-bit quantization...")
            quant_kwargs = load_kwargs.copy()
            quant_kwargs["quantization_config"] = quantization_config
            quant_kwargs["device_map"] = "auto"
            model = model_class.from_pretrained(model_name, **quant_kwargs)
            print(f"  ✓ {model_class.__name__} loaded with 8-bit quantization")
            return model
        except Exception as e:
            print(f"  ⚠ Quantization failed for {model_class.__name__}: {e}")
            print(f"  → Falling back to full precision for {model_class.__name__}")

    # Load in full precision
    model = model_class.from_pretrained(model_name, **load_kwargs).cpu()
    return model


# Create quantization config for AMD ROCm bitsandbytes
quantization_config = get_quantization_config()

if quantization_config is not None:
    print("\nAttempting to load models with 8-bit quantization...")
    print("Note: Will fall back to full precision if quantization fails for any model.\n")
else:
    print("\nLoading models in full precision...\n")

# Load models with automatic fallback to full precision if quantization fails
text_encoder = load_model_with_fallback(
    LlamaModel,
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder',
    dtype=torch.float16,
    quantization_config=quantization_config
)

text_encoder_2 = load_model_with_fallback(
    CLIPTextModel,
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder_2',
    dtype=torch.float16,
    quantization_config=quantization_config
)

image_encoder = load_model_with_fallback(
    SiglipVisionModel,
    "lllyasviel/flux_redux_bfl",
    subfolder='image_encoder',
    dtype=torch.float16,
    quantization_config=quantization_config
)

tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')

# VAE and custom transformer don't support quantization_config, load normally
print("  Loading VAE (full precision, quantization not supported)...")
vae = AutoencoderKLHunyuanVideo.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='vae',
    torch_dtype=torch.float16
).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')

print("  Loading Transformer (full precision, custom model)...")
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    torch_dtype=torch.bfloat16
).cpu()

print("\nModel loading complete.\n")

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()

transformer.high_quality_fp32_output_for_inference = False
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Pin models to RAM for faster CPU-GPU transfers (if enabled)
if ENABLE_PINNED_MEMORY:
    print('\nPinning models to RAM for optimized memory transfers...')
    pin_model_to_memory(text_encoder)
    pin_model_to_memory(text_encoder_2)
    pin_model_to_memory(vae)
    pin_model_to_memory(image_encoder)
    pin_model_to_memory(transformer)
    print('Model pinning complete.\n')
else:
    print('\nSkipping model pinning (disabled or insufficient RAM).\n')

IS_HIP_RUNTIME = getattr(torch.version, "hip", None) is not None

# ==================== Triton Configuration ====================
# Deferred configuration - only show if verbose mode or if compilation will happen
_VERBOSE_STARTUP = _env_flag('FRAMEPACK_VERBOSE_STARTUP', '0') if 'FRAMEPACK_VERBOSE_STARTUP' in os.environ else False

if _VERBOSE_STARTUP:
    print("\n" + "="*60)
    print("Configuring Triton and Torch Compile Optimizations")
    print("="*60)

# Configure Triton for ROCm/HIP
if IS_HIP_RUNTIME:
    if _VERBOSE_STARTUP:
        print("Detected ROCm/HIP runtime - configuring Triton for AMD GPUs")

    # Set Triton to use ROCm backend
    os.environ.setdefault('TRITON_INTERPRET', '0')  # Disable interpreter mode
    os.environ.setdefault('TRITON_PRINT_AUTOTUNING', '0')  # Disable autotuning logs

    # Enable Triton caching for faster recompilation
    triton_cache_dir = os.path.join(os.path.dirname(__file__), '.cache_rocm', 'triton')
    os.makedirs(triton_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = triton_cache_dir
    if _VERBOSE_STARTUP:
        print(f"  Triton cache directory: {triton_cache_dir}")

    # ROCm-specific Triton optimizations
    os.environ.setdefault('TRITON_ALWAYS_COMPILE', '0')  # Use cache when possible
    os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '0')  # Enable TunableOp
    os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')  # Enable runtime tuning
    os.environ.setdefault('PYTORCH_TUNABLEOP_FILENAME', os.path.join(os.path.dirname(__file__), 'tunableop_results.csv'))

    if _VERBOSE_STARTUP:
        print("  Enabled ROCm TunableOp for kernel auto-tuning")
else:
    if _VERBOSE_STARTUP:
        print("Detected CUDA runtime - using standard Triton configuration")
    os.environ.setdefault('TRITON_PRINT_AUTOTUNING', '0')

# Try to import Triton and check if it's available (lazy import to save startup time)
HAS_TRITON = False
def _check_triton():
    global HAS_TRITON
    try:
        import triton
        HAS_TRITON = True
        if _VERBOSE_STARTUP:
            triton_version = getattr(triton, '__version__', 'unknown')
            print(f"  Triton available: version {triton_version}")
        return True
    except ImportError:
        if _VERBOSE_STARTUP:
            print("  Warning: Triton not installed. Some optimizations will be unavailable.")
            print("  Install with: pip install triton")
        return False

# Only check Triton if torch.compile is enabled
if _env_flag('FRAMEPACK_USE_TORCH_COMPILE', '0'):
    _check_triton()

# Configure Torch Inductor (torch.compile backend) for Triton - deferred until first compile
_inductor_configured = False
def _configure_inductor():
    global _inductor_configured
    if _inductor_configured:
        return

    if hasattr(torch, '_inductor'):
        try:
            import torch._inductor.config as inductor_config

            # Enable Triton kernels in Inductor
            inductor_config.triton.cudagraphs = False  # Disable CUDA graphs for compatibility
            inductor_config.fallback_random = True  # Fallback for random ops
            inductor_config.triton.unique_kernel_names = True  # Better caching

            if IS_HIP_RUNTIME:
                # ROCm-specific Inductor settings
                inductor_config.triton.autotune_at_compile_time = True
                inductor_config.max_autotune = True  # Aggressive autotuning
                inductor_config.coordinate_descent_tuning = True  # Advanced tuning
                if _VERBOSE_STARTUP:
                    print("  Enabled aggressive Triton autotuning for ROCm")
            else:
                # CUDA-specific settings
                inductor_config.triton.cudagraphs = True  # CUDA graphs for NVIDIA
                if _VERBOSE_STARTUP:
                    print("  Enabled CUDA graphs for NVIDIA GPUs")

            if _VERBOSE_STARTUP:
                print("  Configured Torch Inductor for Triton kernels")
            _inductor_configured = True
        except Exception as e:
            print(f"  Warning: Could not configure Inductor: {e}")

if _VERBOSE_STARTUP:
    print("="*60 + "\n")

USE_TORCH_COMPILE = _env_flag('FRAMEPACK_USE_TORCH_COMPILE', '0')
TORCH_COMPILE_DYNAMIC = _env_flag('FRAMEPACK_TORCH_COMPILE_DYNAMIC', '0')
TORCH_COMPILE_FULLGRAPH = _env_flag('FRAMEPACK_TORCH_COMPILE_FULLGRAPH', '0')
TORCH_COMPILE_MODE = os.environ.get('FRAMEPACK_TORCH_COMPILE_MODE', 'max-autotune' if IS_HIP_RUNTIME else 'reduce-overhead').strip()

# Use MIGraphX backend if available on ROCm, otherwise use inductor
# MIGraphX provides better optimization for AMD GPUs including FP16/BF16 quantization
if IS_HIP_RUNTIME and HAS_TORCH_MIGRAPHX:
    default_backend = 'migraphx'
else:
    default_backend = 'inductor'
TORCH_COMPILE_BACKEND = os.environ.get('FRAMEPACK_TORCH_COMPILE_BACKEND', default_backend).strip()

# MIGraphX-specific options for AMD ROCm
USE_MIGRAPHX_BF16 = _env_flag('FRAMEPACK_MIGRAPHX_BF16', '0')  # Enable BF16 precision in MIGraphX
USE_MIGRAPHX_DEALLOCATE = _env_flag('FRAMEPACK_MIGRAPHX_DEALLOCATE', '0')  # Deallocate torch memory after compilation

# Enhanced Torch Compile Configuration (only show if verbose or disabled)
if _VERBOSE_STARTUP or not USE_TORCH_COMPILE:
    print("\nTorch Compile Configuration:")
    print(f"  Enabled: {USE_TORCH_COMPILE}")
    if USE_TORCH_COMPILE:
        print(f"  Mode: {TORCH_COMPILE_MODE}")
        print(f"  Backend: {TORCH_COMPILE_BACKEND}")
        print(f"  Dynamic shapes: {TORCH_COMPILE_DYNAMIC}")
        print(f"  Full graph: {TORCH_COMPILE_FULLGRAPH}")

        if TORCH_COMPILE_BACKEND == 'migraphx':
            print(f"  MIGraphX optimizations:")
            print(f"    - AMD ROCm graph optimization enabled")
            print(f"    - BF16 precision: {USE_MIGRAPHX_BF16}")
            print(f"    - Memory deallocation: {USE_MIGRAPHX_DEALLOCATE}")
            print(f"    - Provides FP16/BF16 quantization and kernel fusion")
        elif IS_HIP_RUNTIME:
            print(f"  ROCm optimizations: Aggressive autotuning enabled")
            print(f"  Triton backend: {'Available' if HAS_TRITON else 'Not available'}")
elif USE_TORCH_COMPILE:
    # Minimal startup message
    if TORCH_COMPILE_BACKEND == 'migraphx':
        print(f"\nTorch Compile: Enabled (MIGraphX backend - AMD ROCm optimized)")
    else:
        print(f"\nTorch Compile: Enabled ({TORCH_COMPILE_MODE} mode, {TORCH_COMPILE_BACKEND} backend)")

# Bitsandbytes configuration output
if USE_BITSANDBYTES and HAS_BITSANDBYTES:
    print(f"\nBitsandbytes 8-bit Optimization: Enabled (AMD ROCm)")
    print(f"  This will reduce memory usage and may improve performance")
    print(f"  Models are quantized during loading with BitsAndBytesConfig")
elif USE_BITSANDBYTES and not HAS_BITSANDBYTES:
    print(f"\nBitsandbytes 8-bit Optimization: Requested but not available")
    print(f"  Install AMD ROCm version with: pip install bitsandbytes")
    USE_BITSANDBYTES = False

KEEP_VAE_FP32_NORMALIZATION = _env_flag('FRAMEPACK_VAE_FP32_NORM', '1')
MAX_LATENT_CACHE_ITEMS = int(os.environ.get('FRAMEPACK_LATENT_CACHE_SIZE', '4'))
ENABLE_VAE_TILING = _env_flag('FRAMEPACK_VAE_TILING', '1')
VAE_TILE_SAMPLE_MIN = int(os.environ.get('FRAMEPACK_VAE_TILE_SAMPLE', '256'))
VAE_TILE_LATENT_MIN = int(os.environ.get('FRAMEPACK_VAE_TILE_LATENT', '64'))
VAE_DECODE_CHUNK = int(os.environ.get('FRAMEPACK_VAE_DECODE_CHUNK', '4'))
LATENTS_EXPORT_VERSION = 1
SKIP_IMMEDIATE_DECODE = _env_flag('FRAMEPACK_SKIP_IMMEDIATE_DECODE', '1')


def _torch_compile_kwargs(overrides=None):
    """Build torch.compile keyword arguments with optimizations."""
    kwargs = {
        'backend': TORCH_COMPILE_BACKEND,
    }

    if TORCH_COMPILE_MODE:
        kwargs['mode'] = TORCH_COMPILE_MODE

    if TORCH_COMPILE_DYNAMIC:
        kwargs['dynamic'] = True

    if TORCH_COMPILE_FULLGRAPH:
        kwargs['fullgraph'] = True

    # MIGraphX backend options for AMD ROCm
    if TORCH_COMPILE_BACKEND == 'migraphx':
        migraphx_options = {}
        if USE_MIGRAPHX_BF16:
            migraphx_options['bf16'] = True
        if USE_MIGRAPHX_DEALLOCATE:
            migraphx_options['deallocate'] = True

        if migraphx_options:
            kwargs['options'] = migraphx_options

    # Inductor backend options for CUDA/ROCm
    elif TORCH_COMPILE_MODE in ['max-autotune', 'max-autotune-no-cudagraphs']:
        kwargs['options'] = {
            'triton.cudagraphs': False if IS_HIP_RUNTIME else True,
            'max_autotune': True,
            'epilogue_fusion': True,
            'max_autotune_gemm_backends': 'TRITON,ATen' if HAS_TRITON else 'ATen',
        }

    if overrides:
        # Allow overrides to replace options dict or merge into it
        if 'options' in overrides and 'options' in kwargs:
            kwargs['options'].update(overrides.pop('options'))
        kwargs.update(overrides)

    return kwargs


def _get_compile_cache_path(name: str, compile_kwargs: dict) -> str:
    """Generate cache file path for compiled model."""
    import hashlib

    # Create a hash of compilation settings
    settings_str = f"{name}_{compile_kwargs.get('mode', 'default')}_{compile_kwargs.get('backend', 'inductor')}_{compile_kwargs.get('dynamic', False)}"
    settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]

    cache_dir = os.path.join(os.path.dirname(__file__), '.cache_rocm', 'compiled_models')
    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(cache_dir, f'{name}_{settings_hash}.pt')


def _save_compiled_model(module, cache_path: str, name: str):
    """Save compiled model to cache."""
    try:
        # Use torch.jit to save the compiled module
        # Note: This saves the traced/compiled graph, not the full module
        torch.save({
            'compiled': True,
            'timestamp': time.time(),
        }, cache_path + '.meta')
        if _VERBOSE_STARTUP:
            print(f'    Cached compilation metadata for {name}')
    except Exception as e:
        if _VERBOSE_STARTUP:
            print(f'    Warning: Could not cache {name}: {e}')


def _load_compiled_model_cache(cache_path: str, name: str) -> bool:
    """Check if cached compilation exists and is recent."""
    meta_path = cache_path + '.meta'

    if not os.path.exists(meta_path):
        return False

    try:
        meta = torch.load(meta_path, map_location='cpu')

        # Check if cache is less than 7 days old
        cache_age_days = (time.time() - meta.get('timestamp', 0)) / (24 * 3600)

        if cache_age_days > 7:
            if _VERBOSE_STARTUP:
                print(f'    Cache for {name} is {cache_age_days:.1f} days old, recompiling')
            return False

        if _VERBOSE_STARTUP:
            print(f'    Found cached compilation for {name} ({cache_age_days:.1f} days old)')
        return True

    except Exception as e:
        if _VERBOSE_STARTUP:
            print(f'    Warning: Could not load cache metadata: {e}')
        return False


def maybe_torch_compile(module, name: str, overrides=None):
    """
    Attempt to wrap a module with torch.compile if available.
    Implements compilation caching to avoid recompiling on every startup.

    Args:
        module: PyTorch module to compile
        name: Human-readable name for logging
        overrides: Optional dict to override compile kwargs

    Returns:
        Compiled module or original module if compilation fails
    """
    compile_fn = getattr(torch, 'compile', None)

    if not USE_TORCH_COMPILE:
        return module

    if compile_fn is None:
        if _VERBOSE_STARTUP:
            print(f'  ⚠ torch.compile unavailable for {name} - requires PyTorch 2.0+')
        return module

    # Configure Inductor on first compile
    _configure_inductor()

    try:
        compile_kwargs = _torch_compile_kwargs(overrides)
        cache_path = _get_compile_cache_path(name, compile_kwargs)

        # Check if we have a recent cached compilation
        # Note: torch.compile automatically caches kernels, but we track metadata
        has_cache = _load_compiled_model_cache(cache_path, name)

        # Reduced verbosity - only show compilation notice, not full details
        if has_cache:
            if _VERBOSE_STARTUP:
                print(f'  Loading cached compilation for {name}...')
            else:
                print(f'  {name}: Using cached compilation')
        else:
            if _VERBOSE_STARTUP:
                print(f'  Compiling {name}...')
                print(f'    Backend: {compile_kwargs["backend"]}')
                print(f'    Mode: {compile_kwargs.get("mode", "default")}')
            else:
                print(f'  Compiling {name} ({compile_kwargs.get("mode", "default")} mode)... (first time, will be cached)')

        compiled_module = compile_fn(module, **compile_kwargs)

        # Save cache metadata (actual kernel cache is handled by torch.compile/Triton)
        if not has_cache:
            _save_compiled_model(compiled_module, cache_path, name)

        if _VERBOSE_STARTUP and not has_cache:
            print(f'  ✓ Successfully compiled {name}')

        return compiled_module

    except RuntimeError as exc:
        if 'Triton' in str(exc) and not HAS_TRITON:
            print(f'  ⚠ {name} compilation skipped: Triton not available')
            if _VERBOSE_STARTUP:
                print(f'    Install triton for better performance.')
        else:
            print(f'  ⚠ {name} compilation failed, using eager mode')
            if _VERBOSE_STARTUP:
                print(f'    Error: {exc}')
        return module

    except Exception as exc:
        print(f'  ⚠ {name} compilation failed, using eager mode')
        if _VERBOSE_STARTUP:
            print(f'    Error: {exc}')
        return module


CHANNELS_LAST_3D = getattr(torch, 'channels_last_3d', torch.contiguous_format)
_latent_cache = OrderedDict()
_latent_cache_lock = threading.Lock()


NORM_FP32_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None or tensor.dim() != 5:
        return tensor
    return tensor.contiguous(memory_format=CHANNELS_LAST_3D)


def _to_gpu_channels_last(tensor: torch.Tensor, device, dtype=None):
    if tensor is None:
        return tensor
    tensor = tensor.to(device=device, dtype=dtype)
    return _ensure_channels_last_3d(tensor)


def _prune_latent_cache():
    while len(_latent_cache) > MAX_LATENT_CACHE_ITEMS:
        _latent_cache.popitem(last=False)


def get_cached_latents(key: str):
    if not key or MAX_LATENT_CACHE_ITEMS <= 0:
        return None
    with _latent_cache_lock:
        tensor = _latent_cache.get(key)
        if tensor is None:
            return None
        _latent_cache.move_to_end(key)
        return tensor.clone()


def set_cached_latents(key: str, tensor: torch.Tensor):
    if not key or MAX_LATENT_CACHE_ITEMS <= 0 or tensor is None:
        return
    with _latent_cache_lock:
        _latent_cache[key] = tensor.detach().clone()
        _latent_cache.move_to_end(key)
        _prune_latent_cache()


def image_to_cache_key(image_array: np.ndarray) -> str:
    if image_array is None:
        return ''
    return hashlib.sha1(image_array.tobytes()).hexdigest()


def save_latent_segments(job_id: str, segments: list, metadata: dict, directory: str) -> str | None:
    if not segments:
        return None

    package = {
        'metadata': metadata,
        'segments': [
            {
                'latent_padding': int(seg['latent_padding']),
                'is_last_section': bool(seg['is_last_section']),
                'generated_latents': seg['generated_latents'],
            }
            for seg in segments
        ],
    }

    os.makedirs(directory, exist_ok=True)
    latents_path = os.path.join(directory, f'{job_id}_latents.pt')
    torch.save(package, latents_path)
    return latents_path


def _wrap_norm_module_fp32(module: torch.nn.Module):
    if hasattr(module, '_framepack_fp32_norm'):
        return

    module.to(dtype=torch.float32)
    original_forward = module.forward

    def _forward_fp32_norm(self, *args, **kwargs):
        if not args or not torch.is_tensor(args[0]):
            return original_forward(*args, **kwargs)
        input_tensor = args[0]
        target_dtype = input_tensor.dtype
        target_device = input_tensor.device

        # Ensure module parameters are on the same device as input
        if hasattr(self, 'weight') and self.weight is not None:
            if self.weight.device != target_device:
                self.to(device=target_device)

        converted_args = list(args)
        converted_args[0] = input_tensor.to(dtype=torch.float32)
        output = original_forward(*converted_args, **kwargs)
        if torch.is_tensor(output):
            return output.to(dtype=target_dtype)
        if isinstance(output, (tuple, list)):
            converted = [o.to(dtype=target_dtype) if torch.is_tensor(o) else o for o in output]
            return type(output)(converted)
        return output

    module.forward = _forward_fp32_norm.__get__(module, module.__class__)
    module._framepack_fp32_norm = True


def configure_vae_inference(vae_model: torch.nn.Module, target_device, apply_compile=True):
    vae_model.to(device=target_device, dtype=torch.float16)
    vae_model.to(memory_format=CHANNELS_LAST_3D)

    # Apply torch.compile first if enabled
    if apply_compile:
        vae_model = maybe_torch_compile(vae_model, 'Autoencoder VAE', overrides={'mode': 'max-autotune'})

    # Apply FP32 normalization wrapper AFTER torch.compile (or skip if torch.compile is active)
    # This is because torch.compile creates a wrapper that makes runtime device movement impossible
    if KEEP_VAE_FP32_NORMALIZATION and not (apply_compile and USE_TORCH_COMPILE):
        for module in vae_model.modules():
            if isinstance(module, NORM_FP32_TYPES):
                _wrap_norm_module_fp32(module)
        print("Applied FP32 normalization wrappers to VAE")
    elif KEEP_VAE_FP32_NORMALIZATION and apply_compile and USE_TORCH_COMPILE:
        print("Skipping FP32 normalization wrappers (incompatible with torch.compile)")

    return vae_model

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    transformer.to(gpu)

vae = configure_vae_inference(vae, target_device=gpu, apply_compile=True)
if ENABLE_VAE_TILING:
    vae.enable_tiling()

# Optionally convert transformer to AMD TransformerEngine for better performance
if USE_AMD_TE and HAS_AMD_TE and IS_HIP_RUNTIME:
    print('\n' + '='*60)
    print('AMD TransformerEngine Conversion')
    print('='*60)
    print('ℹ Converting model to use AMD TE optimizations')
    print('  - Optimized FP16 kernels for Linear layers')
    print('  - Fused LayerNorm operations')
    print('  - Automatic dtype casting for mixed precision')
    print('='*60)

    try:
        # Convert model to TE
        # Note: Caching is disabled - model structure must be converted each time
        transformer = convert_model_to_te(
            transformer,
            verbose=True,
            cache_path=None,  # Caching disabled
            force_convert=False
        )

        print('='*60)
        print('✓ Transformer successfully converted to AMD TE')
        print('  Expected speedup: 20-30% on non-FP8 GPUs')
        print('='*60 + '\n')
    except Exception as e:
        print(f'⚠ Failed to convert Transformer to AMD TE: {e}')
        print('  Continuing with standard PyTorch implementation')
        import traceback
        traceback.print_exc()

if high_vram:
    # torch.compile currently only makes sense when the transformer can stay resident on the GPU
    transformer = maybe_torch_compile(transformer, 'Hunyuan Transformer')
elif USE_TORCH_COMPILE:
    print('Skipping torch.compile for transformer because low-VRAM swap mode is active.')

stream = None  # Will be initialized per generation

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# REST API configuration for latent processing
LATENT_API_ENABLED = _env_flag('FRAMEPACK_LATENT_API_ENABLED', '1')  # Disabled by default
LATENT_API_URL = os.environ.get('FRAMEPACK_LATENT_API_URL', 'http://localhost:7861')  # Different port from gradio


def send_latents_to_api(latents_file: str, job_id: str, verbose: bool = True) -> Optional[str]:
    """
    Send saved latents to REST API for final video generation.

    Args:
        latents_file: Path to saved latents .pt file
        job_id: Job identifier
        verbose: Print verbose output

    Returns:
        Path to final generated video or None on failure
    """
    try:
        import requests

        if verbose:
            print(f'\n{"="*60}')
            print(f'Sending latents to API for final video generation')
            print(f'  API URL: {LATENT_API_URL}')
            print(f'  Latents file: {latents_file}')
            print(f'{"="*60}')

        # Prepare request payload
        payload = {
            'latents_path': latents_file,
            'output_name': f'{job_id}_final.mp4',
            'enable_slicing': False,
            'verbose': verbose
        }

        # Send POST request to API
        response = requests.post(
            f'{LATENT_API_URL}/process',
            json=payload,
            timeout=300  # 5 minute timeout
        )

        if response.status_code == 200:
            result = response.json()
            output_path = result.get('output_path')

            if verbose:
                print(f'\n✓ API processing successful!')
                print(f'  Output: {output_path}')
                print(f'  Total frames: {result.get("total_frames")}')
                print(f'  Duration: {result.get("duration")} seconds')
                print(f'  Resolution: {result.get("resolution")}')
                print(f'  Processing time: {result.get("processing_time"):.2f}s')
                print(f'{"="*60}\n')

            # Ensure the path is absolute for Gradio to find it
            if output_path and not os.path.isabs(output_path):
                output_path = os.path.abspath(output_path)

            return output_path
        else:
            error_msg = f'API returned error: {response.status_code}'
            try:
                error_detail = response.json()
                error_msg = f'{error_msg} - {error_detail.get("message", "Unknown error")}'
            except:
                pass

            if verbose:
                print(f'✗ {error_msg}')
            return None

    except requests.exceptions.ConnectionError:
        if verbose:
            print(f'✗ Failed to connect to API at {LATENT_API_URL}')
            print(f'  Make sure the API server is running:')
            print(f'  python process_saved_latents_api.py --port 7861')
        return None
    except requests.exceptions.Timeout:
        if verbose:
            print(f'✗ API request timed out (>300s)')
        return None
    except Exception as e:
        if verbose:
            print(f'✗ Error communicating with API: {e}')
        return None


def flush_rocm_allocator(stage: str = '', min_resident_gb: float = 0.0) -> bool:
    """Force ROCm/HIP BlockAllocator to release cached blocks back to the driver."""
    if not IS_HIP_RUNTIME or not torch.cuda.is_available():
        return False

    torch.cuda.empty_cache()

    try:
        hip = ctypes.CDLL('libamdhip64.so')
    except OSError as exc:
        if stage:
            print(f'[{stage}] HIP mempool trim skipped: {exc}')
        return False

    trim_fn = getattr(hip, 'hipMemPoolTrimTo', None)
    get_pool_fn = getattr(hip, 'hipDeviceGetDefaultMemPool', None)
    if trim_fn is None or get_pool_fn is None:
        if stage:
            print(f'[{stage}] HIP mempool trim unsupported on this runtime.')
        return False

    trim_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    trim_fn.restype = ctypes.c_int
    get_pool_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    get_pool_fn.restype = ctypes.c_int

    pool = ctypes.c_void_p()
    device_index = gpu.index if isinstance(gpu, torch.device) and gpu.index is not None else torch.cuda.current_device()

    status = get_pool_fn(ctypes.byref(pool), ctypes.c_int(device_index))
    if status != 0:
        if stage:
            print(f'[{stage}] hipDeviceGetDefaultMemPool failed with status {status}')
        return False

    bytes_to_keep = max(0, int(min_resident_gb * (1024 ** 3)))
    status = trim_fn(pool, ctypes.c_size_t(bytes_to_keep))
    trimmed = status == 0
    if stage:
        outcome = 'trimmed' if trimmed else f'failed ({status})'
        print(f'[{stage}] ROCm mempool {outcome}')
    return trimmed


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_fbcache, use_magcache, mp4_crf):
    unlimited_length = total_second_length <= 0
    if unlimited_length:
        total_latent_sections = None
    else:
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    flush_rocm_allocator('worker-start')

    latent_segments: list[dict] = []
    latent_metadata = {
        'job_id': job_id,
        'latent_window_size': latent_window_size,
        'mp4_crf': mp4_crf,
        'fps': 30,
        'version': LATENTS_EXPORT_VERSION,
    }
    latent_padding_history: list[int] = []

    first_block_cache = None
    if use_fbcache:
        first_block_cache = FirstBlockCache(FirstBlockCacheConfig())
        transformer.set_first_block_cache(first_block_cache)
        print(f'[FBCache] Enabled (threshold={first_block_cache.config.threshold:.4f})')
    else:
        transformer.set_first_block_cache(None)

    magcache: Optional[MagCacheVideo] = None
    transformer.set_magcache(None)

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, transformer
            )
            flush_rocm_allocator('post-unload/text-enc')

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Unload text encoders immediately after use (not needed anymore)
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2)
            flush_rocm_allocator('post-text-encoding')

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        latent_metadata['height'] = height
        latent_metadata['width'] = width
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        input_image_pt = _to_gpu_channels_last(input_image_pt, gpu, dtype=vae.dtype)

        image_cache_key = image_to_cache_key(input_image_np)

        if use_magcache:
            magcache_config = MagCacheConfig(
                enabled=True,
                threshold=0.24,
                max_skip_steps=6,
                retention_ratio=0.2,
                default_height=height,
                cache_on_cpu=not high_vram,
            )
            magcache = MagCacheVideo(magcache_config)
            magcache.configure(num_steps=steps, video_height=height)
            transformer.set_magcache(magcache)
            print(f'[MagCache] Enabled (threshold={magcache_config.threshold:.2f}, max_skip={magcache_config.max_skip_steps})')
        else:
            transformer.set_magcache(None)

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        start_latent = get_cached_latents(image_cache_key)
        if start_latent is not None:
            start_latent = _to_gpu_channels_last(start_latent, gpu, dtype=vae.dtype)
            print('Reusing cached VAE latents for identical input frame.')
        else:
            start_latent = vae_encode(input_image_pt, vae)
            start_latent = _ensure_channels_last_3d(start_latent)
            set_cached_latents(image_cache_key, start_latent)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Unload image encoder after use (not needed anymore)
        if not high_vram:
            unload_complete_models(image_encoder)
            flush_rocm_allocator('post-clip-vision')

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=transformer.dtype,
            device=gpu,
        )
        history_latents = _ensure_channels_last_3d(history_latents)
        history_pixels = None
        total_generated_latent_frames = 0

        if unlimited_length:
            latent_paddings = itertools.chain([3], itertools.repeat(2))
        else:
            latent_paddings = reversed(range(total_latent_sections))

        if not unlimited_length and total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        first_section = True
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            latent_padding_history.append(int(latent_padding))

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                flush_rocm_allocator('post-unload/transformer')
                # Log memory status on first iteration for debugging
                if first_section:
                    log_memory_status(gpu, prefix="[Before Transformer Load] ")
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_magcache:
                transformer.initialize_teacache(enable_teacache=False)
            elif use_fbcache:
                transformer.initialize_teacache(enable_teacache=False)
            elif use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            generated_latents = _ensure_channels_last_3d(generated_latents)
            if first_block_cache is not None:
                hits = first_block_cache.stats.get('hits', 0)
                misses = first_block_cache.stats.get('misses', 0)
                ratio = first_block_cache._last_diff if hasattr(first_block_cache, '_last_diff') else None
                ratio_str = f', last diff={ratio:.4f}' if isinstance(ratio, (int, float)) else ''
                print(f'[FBCache] stats: hits={hits}, misses={misses}{ratio_str}')
            if magcache is not None:
                stats = magcache.stats
                skips = stats.get('skips', 0)
                full = stats.get('full_pass', 0)
                print(f'[MagCache] stats: skips={skips}, full_pass={full}, last_err={magcache.last_error:.4f}')

            segment_latents = generated_latents
            if is_last_section:
                segment_latents = torch.cat([start_latent.to(segment_latents), segment_latents], dim=2)

            latent_segments.append({
                'latent_padding': int(latent_padding),
                'is_last_section': bool(is_last_section),
                'generated_latents': segment_latents.detach().to('cpu'),
            })

            total_generated_latent_frames += int(segment_latents.shape[2])
            first_section = False

            if SKIP_IMMEDIATE_DECODE:
                if is_last_section:
                    break
                continue

            history_latents = torch.cat([segment_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            real_history_latents = _ensure_channels_last_3d(real_history_latents)

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae)
                history_pixels = _ensure_channels_last_3d(history_pixels).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae)
                current_pixels = _ensure_channels_last_3d(current_pixels).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                history_pixels = _ensure_channels_last_3d(history_pixels)

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break

        latent_metadata['total_latent_frames'] = total_generated_latent_frames
        latent_metadata['latent_paddings'] = latent_padding_history
        latents_file = save_latent_segments(job_id, latent_segments, latent_metadata, outputs_folder)
        if latents_file:
            print(f'Latent segments saved to {latents_file}')
            stream.output_queue.push(('latents', latents_file))

            # Send to REST API for final video generation if enabled
            if LATENT_API_ENABLED:
                stream.output_queue.push(('progress', (None, 'Sending latents to API for final video generation...', make_progress_bar_html(95, 'Generating final video'))))
                try:
                    final_video = send_latents_to_api(latents_file, job_id, verbose=True)
                    if final_video:
                        print(f'Final video received from API: {final_video}')
                        stream.output_queue.push(('final_video', final_video))
                except Exception as e:
                    print(f'Warning: Failed to process latents via API: {e}')
                    traceback.print_exc()
    except:
        traceback.print_exc()
    finally:
        transformer.set_first_block_cache(None)
        transformer.set_magcache(None)
        # Clean up models and GPU memory after completion or exception
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, transformer
            )
            flush_rocm_allocator('worker-cleanup')

    # Print MIOpen fallback statistics
    MIOpenFallbackHandler.print_stats()

    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_fbcache, use_magcache, mp4_crf):
    global stream
    assert input_image is not None, 'No input image!'

    # Signal any existing stream to end before starting new one
    if stream is not None:
        try:
            stream.input_queue.push('end')
            # Give the old worker thread a moment to finish
            import time
            time.sleep(0.5)
        except:
            pass

    # Force GPU memory cleanup before starting new run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    flush_rocm_allocator('pre-generation-cleanup', min_resident_gb=0.0)

    # Clear torch dynamo cache if using torch.compile to prevent stale references
    if USE_TORCH_COMPILE and hasattr(torch, '_dynamo'):
        try:
            torch._dynamo.reset()
        except:
            pass

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_fbcache, use_magcache, mp4_crf)

    output_filename = None

    try:
        while True:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                output_filename = data
                # Force video component to update by providing explicit value
                yield gr.update(value=output_filename), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'latents':
                # Don't update output_filename here - latents are not a video file
                # The final_video flag will set the proper output_filename
                latents_filename = data
                desc = f'Latent segments saved to {os.path.basename(latents_filename)}'
                yield gr.update(value=None), gr.update(visible=False), desc, '', gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'final_video':
                # Final video from API - this is the complete video
                # Update output_filename so the 'end' flag handler will use this video
                output_filename = data
                desc = f'✓ Final video generated: {os.path.basename(output_filename)}'
                print(f'[DEBUG] Displaying final video: {output_filename}')
                print(f'[DEBUG] File exists: {os.path.exists(output_filename)}')
                # Update video component with the final video
                yield gr.update(value=output_filename), gr.update(visible=False), desc, '', gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'progress':
                preview, desc, html = data
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'end':
                yield gr.update(value=output_filename), gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                break
    except GeneratorExit:
        # Process generator was cancelled, signal worker to stop
        if stream is not None:
            stream.input_queue.push('end')
        raise


def end_process():
    if stream is not None:
        stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                use_fbcache = gr.Checkbox(label='Use First Block Cache', value=True, info='Experimental cache for transformer first block to reuse residuals between denoising steps.')
                use_magcache = gr.Checkbox(label='Use MagCache', value=False, info='Skip full transformer passes when residual magnitudes remain stable (experimental).')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=0, maximum=120, value=5, step=0.1, info="Set to 0 for unlimited generation.")
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=4, maximum=128, value=8 if not high_vram else 6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed. For 20-24GB VRAM, use 10GB+ to prevent BlockAllocator failures.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_fbcache, use_magcache, mp4_crf]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
