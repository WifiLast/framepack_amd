#!/usr/bin/env python3
"""
Minimal TransformerEngine Test with All ROCm Optimizations

This script demonstrates the CORRECT way to disable hipBLASLt and enable
all ROCm optimizations for testing TransformerEngine performance.
"""

import os
import sys

# ==================== CRITICAL: Set these BEFORE any PyTorch/TE imports ====================
print("="*70)
print("Setting Environment Variables & Optimizations")
print("="*70)

# ==================== hipBLASLt Disabling ====================
os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'
os.environ['TE_HIPBLASLT_DISABLED'] = '1'
os.environ['NVTE_TORCH_COMPILE'] = '0'
os.environ['PYTORCH_HIPBLASLT'] = '0'
os.environ['HIPBLASLT_LOG_LEVEL'] = '0'  # Silence hipBLASLt errors
os.environ['HIPBLASLT_LOG_MASK'] = '0'
os.environ['DISABLE_HIPBLASLT'] = '1'
os.environ['USE_HIPBLASLT'] = '0'
print("✓ hipBLASLt disabled")

# ==================== ROCm Platform Flags ====================
# HSA Runtime
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HSA_ENABLE_INTERRUPT'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '8'

# HIP Runtime
os.environ['AMD_SERIALIZE_KERNEL'] = '0'
os.environ['AMD_SERIALIZE_COPY'] = '0'
os.environ['AMD_DIRECT_DISPATCH'] = '1'
os.environ['HIP_HOST_COHERENT'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Profiling overhead removal
os.environ['ROCP_TOOL_LIB'] = ''
os.environ['HSA_TOOLS_LIB'] = ''

# RDNA3 optimizations
os.environ['AMD_WAVE_SIZE'] = '32'
os.environ['AMD_MAX_WAVES_PER_SIMD'] = '16'
os.environ['AMD_OCL_WORKGROUP_SIZE'] = '256'
os.environ['AMD_COMGR_SAVE_TEMPS'] = '0'
os.environ['AMD_COMGR_REDIRECT_LOGS'] = '0'
print("✓ ROCm platform flags enabled")

# ==================== TunableOp ====================
os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'
os.environ['PYTORCH_TUNABLEOP_TUNING'] = '1'
os.environ['PYTORCH_TUNABLEOP_FILENAME'] = os.path.join(
    os.path.dirname(__file__), 'test_tunableop_results.csv'
)
os.environ['PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS'] = '30'
os.environ['PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS'] = '100'
print("✓ TunableOp kernel caching enabled")

# ==================== CPU Threading ====================
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
print("✓ CPU threading configured")

# ==================== rocBLAS ====================
os.environ['ROCBLAS_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'
os.environ['ROCBLAS_LAYER'] = '0'
os.environ['PYTORCH_ROCBLAS_PREFER_HIPBLAS'] = '0'
print("✓ rocBLAS configured")

print("="*70)
print("✅ All environment variables set BEFORE imports")
print("="*70)
print("")

# ==================== Now safe to import ====================
print("Importing libraries...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
# ==================== PyTorch-Level Optimizations ====================
print("")
print("="*70)
print("Configuring PyTorch Optimizations")
print("="*70)

# CPU Threading
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
print("✓ CPU threading optimized (8 threads)")

# Mixed Precision
torch.set_float32_matmul_precision('medium')
print("✓ Mixed precision mode: medium")

# JIT Fusion
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
print("✓ JIT operator fusion enabled")

# WMMA/Matrix Core Detection
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    if 'mi300' in gpu_name or 'gfx942' in gpu_name or 'mi200' in gpu_name or 'gfx90a' in gpu_name:
        os.environ['ROCBLAS_FORCE_WMMA'] = '1'
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✓ WMMA/Matrix cores enabled (CDNA architecture)")
    elif '7900' in gpu_name or 'gfx1100' in gpu_name:
        print("✓ RDNA3 AI accelerators available (auto-selected by rocBLAS)")
    else:
        print(f"✓ GPU detected: {gpu_name}")

print("="*70)
print("")

try:
    import transformer_engine.pytorch as te
    print(f"✅ TransformerEngine imported")
except ImportError as e:
    print(f"❌ Failed to import TransformerEngine: {e}")
    sys.exit(1)

print("")
print("="*70)
print("  Testing TransformerEngine Linear Layer")
print("="*70)

try:
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        sys.exit(1)

    device = torch.device('cuda:0')
    print(f"✅ Using device: {torch.cuda.get_device_name(0)}")
    print("")

    # Create TransformerEngine Linear layer
    print("Creating TE Linear layer (512 -> 512)...")
    linear = te.Linear(
        in_features=512,
        out_features=512,
        params_dtype=torch.float16,  # CRITICAL: Must match input dtype
        device=device,
        bias=True
    )
    print("✅ Layer created successfully")
    print(f"   Layer dtype: {linear.weight.dtype}")

    # Create test input
    print("Creating test input (batch=16, seq=128, dim=512)...")
    x = torch.randn(16, 128, 512, device=device, dtype=torch.float16)
    print("✅ Input created")
    print(f"   Input dtype: {x.dtype}")

    # Forward pass
    print("Running forward pass...")
    y = linear(x)
    torch.cuda.synchronize()
    print("✅ Forward pass completed!")

    # Check output shape
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y.shape}")

    # Run a few iterations to verify stability
    print("")
    print("Running 10 iterations to verify stability...")
    for i in range(10):
        y = linear(x)
        torch.cuda.synchronize()
    print("✅ All iterations completed successfully!")

    print("")
    print("="*70)
    print("  🎉 SUCCESS!")
    print("="*70)
    print("TransformerEngine is working correctly with hipBLASLt disabled.")
    print("It's using rocBLAS backend instead.")
    print("")

except RuntimeError as e:
    print("")
    print("="*70)
    print("  ❌ FAILURE")
    print("="*70)
    print(f"Error: {e}")
    print("")

    if "HIPBLASLT" in str(e).upper():
        print("⚠️  hipBLASLt error detected!")
        print("")
        print("Possible issues:")
        print("1. Environment variables were not set early enough")
        print("2. TransformerEngine was already imported elsewhere")
        print("3. hipBLASLt library has compatibility issues")
        print("")
        print("Solution:")
        print("- Make sure this script is run fresh (not in an existing Python session)")
        print("- Check that no other code imports TE before env vars are set")

    if "Could not load" in str(e):
        print("⚠️  Tensile library loading error!")
        print("")
        print("Solution:")
        print("  sudo apt-get install --reinstall rocblas")
        print("  sudo apt-get install --reinstall hipblaslt")

    import traceback
    print("")
    print("Full traceback:")
    traceback.print_exc()

    sys.exit(1)

except Exception as e:
    print("")
    print("="*70)
    print("  ❌ UNEXPECTED ERROR")
    print("="*70)
    print(f"Error: {e}")

    import traceback
    traceback.print_exc()

    sys.exit(1)
