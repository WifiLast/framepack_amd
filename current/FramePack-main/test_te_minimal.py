#!/usr/bin/env python3
"""
Minimal TransformerEngine Test with hipBLASLt Properly Disabled

This script demonstrates the CORRECT way to disable hipBLASLt.
Environment variables MUST be set BEFORE importing TransformerEngine.
"""

import os
import sys

# ==================== CRITICAL: Set these BEFORE any PyTorch/TE imports ====================
print("Setting environment variables...")
os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'
os.environ['TE_HIPBLASLT_DISABLED'] = '1'
os.environ['NVTE_TORCH_COMPILE'] = '0'
os.environ['PYTORCH_HIPBLASLT'] = '0'
os.environ['ROCBLAS_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'

print("✅ Environment variables set BEFORE imports")
print("")

# ==================== Now safe to import ====================
print("Importing libraries...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)

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
