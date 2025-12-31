#!/usr/bin/env python3
"""
Debug script to understand exactly when and how TransformerEngine loads hipBLASLt
"""

import os
import sys

print("="*70)
print("Step 1: Setting environment variables")
print("="*70)

os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'
os.environ['TE_HIPBLASLT_DISABLED'] = '1'
os.environ['NVTE_TORCH_COMPILE'] = '0'
os.environ['PYTORCH_HIPBLASLT'] = '0'
os.environ['ROCBLAS_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'

print("Environment variables set:")
for var in ['NVTE_DISABLE_HIPBLASLT', 'TE_HIPBLASLT_DISABLED', 'NVTE_TORCH_COMPILE', 'PYTORCH_HIPBLASLT', 'ROCBLAS_TENSILE_LIBPATH']:
    print(f"  {var} = {os.environ.get(var, 'NOT SET')}")

print("\n" + "="*70)
print("Step 2: Importing PyTorch")
print("="*70)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

print("\n" + "="*70)
print("Step 3: Importing TransformerEngine")
print("="*70)

try:
    import transformer_engine.pytorch as te
    print("✅ TransformerEngine imported successfully")

    # Check if it has version
    if hasattr(te, '__version__'):
        print(f"   Version: {te.__version__}")

except Exception as e:
    print(f"❌ TransformerEngine import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Step 4: Testing TransformerEngine Linear layer")
print("="*70)

try:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("\nCreating TE Linear layer...")
    linear = te.Linear(
        in_features=512,
        out_features=512,
        device=device,
        bias=True
    )
    print("✅ Layer created")

    print("\nCreating test input...")
    x = torch.randn(2, 512, device=device, dtype=torch.float16)
    print("✅ Input created")

    print("\nRunning forward pass...")
    y = linear(x)
    torch.cuda.synchronize()
    print("✅ Forward pass successful!")

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    print("\n" + "="*70)
    print("🎉 SUCCESS - TransformerEngine is working!")
    print("="*70)

except RuntimeError as e:
    error_msg = str(e)
    print(f"\n❌ FAILED: {error_msg}")

    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)

    if "HIPBLASLT" in error_msg.upper():
        print("\n⚠️  hipBLASLt error detected!")
        print("\nThis means TransformerEngine is trying to use hipBLASLt despite env vars.")
        print("\nPossible reasons:")
        print("1. TransformerEngine was compiled with hipBLASLt hard-coded")
        print("2. Environment variables are being ignored")
        print("3. hipBLASLt library has compatibility issues with your GPU")

        print("\nChecking environment variables again:")
        for var in ['NVTE_DISABLE_HIPBLASLT', 'TE_HIPBLASLT_DISABLED', 'NVTE_TORCH_COMPILE']:
            print(f"  {var} = {os.environ.get(var, 'NOT SET')}")

    if "Could not load" in error_msg:
        print("\n⚠️  Tensile library error detected!")
        print("\nThe rocBLAS backend can't find Tensile libraries.")
        print("\nSolution:")
        print("  sudo apt-get install --reinstall rocblas")

    print("\nFull error:")
    import traceback
    traceback.print_exc()

    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
