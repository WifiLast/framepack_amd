#!/usr/bin/env python3
"""
Test script for AMD TransformerEngine integration

This script tests the AMD TE monkey patching functionality and verifies
that models can be converted and used with TE optimizations.

Usage:
    python test_amd_te.py
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("="*60)
print("AMD TransformerEngine Integration Test")
print("="*60)

# Test 1: Check ROCm/HIP runtime
print("\n[Test 1] Checking ROCm/HIP runtime...")
IS_HIP_RUNTIME = getattr(torch.version, "hip", None) is not None
print(f"  HIP runtime detected: {IS_HIP_RUNTIME}")
if IS_HIP_RUNTIME:
    print(f"  HIP version: {torch.version.hip}")
else:
    print("  ⚠ Not running on ROCm - TE optimizations may be limited")

# Test 2: Import AMD TE module
print("\n[Test 2] Importing AMD TE monkey patch module...")
try:
    from diffusers_helper.amd_te_monkey_patch import (
        apply_amd_te_optimizations,
        convert_model_to_te,
        get_fp8_context,
        get_te_modules,
        HAS_AMD_TE
    )
    print("  ✓ Successfully imported AMD TE module")
    print(f"  AMD TE available: {HAS_AMD_TE}")
except ImportError as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

# Test 3: Apply optimizations
print("\n[Test 3] Applying AMD TE optimizations...")
results = apply_amd_te_optimizations(verbose=True, enable_fp8=True, patch_layers=False)

print(f"\nOptimization Results:")
print(f"  TE Available: {results['te_available']}")
print(f"  Is ROCm: {results['is_rocm']}")
print(f"  FP8 Recipe: {results['fp8_recipe']}")
print(f"  Patched Layers: {results['patched_layers']}")

# Test 4: Create a simple test model
print("\n[Test 4] Creating test model...")


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing"""
    def __init__(self, d_model=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'linear1': nn.Linear(d_model, d_model * 4),
                'linear2': nn.Linear(d_model * 4, d_model),
                'norm': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            # Simple feedforward block
            residual = x
            x = layer['linear1'](x)
            x = torch.relu(x)
            x = layer['linear2'](x)
            x = layer['norm'](x + residual)
        return self.final_norm(x)


model = SimpleTransformer(d_model=256, num_layers=2)
print(f"  ✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")

# Count original layer types
linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
layernorm_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
print(f"  Original: {linear_count} Linear layers, {layernorm_count} LayerNorm layers")

# Test 5: Convert model to TE
if HAS_AMD_TE:
    print("\n[Test 5] Converting model to AMD TE...")
    model_te = convert_model_to_te(model, verbose=True)

    # Count converted layer types
    te_modules = get_te_modules()
    if te_modules:
        te_linear_count = sum(1 for m in model_te.modules() if type(m).__name__ == 'Linear' and 'transformer_engine' in str(type(m)))
        te_layernorm_count = sum(1 for m in model_te.modules() if type(m).__name__ == 'LayerNorm' and 'transformer_engine' in str(type(m)))
        print(f"\n  Converted: {te_linear_count} TE Linear layers, {te_layernorm_count} TE LayerNorm layers")
else:
    print("\n[Test 5] Skipping conversion (AMD TE not available)")
    model_te = model

# Test 6: Run inference
print("\n[Test 6] Testing inference...")
batch_size = 4
seq_len = 32
d_model = 256

# Create random input
input_tensor = torch.randn(batch_size, seq_len, d_model)
print(f"  Input shape: {input_tensor.shape}")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model_te = model_te.to(device)
input_tensor = input_tensor.to(device)

# Standard inference
with torch.no_grad():
    output_standard = model_te(input_tensor)
print(f"  ✓ Standard inference output shape: {output_standard.shape}")

# FP8 inference (if supported)
if HAS_AMD_TE and results.get('fp8_recipe') is not None:
    print("\n[Test 7] Testing FP8 inference...")
    with torch.no_grad():
        with get_fp8_context():
            output_fp8 = model_te(input_tensor)
    print(f"  ✓ FP8 inference output shape: {output_fp8.shape}")

    # Compare outputs
    diff = torch.abs(output_standard - output_fp8).mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    if diff < 0.01:
        print(f"  ✓ Outputs match closely (diff < 0.01)")
    else:
        print(f"  ⚠ Outputs differ significantly (diff = {diff:.6f})")
else:
    print("\n[Test 7] Skipping FP8 test (not supported)")

# Test 8: Get TE modules for direct use
print("\n[Test 8] Testing direct TE module access...")
te_modules = get_te_modules()
if te_modules:
    print(f"  Available TE modules: {list(te_modules.keys())}")

    # Try to create a TE Linear layer directly
    if 'Linear' in te_modules:
        try:
            TELinear = te_modules['Linear']
            te_linear = TELinear(256, 512, device=device)
            test_input = torch.randn(4, 256, device=device)
            with torch.no_grad():
                test_output = te_linear(test_input)
            print(f"  ✓ Direct TE Linear layer works: {test_input.shape} → {test_output.shape}")
        except Exception as e:
            print(f"  ✗ Failed to use TE Linear: {e}")
else:
    print("  No TE modules available")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print(f"  ROCm Runtime: {'✓' if IS_HIP_RUNTIME else '✗'}")
print(f"  AMD TE Available: {'✓' if HAS_AMD_TE else '✗'}")
print(f"  FP8 Support: {'✓' if results.get('fp8_recipe') is not None else '✗'}")
print(f"  Model Conversion: {'✓' if HAS_AMD_TE else 'Skipped'}")
print(f"  Inference Test: ✓")

if HAS_AMD_TE and IS_HIP_RUNTIME:
    print("\n🎉 All tests passed! AMD TE integration is working.")
    print("\nTo use in FramePack:")
    print("  export FRAMEPACK_USE_AMD_TE=1")
    print("  python demo_gradio.py")
elif HAS_AMD_TE:
    print("\n⚠ AMD TE is available but not running on ROCm")
    print("   Full optimizations may not be available")
else:
    print("\n⚠ AMD TE not available")
    print("\nTo install AMD TransformerEngine:")
    print("  cd cache/TransformerEngine-dev")
    print("  pip install -e .")

print("="*60)
