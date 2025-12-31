#!/usr/bin/env python3
"""
ROCm and hipBLASLt Diagnostic Script

This script diagnoses common issues with ROCm, hipBLASLt, and TransformerEngine.
"""

import os
import sys
import subprocess

def check_file(path, description):
    """Check if a file exists and report its size."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"

    if exists:
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"{status} {description}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"{status} {description}: {path} NOT FOUND")

    return exists

def check_directory_files(directory, pattern, description):
    """Check for files matching a pattern in a directory."""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []

    import glob
    files = glob.glob(os.path.join(directory, pattern))

    if files:
        print(f"✅ {description} in {directory}:")
        for f in files:
            size_mb = os.path.getsize(f) / (1024**2)
            print(f"   - {os.path.basename(f)} ({size_mb:.1f} MB)")
    else:
        print(f"❌ No {description} found in {directory}")

    return files

def main():
    print("="*70)
    print("  ROCm and hipBLASLt Diagnostic Report")
    print("="*70)

    # Check ROCm installation
    print("\n1. ROCm Installation:")
    print("-"*70)

    rocm_paths = ['/opt/rocm', '/opt/rocm-6.4.2', '/opt/rocm-6.4.0']
    rocm_path = None

    for path in rocm_paths:
        if os.path.exists(path):
            print(f"✅ Found ROCm: {path}")
            rocm_path = path
            break

    if not rocm_path:
        print("❌ ROCm not found in standard locations")
        return 1

    # Check critical libraries
    print("\n2. Critical Libraries:")
    print("-"*70)

    check_file(f"{rocm_path}/lib/libhipblaslt.so", "hipBLASLt library")
    check_file(f"{rocm_path}/lib/librocblas.so", "rocBLAS library")
    check_file(f"{rocm_path}/lib/libamdhip64.so", "HIP runtime")

    # Check Tensile libraries
    print("\n3. Tensile Libraries (rocBLAS backend):")
    print("-"*70)

    tensile_dir = f"{rocm_path}/lib/rocblas/library"
    if os.path.exists(tensile_dir):
        print(f"✅ Tensile directory exists: {tensile_dir}")

        # Check for specific architectures
        architectures = ['gfx1100', 'gfx1030', 'gfx90a', 'gfx942']
        found_archs = []

        for arch in architectures:
            lazy_file = f"{tensile_dir}/TensileLibrary_lazy_{arch}.dat"
            yaml_file = f"{tensile_dir}/TensileLibrary_lazy_{arch}.yaml"

            if os.path.exists(lazy_file) or os.path.exists(yaml_file):
                found_archs.append(arch)
                check_file(lazy_file, f"Tensile {arch} (dat)")
                check_file(yaml_file, f"Tensile {arch} (yaml)")

        if not found_archs:
            print("❌ No Tensile libraries found for any architecture!")
            print("   This will cause rocBLAS operations to fail")

        # List all files in tensile directory
        print("\n   All files in Tensile directory:")
        try:
            for f in os.listdir(tensile_dir):
                full_path = os.path.join(tensile_dir, f)
                if os.path.isfile(full_path):
                    size_mb = os.path.getsize(full_path) / (1024**2)
                    print(f"   - {f} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Error listing directory: {e}")
    else:
        print(f"❌ Tensile directory not found: {tensile_dir}")
        print("   This is a CRITICAL issue - rocBLAS cannot function without Tensile")

    # Check GPU info
    print("\n4. GPU Detection:")
    print("-"*70)

    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Look for gfx architecture
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower() or 'name' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print("❌ rocminfo command failed")
    except FileNotFoundError:
        print("❌ rocminfo not found (not in PATH)")
    except Exception as e:
        print(f"❌ Error running rocminfo: {e}")

    # Check Python environment
    print("\n5. Python Environment:")
    print("-"*70)

    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")

    # Check installed packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        if hasattr(torch.version, 'hip'):
            print(f"✅ HIP version: {torch.version.hip}")

        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ No GPU detected by PyTorch")
    except ImportError:
        print("❌ PyTorch not installed")

    try:
        import transformer_engine
        print(f"✅ TransformerEngine installed: {transformer_engine.__version__}")
    except ImportError:
        print("⚠️  TransformerEngine not installed")
    except AttributeError:
        print("✅ TransformerEngine installed (version unknown)")

    # Check environment variables
    print("\n6. Environment Variables:")
    print("-"*70)

    env_vars = [
        'ROCM_PATH',
        'HIP_VISIBLE_DEVICES',
        'PYTORCH_HIPBLASLT',
        'HIPBLASLT_TENSILE_LIBPATH',
        'NVTE_DISABLE_HIPBLASLT',
        'TE_HIPBLASLT_DISABLED',
        'ROCBLAS_TENSILE_LIBPATH',
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"   {var} not set")

    # Recommendations
    print("\n7. Recommendations:")
    print("-"*70)

    print("\n🔧 To fix the missing Tensile library issue:")
    print("   1. Check ROCm installation completeness:")
    print("      sudo apt-get install --reinstall rocblas")
    print("      sudo apt-get install --reinstall hipblaslt")
    print("\n   2. Verify file permissions:")
    print(f"      ls -la {tensile_dir}/")
    print("\n   3. Set ROCBLAS_TENSILE_LIBPATH in your script:")
    print(f"      os.environ['ROCBLAS_TENSILE_LIBPATH'] = '{tensile_dir}'")

    print("\n🔧 To disable hipBLASLt properly:")
    print("   Add these lines BEFORE importing TransformerEngine:")
    print("      os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'")
    print("      os.environ['TE_HIPBLASLT_DISABLED'] = '1'")
    print("      os.environ['NVTE_TORCH_COMPILE'] = '0'")

    print("\n" + "="*70)
    print("Diagnostic complete!")
    print("="*70)

if __name__ == '__main__':
    sys.exit(main())
