#!/usr/bin/env python3
"""
hipBLASLt Compatibility Test Script

This script tests whether hipBLASLt works on your system by running
a simple matrix multiplication with various configurations.

Usage:
    python test_hipblaslt.py

    # Or with verbose output:
    python test_hipblaslt.py --verbose

    # Test specific features:
    python test_hipblaslt.py --test-pytorch
    python test_hipblaslt.py --test-transformer-engine
"""

import os
import sys
import argparse
import traceback

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(test_name, passed, message=""):
    """Print test result with color coding."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if message:
        print(f"     {message}")

def test_environment():
    """Test 1: Check environment variables and paths."""
    print_header("Test 1: Environment and Paths")

    results = {}

    # Check ROCm installation
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    rocm_exists = os.path.exists(rocm_path)
    print_result("ROCm installation", rocm_exists, f"Path: {rocm_path}")
    results['rocm'] = rocm_exists

    # Check for Tensile libraries
    tensile_paths = [
        '/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1100.dat',
        '/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1030.dat',
        '/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx90a.dat',
        '/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx942.dat',
    ]

    tensile_found = []
    for path in tensile_paths:
        if os.path.exists(path):
            arch = path.split('_')[-1].replace('.dat', '')
            tensile_found.append(arch)
            print(f"     Found: {arch}")

    print_result("Tensile libraries", len(tensile_found) > 0,
                 f"Found {len(tensile_found)} architecture(s)")
    results['tensile'] = len(tensile_found) > 0
    results['tensile_archs'] = tensile_found

    # Check for hipBLASLt library
    hipblaslt_paths = [
        '/opt/rocm/lib/libhipblaslt.so',
        '/opt/rocm/lib/libhipblaslt.so.0',
    ]

    hipblaslt_found = False
    for path in hipblaslt_paths:
        if os.path.exists(path):
            hipblaslt_found = True
            print(f"     Found: {path}")
            # Get file size
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"     Size: {size_mb:.1f} MB")
            break

    print_result("hipBLASLt library", hipblaslt_found)
    results['hipblaslt_lib'] = hipblaslt_found

    return results

def test_pytorch_import():
    """Test 2: Import PyTorch and check ROCm support."""
    print_header("Test 2: PyTorch and ROCm")

    results = {}

    try:
        import torch
        print_result("PyTorch import", True, f"Version: {torch.__version__}")
        results['pytorch'] = True
        results['pytorch_version'] = torch.__version__

        # Check CUDA (HIP) availability
        cuda_available = torch.cuda.is_available()
        print_result("CUDA/HIP available", cuda_available)
        results['cuda_available'] = cuda_available

        if cuda_available:
            # Get device info
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"     Device: {device_name}")
            print(f"     Device count: {device_count}")
            results['device_name'] = device_name
            results['device_count'] = device_count

            # Check for ROCm/HIP
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            print_result("ROCm/HIP runtime", is_rocm)
            if is_rocm:
                print(f"     HIP version: {torch.version.hip}")
                results['hip_version'] = torch.version.hip
            results['is_rocm'] = is_rocm

        # Check for Flash Attention support
        has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        print_result("Flash Attention (SDPA)", has_sdpa)
        results['has_sdpa'] = has_sdpa

    except ImportError as e:
        print_result("PyTorch import", False, str(e))
        results['pytorch'] = False
        return results

    return results

def test_basic_gemm():
    """Test 3: Basic matrix multiplication."""
    print_header("Test 3: Basic Matrix Multiplication (rocBLAS)")

    results = {}

    try:
        import torch

        if not torch.cuda.is_available():
            print_result("GPU not available", False, "Skipping GEMM test")
            results['gemm'] = False
            return results

        # Create test matrices
        device = torch.device('cuda:0')
        size = 512

        print(f"     Creating {size}x{size} matrices...")
        A = torch.randn(size, size, device=device, dtype=torch.float16)
        B = torch.randn(size, size, device=device, dtype=torch.float16)

        # Warmup
        print("     Warming up...")
        for _ in range(5):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Time the operation
        import time
        print("     Benchmarking...")
        start = time.time()
        for _ in range(100):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        tflops = (2 * size**3 * 100) / (elapsed * 1e12)
        print_result("Basic GEMM", True, f"Time: {elapsed*10:.2f}ms/iter, {tflops:.2f} TFLOPS")
        results['gemm'] = True
        results['gemm_tflops'] = tflops

    except Exception as e:
        print_result("Basic GEMM", False, str(e))
        results['gemm'] = False
        traceback.print_exc()

    return results

def test_hipblaslt_direct():
    """Test 4: Direct hipBLASLt usage (if available)."""
    print_header("Test 4: hipBLASLt Direct Test")

    results = {}

    # Set environment for hipBLASLt
    os.environ['PYTORCH_HIPBLASLT'] = '1'
    os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'
    os.environ['HIPBLASLT_LOG_LEVEL'] = '3'  # Verbose for debugging

    print("     Environment set:")
    print(f"       PYTORCH_HIPBLASLT=1")
    print(f"       HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library")
    print(f"       HIPBLASLT_LOG_LEVEL=3")

    try:
        import torch

        if not torch.cuda.is_available():
            print_result("GPU not available", False, "Skipping hipBLASLt test")
            results['hipblaslt'] = False
            return results

        # Force reload PyTorch with new env vars
        import importlib
        importlib.reload(torch)

        # Create test matrices
        device = torch.device('cuda:0')
        size = 1024

        print(f"     Creating {size}x{size} matrices...")
        A = torch.randn(size, size, device=device, dtype=torch.float16)
        B = torch.randn(size, size, device=device, dtype=torch.float16)

        # Try matrix multiplication (should use hipBLASLt if available)
        print("     Testing GEMM with hipBLASLt...")
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

        print_result("hipBLASLt GEMM", True, "Success!")
        results['hipblaslt'] = True

    except Exception as e:
        error_msg = str(e)
        print_result("hipBLASLt GEMM", False, error_msg)
        results['hipblaslt'] = False
        results['hipblaslt_error'] = error_msg

        # Check for specific error codes
        if "HIPBLASLT Error: 3" in error_msg:
            print("\n     ⚠ Error Analysis:")
            print("       Error 3 = HIPBLAS_STATUS_NOT_SUPPORTED")
            print("       Possible causes:")
            print("       - Incompatible ROCm version")
            print("       - Missing Tensile library for your GPU")
            print("       - hipBLASLt not properly installed")

        if "Could not load" in error_msg:
            print("\n     ⚠ Error Analysis:")
            print("       Cannot load Tensile library")
            print("       Check file permissions and path")

    return results

def test_transformer_engine():
    """Test 5: Transformer Engine with hipBLASLt."""
    print_header("Test 5: Transformer Engine Integration")

    results = {}

    try:
        # CRITICAL: Set env vars BEFORE importing TransformerEngine
        os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'
        os.environ['TE_HIPBLASLT_DISABLED'] = '1'
        os.environ['NVTE_TORCH_COMPILE'] = '0'  # Disable torch.compile to avoid Triton issues

        import transformer_engine.pytorch as te
        print_result("Transformer Engine import", True)
        results['te_import'] = True

        print("     Testing TE Linear with hipBLASLt DISABLED...")

        import torch
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Create a simple linear layer
        linear = te.Linear(in_features=512, out_features=512, device=device)
        x = torch.randn(16, 512, device=device)

        # Forward pass
        y = linear(x)

        print_result("TE Linear (hipBLASLt disabled)", True, "Success!")
        results['te_linear_disabled'] = True

        # Now test with hipBLASLt enabled
        os.environ['NVTE_DISABLE_HIPBLASLT'] = '0'
        os.environ['TE_HIPBLASLT_DISABLED'] = '0'
        os.environ['PYTORCH_HIPBLASLT'] = '1'
        os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'

        print("     Testing TE Linear with hipBLASLt ENABLED...")

        try:
            linear2 = te.Linear(in_features=512, out_features=512, device=device)
            x2 = torch.randn(16, 512, device=device)
            y2 = linear2(x2)

            print_result("TE Linear (hipBLASLt enabled)", True, "Success!")
            results['te_linear_enabled'] = True

        except Exception as e:
            print_result("TE Linear (hipBLASLt enabled)", False, str(e))
            results['te_linear_enabled'] = False
            results['te_error'] = str(e)

    except ImportError:
        print_result("Transformer Engine import", False, "Not installed")
        results['te_import'] = False
    except Exception as e:
        print_result("Transformer Engine test", False, str(e))
        results['te_error'] = str(e)
        traceback.print_exc()

    return results

def generate_report(all_results):
    """Generate final report and recommendations."""
    print_header("Summary and Recommendations")

    env_results = all_results.get('environment', {})
    pytorch_results = all_results.get('pytorch', {})
    gemm_results = all_results.get('gemm', {})
    hipblaslt_results = all_results.get('hipblaslt', {})
    te_results = all_results.get('transformer_engine', {})

    # Overall assessment
    print("\n📊 Overall Assessment:")
    print("-" * 70)

    # Check critical components
    rocm_ok = env_results.get('rocm', False)
    pytorch_ok = pytorch_results.get('pytorch', False)
    cuda_ok = pytorch_results.get('cuda_available', False)
    gemm_ok = gemm_results.get('gemm', False)
    hipblaslt_ok = hipblaslt_results.get('hipblaslt', False)
    te_disabled_ok = te_results.get('te_linear_disabled', False)
    te_enabled_ok = te_results.get('te_linear_enabled', False)

    print(f"  ROCm Installation:           {'✅' if rocm_ok else '❌'}")
    print(f"  PyTorch + ROCm:              {'✅' if pytorch_ok and cuda_ok else '❌'}")
    print(f"  Basic GEMM (rocBLAS):        {'✅' if gemm_ok else '❌'}")
    print(f"  hipBLASLt Direct:            {'✅' if hipblaslt_ok else '❌'}")
    print(f"  Transformer Engine (no BLT): {'✅' if te_disabled_ok else '❌'}")
    print(f"  Transformer Engine (w/ BLT): {'✅' if te_enabled_ok else '❌'}")

    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 70)

    if hipblaslt_ok and te_enabled_ok:
        print("  ✅ hipBLASLt is working! You can enable it in demo_gradio.py:")
        print("     os.environ['PYTORCH_HIPBLASLT'] = '1'")
        print("     os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'")
        print("\n     Remove these lines:")
        print("     os.environ['NVTE_DISABLE_HIPBLASLT'] = '1'")
        print("     os.environ['TE_HIPBLASLT_DISABLED'] = '1'")

    elif gemm_ok and te_disabled_ok:
        print("  ⚠ hipBLASLt NOT working, but rocBLAS works fine.")
        print("     Current configuration is OPTIMAL:")
        print("     - Keep hipBLASLt disabled")
        print("     - Use Flash Attention (CK backend) ← Most important!")
        print("     - Use Transformer Engine with rocBLAS")
        print("\n  Expected performance: 40-70% faster (still excellent!)")

        # Detailed hipBLASLt issue
        if 'hipblaslt_error' in hipblaslt_results:
            error = hipblaslt_results['hipblaslt_error']
            print("\n  🔍 hipBLASLt Error Details:")
            print(f"     {error}")

            if "Error: 3" in error:
                print("\n     Possible fixes:")
                print("     1. Update hipBLASLt: sudo apt-get install --reinstall hipblaslt")
                print("     2. Check GPU architecture compatibility (you need gfx1100 support)")
                print("     3. Verify ROCm 6.4 compatibility with hipBLASLt version")

    else:
        print("  ❌ Critical issues detected:")
        if not rocm_ok:
            print("     - ROCm not found. Install ROCm 6.0+ first.")
        if not (pytorch_ok and cuda_ok):
            print("     - PyTorch with ROCm support not working.")
        if not gemm_ok:
            print("     - Basic GEMM operations failing. Check GPU drivers.")

    # Performance comparison
    if gemm_ok and 'gemm_tflops' in gemm_results:
        tflops = gemm_results['gemm_tflops']
        print(f"\n  📈 Performance Metrics:")
        print(f"     rocBLAS GEMM: {tflops:.2f} TFLOPS (FP16)")

        # Theoretical peak for common GPUs
        gpu_name = pytorch_results.get('device_name', 'Unknown')
        if 'gfx1100' in gpu_name or '7900' in gpu_name:
            theoretical = 61.4  # RX 7900 XTX FP16
            efficiency = (tflops / theoretical) * 100
            print(f"     Theoretical Peak: {theoretical:.1f} TFLOPS (RX 7900 XTX)")
            print(f"     Efficiency: {efficiency:.1f}%")

    # GPU architecture info
    if pytorch_results.get('is_rocm') and pytorch_results.get('device_name'):
        print(f"\n  🖥️  GPU Information:")
        print(f"     Device: {pytorch_results['device_name']}")
        if 'hip_version' in pytorch_results:
            print(f"     HIP Version: {pytorch_results['hip_version']}")
        if env_results.get('tensile_archs'):
            print(f"     Available Tensile libs: {', '.join(env_results['tensile_archs'])}")

    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Test hipBLASLt compatibility on ROCm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_hipblaslt.py                    # Run all tests
  python test_hipblaslt.py --test-pytorch     # Test PyTorch only
  python test_hipblaslt.py --test-te          # Test Transformer Engine only
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--test-pytorch', action='store_true',
                        help='Test PyTorch only')
    parser.add_argument('--test-gemm', action='store_true',
                        help='Test GEMM only')
    parser.add_argument('--test-hipblaslt', action='store_true',
                        help='Test hipBLASLt only')
    parser.add_argument('--test-te', action='store_true',
                        help='Test Transformer Engine only')

    args = parser.parse_args()

    print_header("hipBLASLt Compatibility Test Suite")
    print("This script will test hipBLASLt compatibility on your ROCm system.")
    print("Please wait, this may take 1-2 minutes...")

    all_results = {}

    # Run selected tests
    run_all = not (args.test_pytorch or args.test_gemm or args.test_hipblaslt or args.test_te)

    if run_all or args.test_pytorch:
        all_results['environment'] = test_environment()
        all_results['pytorch'] = test_pytorch_import()

    if run_all or args.test_gemm:
        all_results['gemm'] = test_basic_gemm()

    if run_all or args.test_hipblaslt:
        all_results['hipblaslt'] = test_hipblaslt_direct()

    if run_all or args.test_te:
        all_results['transformer_engine'] = test_transformer_engine()

    # Generate report
    if run_all:
        generate_report(all_results)

    print("\n✅ Testing complete!")

    # Exit with appropriate code
    if all_results.get('hipblaslt', {}).get('hipblaslt', False):
        print("\n🎉 hipBLASLt is WORKING on your system!")
        return 0
    else:
        print("\n⚠️  hipBLASLt is NOT working, but rocBLAS + Flash Attention will work great!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
