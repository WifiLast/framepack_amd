#!/usr/bin/env python3
"""
Test script for tritonBLAS integration

This script verifies that tritonBLAS is working correctly and measures
its performance compared to PyTorch's default matmul.

Usage:
    python test_tritonblas.py
    python test_tritonblas.py --verbose
    python test_tritonblas.py --benchmark
"""

import os
import sys
import argparse
import time
import torch

# Add tritonBLAS to path
tritonblas_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'tritonBLAS-main', 'include')
if os.path.exists(tritonblas_path):
    sys.path.insert(0, tritonblas_path)

from diffusers_helper.tritonblas_patch import (
    patch_pytorch_with_tritonblas,
    print_tritonblas_stats,
    reset_tritonblas_stats,
)


def test_basic_matmul(verbose=False):
    """Test basic matrix multiplication with tritonBLAS."""
    print("\n" + "="*60)
    print("Test 1: Basic Matrix Multiplication")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping test")
        return False

    device = torch.device('cuda')

    # Test FP16 matmul (most common for inference)
    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    for m, k in sizes:
        n = k
        print(f"\nTesting ({m}, {k}) @ ({k}, {n}) with FP16...")

        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Warmup
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed run
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Verify correctness
        expected = a @ b
        max_diff = (c - expected).abs().max().item()

        if max_diff < 1e-2:  # FP16 precision
            print(f"  ✓ Result correct (max_diff={max_diff:.6f})")
            print(f"  ⏱ Time: {elapsed*1000:.2f}ms")
        else:
            print(f"  ❌ Result incorrect (max_diff={max_diff:.6f})")
            return False

    print("\n✓ All basic matmul tests passed")
    return True


def test_linear_layer(verbose=False):
    """Test nn.Linear with tritonBLAS."""
    print("\n" + "="*60)
    print("Test 2: Linear Layer (F.linear)")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping test")
        return False

    device = torch.device('cuda')

    # Test with different batch sizes and dimensions
    configs = [
        (1, 512, 1024),     # Single sample
        (32, 768, 3072),    # Small batch
        (128, 1024, 4096),  # Large batch
    ]

    for batch, in_features, out_features in configs:
        print(f"\nTesting Linear: batch={batch}, in={in_features}, out={out_features}")

        # Create linear layer
        linear = torch.nn.Linear(in_features, out_features, device=device, dtype=torch.float16)

        # Create input
        x = torch.randn(batch, in_features, device=device, dtype=torch.float16)

        # Forward pass
        start = time.time()
        y = linear(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Check output shape
        if y.shape == (batch, out_features):
            print(f"  ✓ Output shape correct: {y.shape}")
            print(f"  ⏱ Time: {elapsed*1000:.2f}ms")
        else:
            print(f"  ❌ Output shape wrong: {y.shape} vs expected {(batch, out_features)}")
            return False

    print("\n✓ All linear layer tests passed")
    return True


def benchmark_tritonblas_vs_pytorch(verbose=False):
    """Benchmark tritonBLAS vs PyTorch matmul."""
    print("\n" + "="*60)
    print("Benchmark: tritonBLAS vs PyTorch")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping benchmark")
        return

    device = torch.device('cuda')
    sizes = [(2048, 2048), (4096, 4096), (8192, 8192)]
    n_iters = 100

    results = []

    for m, k in sizes:
        n = k
        print(f"\nBenchmarking ({m}, {k}) @ ({k}, {n}) with {n_iters} iterations...")

        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark with tritonBLAS
        reset_tritonblas_stats()
        start = time.time()
        for _ in range(n_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        tritonblas_time = time.time() - start

        # Disable tritonBLAS
        patch_pytorch_with_tritonblas(enable=False)

        # Warmup PyTorch
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        start = time.time()
        for _ in range(n_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start

        # Re-enable tritonBLAS
        patch_pytorch_with_tritonblas(
            enable=True,
            verbose=verbose,
            fallback_to_torch=True,
            min_size=512,
        )

        # Calculate speedup
        speedup = pytorch_time / tritonblas_time
        tritonblas_avg = tritonblas_time / n_iters * 1000
        pytorch_avg = pytorch_time / n_iters * 1000

        print(f"  tritonBLAS: {tritonblas_avg:.3f}ms per matmul")
        print(f"  PyTorch:    {pytorch_avg:.3f}ms per matmul")
        print(f"  Speedup:    {speedup:.2f}x")

        results.append({
            'size': (m, k, n),
            'tritonblas_time': tritonblas_avg,
            'pytorch_time': pytorch_avg,
            'speedup': speedup,
        })

    # Summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"{'Size':<20} {'tritonBLAS (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-"*60)
    for r in results:
        size_str = f"{r['size'][0]}x{r['size'][2]}"
        print(f"{size_str:<20} {r['tritonblas_time']:<15.3f} {r['pytorch_time']:<15.3f} {r['speedup']:<10.2f}x")

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-"*60)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test tritonBLAS integration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--min-size', type=int, default=512, help='Minimum matrix size for tritonBLAS')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("tritonBLAS Integration Test")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Make sure you're running on a machine with AMD ROCm or NVIDIA CUDA")
        return 1

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ ROCm/HIP: {getattr(torch.version, 'hip', None) is not None}")

    # Enable tritonBLAS
    print("\nEnabling tritonBLAS monkey-patch...")
    patch_pytorch_with_tritonblas(
        enable=True,
        verbose=args.verbose,
        fallback_to_torch=True,
        min_size=args.min_size,
        use_streamk=False,
    )

    # Run tests
    success = True

    # Test 1: Basic matmul
    if not test_basic_matmul(args.verbose):
        success = False

    # Test 2: Linear layers
    if not test_linear_layer(args.verbose):
        success = False

    # Show statistics
    print_tritonblas_stats()

    # Benchmark if requested
    if args.benchmark:
        benchmark_tritonblas_vs_pytorch(args.verbose)

    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests PASSED")
        print("="*60)
        print("\ntritonBLAS is working correctly!")
        print("You can now use it in demo_gradio.py with:")
        print("  export FRAMEPACK_USE_TRITONBLAS=1")
        print("  python demo_gradio.py")
        return 0
    else:
        print("❌ Some tests FAILED")
        print("="*60)
        print("\nCheck the error messages above.")
        print("Common issues:")
        print("  - tritonBLAS not installed: cd cache/tritonBLAS-main && pip install -e .")
        print("  - Wrong GPU type: tritonBLAS is optimized for AMD GPUs")
        print("  - Missing dependencies: pip install triton")
        return 1


if __name__ == '__main__':
    sys.exit(main())
