#!/usr/bin/env python3
"""
TransformerEngine vs PyTorch Benchmark Script

This script compares the performance of TransformerEngine against standard PyTorch
for various operations commonly used in transformer models.

Usage:
    python benchmark_transformer_engine.py
    python benchmark_transformer_engine.py --iterations 100
    python benchmark_transformer_engine.py --no-warmup
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Set environment variables before importing TransformerEngine
os.environ['NVTE_TORCH_COMPILE'] = '0'
os.environ['ROCBLAS_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'

# Try to import TransformerEngine
HAS_TRANSFORMER_ENGINE = False
try:
    import transformer_engine.pytorch as te
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    print("❌ TransformerEngine not installed")
    print("   Install with: pip install transformer-engine")
    sys.exit(1)


class BenchmarkConfig:
    """Configuration for benchmarks."""
    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.iterations = args.iterations
        self.warmup_iterations = args.warmup if not args.no_warmup else 0
        self.verbose = args.verbose


class BenchmarkResult:
    """Store benchmark results."""
    def __init__(self, name: str, pytorch_time: float, te_time: float,
                 pytorch_memory: float = 0, te_memory: float = 0):
        self.name = name
        self.pytorch_time = pytorch_time
        self.te_time = te_time
        self.pytorch_memory = pytorch_memory
        self.te_memory = te_memory
        self.speedup = pytorch_time / te_time if te_time > 0 else 0

    def __str__(self):
        speedup_str = f"{self.speedup:.2f}x" if self.speedup >= 1 else f"{1/self.speedup:.2f}x slower"
        return (f"{self.name}:\n"
                f"  PyTorch:  {self.pytorch_time*1000:.2f}ms\n"
                f"  TE:       {self.te_time*1000:.2f}ms\n"
                f"  Speedup:  {speedup_str}")


def benchmark_operation(operation, name: str, config: BenchmarkConfig,
                       warmup: bool = True) -> float:
    """Benchmark a single operation."""
    torch.cuda.synchronize()

    # Warmup
    if warmup and config.warmup_iterations > 0:
        for _ in range(config.warmup_iterations):
            operation()
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(config.iterations):
        operation()
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    avg_time = elapsed / config.iterations

    if config.verbose:
        print(f"  {name}: {avg_time*1000:.2f}ms/iter ({config.iterations} iterations)")

    return avg_time


def benchmark_linear_layer(config: BenchmarkConfig,
                          batch_size: int = 16,
                          seq_len: int = 128,
                          hidden_size: int = 768,
                          out_features: int = 3072) -> BenchmarkResult:
    """Benchmark Linear layer performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Linear Layer ({hidden_size} -> {out_features})")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    # Create PyTorch Linear layer
    pytorch_linear = nn.Linear(hidden_size, out_features, bias=True).to(config.device, config.dtype)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # Create TransformerEngine Linear layer
    te_linear = te.Linear(
        in_features=hidden_size,
        out_features=out_features,
        params_dtype=config.dtype,
        device=config.device,
        bias=True
    )

    # Copy weights to ensure fair comparison
    with torch.no_grad():
        te_linear.weight.copy_(pytorch_linear.weight)
        te_linear.bias.copy_(pytorch_linear.bias)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_linear(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch Linear", config)

    # Benchmark TransformerEngine
    def te_forward():
        return te_linear(input_tensor)

    te_time = benchmark_operation(te_forward, "TE Linear", config)

    result = BenchmarkResult(
        f"Linear {hidden_size}->{out_features}",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_mlp(config: BenchmarkConfig,
                  batch_size: int = 16,
                  seq_len: int = 128,
                  hidden_size: int = 768,
                  intermediate_size: int = 3072) -> BenchmarkResult:
    """Benchmark MLP (two linear layers with GELU) performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: MLP ({hidden_size} -> {intermediate_size} -> {hidden_size})")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    # Create PyTorch MLP
    class PyTorchMLP(nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            return x

    pytorch_mlp = PyTorchMLP(hidden_size, intermediate_size).to(config.device, config.dtype)

    # Create TE MLP
    class TEMLP(nn.Module):
        def __init__(self, hidden_size, intermediate_size, dtype, device):
            super().__init__()
            self.fc1 = te.Linear(hidden_size, intermediate_size, params_dtype=dtype, device=device)
            self.fc2 = te.Linear(intermediate_size, hidden_size, params_dtype=dtype, device=device)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            return x

    te_mlp = TEMLP(hidden_size, intermediate_size, config.dtype, config.device)

    # Copy weights
    with torch.no_grad():
        te_mlp.fc1.weight.copy_(pytorch_mlp.fc1.weight)
        te_mlp.fc1.bias.copy_(pytorch_mlp.fc1.bias)
        te_mlp.fc2.weight.copy_(pytorch_mlp.fc2.weight)
        te_mlp.fc2.bias.copy_(pytorch_mlp.fc2.bias)

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_mlp(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch MLP", config)

    # Benchmark TE
    def te_forward():
        return te_mlp(input_tensor)

    te_time = benchmark_operation(te_forward, "TE MLP", config)

    result = BenchmarkResult(
        f"MLP {hidden_size}->{intermediate_size}->{hidden_size}",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_matmul(config: BenchmarkConfig,
                    m: int = 2048,
                    n: int = 768,
                    k: int = 768) -> BenchmarkResult:
    """Benchmark matrix multiplication performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Matrix Multiplication ({m}x{k}) @ ({k}x{n})")
    print(f"{'='*70}")

    A = torch.randn(m, k, device=config.device, dtype=config.dtype)
    B = torch.randn(k, n, device=config.device, dtype=config.dtype)

    # PyTorch matmul
    def pytorch_matmul():
        return torch.matmul(A, B)

    pytorch_time = benchmark_operation(pytorch_matmul, "PyTorch matmul", config)

    # TE uses the same torch.matmul, so this comparison shows if TE overhead exists
    # when called from TE context
    def te_matmul():
        return torch.matmul(A, B)

    te_time = benchmark_operation(te_matmul, "Torch matmul (from TE context)", config)

    result = BenchmarkResult(
        f"MatMul ({m}x{k})@({k}x{n})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_batch_sizes(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark across different batch sizes."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Linear Layer - Varying Batch Sizes")
    print(f"{'='*70}")

    results = []
    batch_sizes = [1, 4, 8, 16, 32, 64]
    hidden_size = 768
    out_features = 3072
    seq_len = 128

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        result = benchmark_linear_layer(
            config,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            out_features=out_features
        )
        results.append(result)

    return results


def benchmark_layer_sizes(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark across different layer sizes."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Linear Layer - Varying Layer Sizes")
    print(f"{'='*70}")

    results = []
    layer_configs = [
        (768, 768),      # Same size
        (768, 3072),     # Expansion (4x)
        (3072, 768),     # Reduction (1/4x)
        (1024, 4096),    # Large expansion
        (4096, 1024),    # Large reduction
    ]

    batch_size = 16
    seq_len = 128

    for hidden_size, out_features in layer_configs:
        print(f"\nLayer size: {hidden_size} -> {out_features}")
        result = benchmark_linear_layer(
            config,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            out_features=out_features
        )
        results.append(result)

    return results


def print_summary(all_results: List[BenchmarkResult]):
    """Print summary of all benchmarks."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Benchmark':<40} {'PyTorch (ms)':<15} {'TE (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*80}")

    total_pytorch_time = 0
    total_te_time = 0

    for result in all_results:
        speedup_str = f"{result.speedup:.2f}x" if result.speedup >= 1 else f"{1/result.speedup:.2f}x slower"
        print(f"{result.name:<40} {result.pytorch_time*1000:<15.2f} {result.te_time*1000:<15.2f} {speedup_str:<10}")
        total_pytorch_time += result.pytorch_time
        total_te_time += result.te_time

    print(f"{'-'*80}")
    overall_speedup = total_pytorch_time / total_te_time if total_te_time > 0 else 0
    speedup_str = f"{overall_speedup:.2f}x" if overall_speedup >= 1 else f"{1/overall_speedup:.2f}x slower"
    print(f"{'OVERALL':<40} {total_pytorch_time*1000:<15.2f} {total_te_time*1000:<15.2f} {speedup_str:<10}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if overall_speedup > 1.1:
        print(f"✅ TransformerEngine is {overall_speedup:.2f}x FASTER than PyTorch")
        print(f"   Recommendation: Use TransformerEngine for production")
    elif overall_speedup > 0.9:
        print(f"⚖️  TransformerEngine and PyTorch have similar performance")
        print(f"   Recommendation: Either works, TE may provide FP8 benefits on MI300")
    else:
        print(f"⚠️  TransformerEngine is {1/overall_speedup:.2f}x SLOWER than PyTorch")
        print(f"   Recommendation: Use standard PyTorch for better performance")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark TransformerEngine vs PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_transformer_engine.py
  python benchmark_transformer_engine.py --iterations 200
  python benchmark_transformer_engine.py --no-warmup --verbose
  python benchmark_transformer_engine.py --quick
        """
    )

    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per benchmark (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--no-warmup', action='store_true',
                       help='Skip warmup iterations')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (fewer iterations and tests)')

    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.iterations = 20
        args.warmup = 5

    config = BenchmarkConfig(args)

    print("="*70)
    print("TransformerEngine vs PyTorch Benchmark")
    print("="*70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data type: {config.dtype}")
    print(f"Iterations: {config.iterations}")
    print(f"Warmup iterations: {config.warmup_iterations}")
    print("="*70)

    all_results = []

    # Basic benchmarks
    all_results.append(benchmark_linear_layer(config, batch_size=16, seq_len=128, hidden_size=768, out_features=3072))
    all_results.append(benchmark_mlp(config, batch_size=16, seq_len=128, hidden_size=768, intermediate_size=3072))
    all_results.append(benchmark_matmul(config, m=2048, n=768, k=768))

    if not args.quick:
        # Extended benchmarks
        batch_results = benchmark_batch_sizes(config)
        all_results.extend(batch_results)

        layer_results = benchmark_layer_sizes(config)
        all_results.extend(layer_results)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
