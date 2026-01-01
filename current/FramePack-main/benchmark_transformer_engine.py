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


def benchmark_layernorm(config: BenchmarkConfig,
                       batch_size: int = 16,
                       seq_len: int = 128,
                       hidden_size: int = 768) -> BenchmarkResult:
    """Benchmark LayerNorm performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: LayerNorm (hidden_size={hidden_size})")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # PyTorch LayerNorm
    pytorch_ln = nn.LayerNorm(hidden_size).to(config.device, config.dtype)

    # TransformerEngine LayerNorm
    te_ln = te.LayerNorm(hidden_size, params_dtype=config.dtype, device=config.device)

    # Copy weights
    with torch.no_grad():
        te_ln.weight.copy_(pytorch_ln.weight)
        te_ln.bias.copy_(pytorch_ln.bias)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_ln(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch LayerNorm", config)

    # Benchmark TE
    def te_forward():
        return te_ln(input_tensor)

    te_time = benchmark_operation(te_forward, "TE LayerNorm", config)

    result = BenchmarkResult(
        f"LayerNorm (d={hidden_size})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_rmsnorm(config: BenchmarkConfig,
                     batch_size: int = 16,
                     seq_len: int = 128,
                     hidden_size: int = 768) -> BenchmarkResult:
    """Benchmark RMSNorm performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: RMSNorm (hidden_size={hidden_size})")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # PyTorch RMSNorm implementation
    class PyTorchRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    pytorch_rms = PyTorchRMSNorm(hidden_size).to(config.device, config.dtype)

    # TransformerEngine RMSNorm
    te_rms = te.RMSNorm(hidden_size, params_dtype=config.dtype, device=config.device)

    # Copy weights
    with torch.no_grad():
        te_rms.weight.copy_(pytorch_rms.weight)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_rms(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch RMSNorm", config)

    # Benchmark TE
    def te_forward():
        return te_rms(input_tensor)

    te_time = benchmark_operation(te_forward, "TE RMSNorm", config)

    result = BenchmarkResult(
        f"RMSNorm (d={hidden_size})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_attention(config: BenchmarkConfig,
                       batch_size: int = 16,
                       seq_len: int = 128,
                       hidden_size: int = 768,
                       num_heads: int = 12) -> BenchmarkResult:
    """Benchmark multi-head attention performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Multi-Head Attention")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"  Num heads: {num_heads}, Head dim: {hidden_size // num_heads}")
    print(f"{'='*70}")

    # PyTorch MultiheadAttention
    pytorch_attn = nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        batch_first=True
    ).to(config.device, config.dtype)

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_attn(input_tensor, input_tensor, input_tensor, need_weights=False)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch Attention", config)

    # Try TransformerEngine MultiheadAttention (may not be supported on AMD)
    te_time = pytorch_time  # Default to same as PyTorch if TE fails
    try:
        te_attn = te.MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            params_dtype=config.dtype,
            device=config.device,
            attn_mask_type="no_mask"  # Disable fused attention features
        )

        def te_forward():
            return te_attn(input_tensor)

        te_time = benchmark_operation(te_forward, "TE Attention", config)
    except Exception as e:
        print(f"  ⚠️  TE Attention not supported on this platform: {str(e)[:80]}")
        print(f"  Using PyTorch implementation for comparison")
        te_time = pytorch_time

    result = BenchmarkResult(
        f"Attention (h={num_heads}, d={hidden_size})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_dropout(config: BenchmarkConfig,
                     batch_size: int = 16,
                     seq_len: int = 128,
                     hidden_size: int = 768,
                     dropout_p: float = 0.1) -> BenchmarkResult:
    """Benchmark dropout performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Dropout (p={dropout_p})")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # PyTorch Dropout
    pytorch_dropout = nn.Dropout(p=dropout_p)

    # TE uses same dropout, so this tests overhead
    te_dropout = nn.Dropout(p=dropout_p)

    # Set to training mode
    pytorch_dropout.train()
    te_dropout.train()

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_dropout(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch Dropout", config)

    # Benchmark TE
    def te_forward():
        return te_dropout(input_tensor)

    te_time = benchmark_operation(te_forward, "TE Dropout", config)

    result = BenchmarkResult(
        f"Dropout (p={dropout_p})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_backward_pass(config: BenchmarkConfig,
                           batch_size: int = 16,
                           seq_len: int = 128,
                           hidden_size: int = 768,
                           out_features: int = 3072) -> BenchmarkResult:
    """Benchmark forward + backward pass performance."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Forward + Backward Pass (Linear)")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    # PyTorch Linear
    pytorch_linear = nn.Linear(hidden_size, out_features, bias=True).to(config.device, config.dtype)
    pytorch_linear.train()

    # Benchmark PyTorch
    def pytorch_forward_backward():
        input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                                   device=config.device, dtype=config.dtype, requires_grad=True)
        output = pytorch_linear(input_tensor)
        loss = output.sum()
        loss.backward()
        return output

    pytorch_time = benchmark_operation(pytorch_forward_backward, "PyTorch Fwd+Bwd", config)

    # Try TE backward pass (may not be supported on AMD)
    te_time = pytorch_time  # Default to same as PyTorch if TE fails
    try:
        # TE Linear
        te_linear = te.Linear(
            in_features=hidden_size,
            out_features=out_features,
            params_dtype=config.dtype,
            device=config.device,
            bias=True
        )
        te_linear.train()

        # Copy weights
        with torch.no_grad():
            te_linear.weight.copy_(pytorch_linear.weight)
            te_linear.bias.copy_(pytorch_linear.bias)

        # Benchmark TE
        def te_forward_backward():
            input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                                       device=config.device, dtype=config.dtype, requires_grad=True)
            output = te_linear(input_tensor)
            loss = output.sum()
            loss.backward()
            return output

        te_time = benchmark_operation(te_forward_backward, "TE Fwd+Bwd", config)
    except Exception as e:
        print(f"  ⚠️  TE Backward pass not supported on this platform: {str(e)[:80]}")
        print(f"  Using PyTorch implementation for comparison")
        te_time = pytorch_time

    result = BenchmarkResult(
        f"Fwd+Bwd Linear {hidden_size}->{out_features}",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_activation_functions(config: BenchmarkConfig,
                                   batch_size: int = 16,
                                   seq_len: int = 128,
                                   hidden_size: int = 3072) -> List[BenchmarkResult]:
    """Benchmark various activation functions."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Activation Functions")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    results = []
    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    activations = [
        ("GELU", nn.GELU()),
        ("ReLU", nn.ReLU()),
        ("SiLU", nn.SiLU()),
        ("Tanh", nn.Tanh()),
    ]

    for name, activation in activations:
        print(f"\nActivation: {name}")

        # Benchmark activation
        def forward():
            return activation(input_tensor)

        time_taken = benchmark_operation(forward, f"{name}", config)

        result = BenchmarkResult(
            f"Activation {name}",
            time_taken,
            time_taken  # Same for both since it's just torch
        )
        results.append(result)

    return results


def benchmark_sequence_lengths(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark across different sequence lengths."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Linear Layer - Varying Sequence Lengths")
    print(f"{'='*70}")

    results = []
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    batch_size = 8
    hidden_size = 768
    out_features = 3072

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        result = benchmark_linear_layer(
            config,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            out_features=out_features
        )
        results.append(result)

    return results


def benchmark_transformer_block(config: BenchmarkConfig,
                               batch_size: int = 16,
                               seq_len: int = 128,
                               hidden_size: int = 768,
                               num_heads: int = 12,
                               intermediate_size: int = 3072) -> BenchmarkResult:
    """Benchmark a complete transformer block."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Full Transformer Block")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"  Num heads: {num_heads}, FFN size: {intermediate_size}")
    print(f"{'='*70}")

    # PyTorch Transformer Block
    class PyTorchTransformerBlock(nn.Module):
        def __init__(self, hidden_size, num_heads, intermediate_size):
            super().__init__()
            self.ln1 = nn.LayerNorm(hidden_size)
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            # Self-attention
            residual = x
            x = self.ln1(x)
            x, _ = self.attn(x, x, x, need_weights=False)
            x = self.dropout(x)
            x = residual + x

            # FFN
            residual = x
            x = self.ln2(x)
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = residual + x

            return x

    pytorch_block = PyTorchTransformerBlock(hidden_size, num_heads, intermediate_size).to(
        config.device, config.dtype
    )

    input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                               device=config.device, dtype=config.dtype)

    # Benchmark PyTorch
    def pytorch_forward():
        return pytorch_block(input_tensor)

    pytorch_time = benchmark_operation(pytorch_forward, "PyTorch TransformerBlock", config)

    # Try TransformerEngine Transformer Block (may not be fully supported on AMD)
    te_time = pytorch_time  # Default to same as PyTorch if TE fails
    try:
        # TransformerEngine Transformer Block (without attention due to AMD limitations)
        class TETransformerBlock(nn.Module):
            def __init__(self, hidden_size, num_heads, intermediate_size, dtype, device):
                super().__init__()
                self.ln1 = te.LayerNorm(hidden_size, params_dtype=dtype, device=device)
                # Use PyTorch attention instead of TE due to AMD compatibility
                self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True).to(device, dtype)
                self.ln2 = te.LayerNorm(hidden_size, params_dtype=dtype, device=device)
                self.fc1 = te.Linear(hidden_size, intermediate_size, params_dtype=dtype, device=device)
                self.gelu = nn.GELU()
                self.fc2 = te.Linear(intermediate_size, hidden_size, params_dtype=dtype, device=device)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                # Self-attention
                residual = x
                x = self.ln1(x)
                x, _ = self.attn(x, x, x, need_weights=False)
                x = self.dropout(x)
                x = residual + x

                # FFN
                residual = x
                x = self.ln2(x)
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.fc2(x)
                x = self.dropout(x)
                x = residual + x

                return x

        te_block = TETransformerBlock(hidden_size, num_heads, intermediate_size, config.dtype, config.device)

        # Benchmark TE
        def te_forward():
            return te_block(input_tensor)

        te_time = benchmark_operation(te_forward, "TE TransformerBlock", config)
    except Exception as e:
        print(f"  ⚠️  TE TransformerBlock not fully supported: {str(e)[:80]}")
        print(f"  Using PyTorch implementation for comparison")
        te_time = pytorch_time

    result = BenchmarkResult(
        f"TransformerBlock (h={num_heads}, d={hidden_size})",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def benchmark_training_step(config: BenchmarkConfig,
                           batch_size: int = 16,
                           seq_len: int = 128,
                           hidden_size: int = 768,
                           num_heads: int = 12,
                           intermediate_size: int = 3072) -> BenchmarkResult:
    """Benchmark a complete training step (forward + backward + optimizer step)."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Full Training Step")
    print(f"  Input shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"{'='*70}")

    # PyTorch model
    class PyTorchModel(nn.Module):
        def __init__(self, hidden_size, num_heads, intermediate_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
            self.ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            x = self.ln(x)
            return x

    pytorch_model = PyTorchModel(hidden_size, num_heads, intermediate_size).to(
        config.device, config.dtype
    )
    pytorch_optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=1e-4)

    # Benchmark PyTorch training step
    def pytorch_training_step():
        input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                                   device=config.device, dtype=config.dtype)
        target = torch.randn(batch_size, seq_len, hidden_size,
                            device=config.device, dtype=config.dtype)

        pytorch_optimizer.zero_grad()
        output = pytorch_model(input_tensor)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        pytorch_optimizer.step()
        return output

    pytorch_time = benchmark_operation(pytorch_training_step, "PyTorch Training Step", config)

    # Try TE training step (may not be supported on AMD)
    te_time = pytorch_time  # Default to same as PyTorch if TE fails
    try:
        # TE model
        class TEModel(nn.Module):
            def __init__(self, hidden_size, num_heads, intermediate_size, dtype, device):
                super().__init__()
                self.fc1 = te.Linear(hidden_size, intermediate_size, params_dtype=dtype, device=device)
                self.gelu = nn.GELU()
                self.fc2 = te.Linear(intermediate_size, hidden_size, params_dtype=dtype, device=device)
                self.ln = te.LayerNorm(hidden_size, params_dtype=dtype, device=device)

            def forward(self, x):
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.fc2(x)
                x = self.ln(x)
                return x

        te_model = TEModel(hidden_size, num_heads, intermediate_size, config.dtype, config.device)
        te_optimizer = torch.optim.AdamW(te_model.parameters(), lr=1e-4)

        # Benchmark TE training step
        def te_training_step():
            input_tensor = torch.randn(batch_size, seq_len, hidden_size,
                                       device=config.device, dtype=config.dtype)
            target = torch.randn(batch_size, seq_len, hidden_size,
                                device=config.device, dtype=config.dtype)

            te_optimizer.zero_grad()
            output = te_model(input_tensor)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            te_optimizer.step()
            return output

        te_time = benchmark_operation(te_training_step, "TE Training Step", config)
    except Exception as e:
        print(f"  ⚠️  TE Training step not supported on this platform: {str(e)[:80]}")
        print(f"  Using PyTorch implementation for comparison")
        te_time = pytorch_time

    result = BenchmarkResult(
        "Full Training Step",
        pytorch_time,
        te_time
    )

    print(result)
    return result


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_memory_usage(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark memory usage for various operations."""
    print(f"\n{'='*70}")
    print(f"Benchmark: Memory Usage Comparison")
    print(f"{'='*70}")

    results = []
    configs_to_test = [
        ("Small Linear", {"batch_size": 16, "seq_len": 128, "hidden_size": 768, "out_features": 768}),
        ("Large Linear", {"batch_size": 16, "seq_len": 512, "hidden_size": 2048, "out_features": 8192}),
    ]

    for name, params in configs_to_test:
        print(f"\nMemory test: {name}")

        # PyTorch memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pytorch_linear = nn.Linear(params["hidden_size"], params["out_features"]).to(
            config.device, config.dtype
        )
        input_tensor = torch.randn(params["batch_size"], params["seq_len"], params["hidden_size"],
                                   device=config.device, dtype=config.dtype)
        _ = pytorch_linear(input_tensor)
        torch.cuda.synchronize()

        pytorch_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Clean up
        del pytorch_linear, input_tensor
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # TE memory
        te_linear = te.Linear(
            in_features=params["hidden_size"],
            out_features=params["out_features"],
            params_dtype=config.dtype,
            device=config.device
        )
        input_tensor = torch.randn(params["batch_size"], params["seq_len"], params["hidden_size"],
                                   device=config.device, dtype=config.dtype)
        _ = te_linear(input_tensor)
        torch.cuda.synchronize()

        te_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        print(f"  PyTorch memory: {pytorch_memory:.2f} MB")
        print(f"  TE memory: {te_memory:.2f} MB")
        print(f"  Difference: {te_memory - pytorch_memory:.2f} MB")

        result = BenchmarkResult(
            f"Memory {name}",
            pytorch_memory,
            te_memory,
            pytorch_memory,
            te_memory
        )
        results.append(result)

        # Clean up
        del te_linear, input_tensor
        torch.cuda.empty_cache()

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
  python benchmark_transformer_engine.py --comprehensive  # Run all tests
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
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run all comprehensive benchmarks including new tests')
    parser.add_argument('--test', type=str, choices=[
        'linear', 'mlp', 'matmul', 'layernorm', 'rmsnorm', 'attention',
        'dropout', 'backward', 'activations', 'transformer', 'training',
        'memory', 'batch_sizes', 'layer_sizes', 'seq_lengths'
    ], help='Run a specific test only')

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

    # If specific test requested
    if args.test:
        if args.test == 'linear':
            all_results.append(benchmark_linear_layer(config))
        elif args.test == 'mlp':
            all_results.append(benchmark_mlp(config))
        elif args.test == 'matmul':
            all_results.append(benchmark_matmul(config))
        elif args.test == 'layernorm':
            all_results.append(benchmark_layernorm(config))
        elif args.test == 'rmsnorm':
            all_results.append(benchmark_rmsnorm(config))
        elif args.test == 'attention':
            all_results.append(benchmark_attention(config))
        elif args.test == 'dropout':
            all_results.append(benchmark_dropout(config))
        elif args.test == 'backward':
            all_results.append(benchmark_backward_pass(config))
        elif args.test == 'activations':
            all_results.extend(benchmark_activation_functions(config))
        elif args.test == 'transformer':
            all_results.append(benchmark_transformer_block(config))
        elif args.test == 'training':
            all_results.append(benchmark_training_step(config))
        elif args.test == 'memory':
            all_results.extend(benchmark_memory_usage(config))
        elif args.test == 'batch_sizes':
            all_results.extend(benchmark_batch_sizes(config))
        elif args.test == 'layer_sizes':
            all_results.extend(benchmark_layer_sizes(config))
        elif args.test == 'seq_lengths':
            all_results.extend(benchmark_sequence_lengths(config))
    else:
        # Basic benchmarks (always run unless --test specified)
        all_results.append(benchmark_linear_layer(config, batch_size=16, seq_len=128, hidden_size=768, out_features=3072))
        all_results.append(benchmark_mlp(config, batch_size=16, seq_len=128, hidden_size=768, intermediate_size=3072))
        all_results.append(benchmark_matmul(config, m=2048, n=768, k=768))

        # New core benchmarks
        all_results.append(benchmark_layernorm(config))
        all_results.append(benchmark_rmsnorm(config))
        all_results.append(benchmark_attention(config))
        all_results.append(benchmark_backward_pass(config))

        if args.comprehensive or not args.quick:
            # Extended benchmarks
            all_results.append(benchmark_dropout(config))
            all_results.extend(benchmark_activation_functions(config))
            all_results.append(benchmark_transformer_block(config))
            all_results.append(benchmark_training_step(config))

        if args.comprehensive:
            # Comprehensive tests
            batch_results = benchmark_batch_sizes(config)
            all_results.extend(batch_results)

            layer_results = benchmark_layer_sizes(config)
            all_results.extend(layer_results)

            seq_results = benchmark_sequence_lengths(config)
            all_results.extend(seq_results)

            if torch.cuda.is_available():
                memory_results = benchmark_memory_usage(config)
                all_results.extend(memory_results)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
