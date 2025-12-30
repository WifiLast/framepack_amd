"""
tritonBLAS Monkey-Patch for PyTorch Linear Operations
=====================================================

This module provides a monkey-patch to replace PyTorch's default matmul operations
with tritonBLAS optimized GEMM kernels for AMD ROCm GPUs (RX 7900, MI200, MI300).

tritonBLAS uses analytical models to select optimal kernel configurations without
autotuning, providing:
- Faster kernel selection (no autotuning overhead)
- Optimized GEMM for AMD architectures
- Support for FP16, BF16, FP32 (FP8 disabled for RX 7900)
- Stream-K algorithm for better load balancing

Usage:
    from diffusers_helper.tritonblas_patch import patch_pytorch_with_tritonblas

    # Enable tritonBLAS globally
    patch_pytorch_with_tritonblas(
        enable=True,
        verbose=True,
        fallback_to_torch=True,
        min_size=512  # Only use tritonBLAS for M*N*K >= 512^3
    )

Environment Variables:
    FRAMEPACK_USE_TRITONBLAS=1          # Enable tritonBLAS (default: 0)
    FRAMEPACK_TRITONBLAS_VERBOSE=1      # Verbose logging (default: 0)
    FRAMEPACK_TRITONBLAS_MIN_SIZE=512   # Minimum size threshold (default: 512)
    FRAMEPACK_TRITONBLAS_STREAMK=0      # Enable Stream-K algorithm (default: 0)
    FRAMEPACK_TRITONBLAS_FALLBACK=1     # Fallback to PyTorch on error (default: 1)
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import functools

# Global configuration
_TRITONBLAS_ENABLED = False
_TRITONBLAS_VERBOSE = False
_TRITONBLAS_MIN_SIZE = 512
_TRITONBLAS_USE_STREAMK = False
_TRITONBLAS_FALLBACK = True
_TRITONBLAS_AVAILABLE = False

# Statistics
_tritonblas_stats = {
    'total_calls': 0,
    'tritonblas_calls': 0,
    'torch_fallback_calls': 0,
    'size_filtered_calls': 0,
    'error_calls': 0,
}

# Original PyTorch functions (stored for fallback)
_original_torch_matmul = None
_original_F_linear = None
_original_addmm = None

# tritonBLAS imports (lazy loaded)
_tritonblas = None


def _lazy_import_tritonblas():
    """Lazy import tritonBLAS to avoid startup overhead."""
    global _tritonblas, _TRITONBLAS_AVAILABLE

    if _tritonblas is not None:
        return _tritonblas

    try:
        # Add tritonBLAS to path if needed
        tritonblas_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'cache', 'tritonBLAS-main', 'include'
        )
        if os.path.exists(tritonblas_path) and tritonblas_path not in sys.path:
            sys.path.insert(0, tritonblas_path)

        import tritonblas
        _tritonblas = tritonblas
        _TRITONBLAS_AVAILABLE = True

        if _TRITONBLAS_VERBOSE:
            print(f"✓ tritonBLAS loaded successfully from {tritonblas.__file__}")

        return _tritonblas

    except ImportError as e:
        if _TRITONBLAS_VERBOSE:
            print(f"✗ Failed to import tritonBLAS: {e}")
            print(f"  Searched in: {tritonblas_path}")
        _TRITONBLAS_AVAILABLE = False
        return None


def _should_use_tritonblas(m: int, n: int, k: int) -> bool:
    """
    Heuristic to determine if tritonBLAS should be used for this matmul.

    tritonBLAS is most beneficial for:
    - Large matrix sizes (compute-bound)
    - Batch sizes that align well with GPU compute units

    For very small matrices, PyTorch overhead is negligible and
    the selector overhead might dominate.

    Args:
        m, n, k: Matrix dimensions

    Returns:
        True if tritonBLAS should be used
    """
    if not _TRITONBLAS_ENABLED or not _TRITONBLAS_AVAILABLE:
        return False

    # Size threshold - tritonBLAS helps most with larger matmuls
    # RX 7900 has 96 CUs, so we want enough work to saturate the GPU
    min_dim = min(m, n, k)

    # Use tritonBLAS if any dimension is large enough
    if min_dim >= _TRITONBLAS_MIN_SIZE:
        return True

    # Also use for medium-large total work even if one dimension is small
    # (e.g., batch matmul with large batch size)
    total_work = m * n * k
    threshold_work = _TRITONBLAS_MIN_SIZE ** 3

    return total_work >= threshold_work


def tritonblas_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    tritonBLAS-accelerated matrix multiplication.

    Replaces torch.matmul with tritonBLAS.matmul for better performance on AMD GPUs.

    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N) or (N, K)^T

    Returns:
        Result matrix (M, N)
    """
    global _tritonblas_stats

    _tritonblas_stats['total_calls'] += 1

    # Handle batched matmul and broadcasting
    if a.dim() > 2 or b.dim() > 2:
        # Fall back to PyTorch for batched/complex cases
        _tritonblas_stats['torch_fallback_calls'] += 1
        return _original_torch_matmul(a, b)

    # Ensure 2D tensors
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    m, k1 = a.shape
    if b.shape[0] == k1:
        # b is (K, N) - already correct layout
        k2, n = b.shape
    elif b.shape[1] == k1:
        # b is (N, K) - need to transpose
        n, k2 = b.shape
        b = b.T
    else:
        # Incompatible dimensions, fall back
        _tritonblas_stats['torch_fallback_calls'] += 1
        return _original_torch_matmul(a, b)

    assert k1 == k2, f"Dimension mismatch: {k1} != {k2}"
    k = k1

    # Check if we should use tritonBLAS
    if not _should_use_tritonblas(m, n, k):
        _tritonblas_stats['size_filtered_calls'] += 1
        return _original_torch_matmul(a, b)

    try:
        # Import tritonBLAS
        tritonblas = _lazy_import_tritonblas()
        if tritonblas is None:
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_torch_matmul(a, b)

        # Allocate output tensor
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        # Call tritonBLAS matmul
        tritonblas.matmul(a, b, c, enable_streamk=_TRITONBLAS_USE_STREAMK)

        _tritonblas_stats['tritonblas_calls'] += 1

        if _TRITONBLAS_VERBOSE and _tritonblas_stats['tritonblas_calls'] <= 5:
            print(f"  tritonBLAS matmul: ({m}, {k}) @ ({k}, {n}) -> ({m}, {n})")

        return c

    except Exception as e:
        _tritonblas_stats['error_calls'] += 1

        if _TRITONBLAS_VERBOSE:
            print(f"  ⚠ tritonBLAS error for ({m}, {k}) @ ({k}, {n}): {e}")

        if _TRITONBLAS_FALLBACK:
            if _TRITONBLAS_VERBOSE:
                print(f"    → Falling back to PyTorch")
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_torch_matmul(a, b)
        else:
            raise


def tritonblas_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    tritonBLAS-accelerated linear layer.

    Replaces F.linear with tritonBLAS.matmul + bias addition.

    Linear operation: y = xW^T + b
    Where x is (*, in_features), W is (out_features, in_features)

    Args:
        input: Input tensor (*, in_features)
        weight: Weight tensor (out_features, in_features)
        bias: Optional bias tensor (out_features)

    Returns:
        Output tensor (*, out_features)
    """
    global _tritonblas_stats

    _tritonblas_stats['total_calls'] += 1

    # Handle multi-dimensional input
    input_shape = input.shape
    if input.dim() > 2:
        # Flatten all batch dimensions
        input_2d = input.reshape(-1, input.shape[-1])
    else:
        input_2d = input

    # Ensure weight is 2D (out_features, in_features)
    if weight.dim() != 2:
        _tritonblas_stats['torch_fallback_calls'] += 1
        return _original_F_linear(input, weight, bias)

    out_features, in_features = weight.shape
    batch_size = input_2d.shape[0]

    # Check if we should use tritonBLAS
    # Linear: (batch, in_features) @ (in_features, out_features)
    # We need to transpose weight: (out_features, in_features)^T = (in_features, out_features)
    if not _should_use_tritonblas(batch_size, out_features, in_features):
        _tritonblas_stats['size_filtered_calls'] += 1
        return _original_F_linear(input, weight, bias)

    try:
        # Import tritonBLAS
        tritonblas = _lazy_import_tritonblas()
        if tritonblas is None:
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_F_linear(input, weight, bias)

        # Transpose weight to (in_features, out_features)
        weight_t = weight.T

        # Allocate output tensor
        output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)

        # Call tritonBLAS: input @ weight^T
        tritonblas.matmul(input_2d, weight_t, output, enable_streamk=_TRITONBLAS_USE_STREAMK)

        # Add bias if present
        if bias is not None:
            output = output + bias

        # Reshape to original batch dimensions
        if len(input_shape) > 2:
            output = output.reshape(*input_shape[:-1], out_features)

        _tritonblas_stats['tritonblas_calls'] += 1

        if _TRITONBLAS_VERBOSE and _tritonblas_stats['tritonblas_calls'] <= 5:
            print(f"  tritonBLAS linear: ({batch_size}, {in_features}) @ ({in_features}, {out_features})")

        return output

    except Exception as e:
        _tritonblas_stats['error_calls'] += 1

        if _TRITONBLAS_VERBOSE:
            print(f"  ⚠ tritonBLAS error for linear ({batch_size}, {in_features}) -> ({batch_size}, {out_features}): {e}")

        if _TRITONBLAS_FALLBACK:
            if _TRITONBLAS_VERBOSE:
                print(f"    → Falling back to PyTorch")
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_F_linear(input, weight, bias)
        else:
            raise


def tritonblas_addmm(
    bias: torch.Tensor,
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    tritonBLAS-accelerated addmm operation.

    Replaces torch.addmm with tritonBLAS.matmul.

    Operation: out = beta * bias + alpha * (input @ mat2)

    Args:
        bias: Bias tensor to add (can be scalar or matrix)
        input: First matrix (M, K)
        mat2: Second matrix (K, N)
        beta: Scale factor for bias
        alpha: Scale factor for matmul result

    Returns:
        Result tensor (M, N)
    """
    global _tritonblas_stats

    _tritonblas_stats['total_calls'] += 1

    # Get dimensions
    if input.dim() != 2 or mat2.dim() != 2:
        _tritonblas_stats['torch_fallback_calls'] += 1
        return _original_addmm(bias, input, mat2, beta=beta, alpha=alpha)

    m, k1 = input.shape
    k2, n = mat2.shape

    if k1 != k2:
        _tritonblas_stats['torch_fallback_calls'] += 1
        return _original_addmm(bias, input, mat2, beta=beta, alpha=alpha)

    k = k1

    # Check if we should use tritonBLAS
    if not _should_use_tritonblas(m, n, k):
        _tritonblas_stats['size_filtered_calls'] += 1
        return _original_addmm(bias, input, mat2, beta=beta, alpha=alpha)

    try:
        # Import tritonBLAS
        tritonblas = _lazy_import_tritonblas()
        if tritonblas is None:
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_addmm(bias, input, mat2, beta=beta, alpha=alpha)

        # Allocate output tensor
        output = torch.empty((m, n), device=input.device, dtype=input.dtype)

        # Call tritonBLAS matmul
        tritonblas.matmul(input, mat2, output, enable_streamk=_TRITONBLAS_USE_STREAMK)

        # Apply alpha scaling
        if alpha != 1.0:
            output = output * alpha

        # Add beta * bias
        if beta != 0.0:
            if beta == 1.0:
                output = output + bias
            else:
                output = output + beta * bias

        _tritonblas_stats['tritonblas_calls'] += 1

        return output

    except Exception as e:
        _tritonblas_stats['error_calls'] += 1

        if _TRITONBLAS_VERBOSE:
            print(f"  ⚠ tritonBLAS error for addmm: {e}")

        if _TRITONBLAS_FALLBACK:
            _tritonblas_stats['torch_fallback_calls'] += 1
            return _original_addmm(bias, input, mat2, beta=beta, alpha=alpha)
        else:
            raise


def patch_pytorch_with_tritonblas(
    enable: bool = True,
    verbose: bool = False,
    fallback_to_torch: bool = True,
    min_size: int = 512,
    use_streamk: bool = False,
):
    """
    Monkey-patch PyTorch to use tritonBLAS for matmul operations.

    This patches:
    - torch.matmul
    - torch.nn.functional.linear
    - torch.addmm

    Args:
        enable: Enable tritonBLAS patching
        verbose: Print debug information
        fallback_to_torch: Fall back to PyTorch on errors
        min_size: Minimum matrix dimension to use tritonBLAS (smaller matrices use PyTorch)
        use_streamk: Enable Stream-K algorithm for better load balancing
    """
    global _TRITONBLAS_ENABLED, _TRITONBLAS_VERBOSE, _TRITONBLAS_FALLBACK
    global _TRITONBLAS_MIN_SIZE, _TRITONBLAS_USE_STREAMK
    global _original_torch_matmul, _original_F_linear, _original_addmm

    _TRITONBLAS_ENABLED = enable
    _TRITONBLAS_VERBOSE = verbose
    _TRITONBLAS_FALLBACK = fallback_to_torch
    _TRITONBLAS_MIN_SIZE = min_size
    _TRITONBLAS_USE_STREAMK = use_streamk

    if not enable:
        # Restore original functions if previously patched
        if _original_torch_matmul is not None:
            torch.matmul = _original_torch_matmul
        if _original_F_linear is not None:
            F.linear = _original_F_linear
        if _original_addmm is not None:
            torch.addmm = _original_addmm

        if verbose:
            print("tritonBLAS monkey-patch disabled")
        return

    # Try to import tritonBLAS
    tritonblas = _lazy_import_tritonblas()
    if tritonblas is None:
        print("✗ Cannot enable tritonBLAS: import failed")
        print("  Make sure tritonBLAS is installed and accessible")
        return

    # Store original functions if not already stored
    if _original_torch_matmul is None:
        _original_torch_matmul = torch.matmul
    if _original_F_linear is None:
        _original_F_linear = F.linear
    if _original_addmm is None:
        _original_addmm = torch.addmm

    # Apply patches
    torch.matmul = tritonblas_matmul
    F.linear = tritonblas_linear
    torch.addmm = tritonblas_addmm

    if verbose:
        print("\n" + "="*60)
        print("tritonBLAS Monkey-Patch Enabled")
        print("="*60)
        print(f"  Minimum size threshold: {min_size}")
        print(f"  Stream-K algorithm: {'Enabled' if use_streamk else 'Disabled'}")
        print(f"  Fallback to PyTorch: {'Enabled' if fallback_to_torch else 'Disabled'}")
        print(f"  Patched functions:")
        print(f"    - torch.matmul -> tritonblas_matmul")
        print(f"    - torch.nn.functional.linear -> tritonblas_linear")
        print(f"    - torch.addmm -> tritonblas_addmm")
        print("="*60 + "\n")


def print_tritonblas_stats():
    """Print statistics about tritonBLAS usage."""
    print("\n" + "="*60)
    print("tritonBLAS Usage Statistics")
    print("="*60)
    print(f"  Total matmul/linear calls: {_tritonblas_stats['total_calls']}")
    print(f"  tritonBLAS calls: {_tritonblas_stats['tritonblas_calls']}")
    print(f"  PyTorch fallback calls: {_tritonblas_stats['torch_fallback_calls']}")
    print(f"  Size-filtered calls: {_tritonblas_stats['size_filtered_calls']}")
    print(f"  Error calls: {_tritonblas_stats['error_calls']}")

    if _tritonblas_stats['total_calls'] > 0:
        pct = 100.0 * _tritonblas_stats['tritonblas_calls'] / _tritonblas_stats['total_calls']
        print(f"  tritonBLAS usage: {pct:.1f}%")

    print("="*60 + "\n")


def reset_tritonblas_stats():
    """Reset tritonBLAS usage statistics."""
    global _tritonblas_stats
    _tritonblas_stats = {
        'total_calls': 0,
        'tritonblas_calls': 0,
        'torch_fallback_calls': 0,
        'size_filtered_calls': 0,
        'error_calls': 0,
    }
