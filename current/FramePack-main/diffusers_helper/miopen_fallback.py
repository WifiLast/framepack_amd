"""
MIOpen Fallback Handler for AMD GPUs
Provides automatic fallback mechanisms when MIOpen convolution fails.

Fallback Strategy (in order):
1. Primary: Optimized 3D XDLOPS implicit GEMM (configured via environment variables)
   - MIOpen tries these first during algorithm Find phase
2. Tier 1 Fallback: Naive convolution algorithm on GPU (MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD)
   - Added to MIOpen's algorithm search space at startup
   - MIOpen automatically selects this if optimized algorithms fail during Find
   - Still runs on GPU, slightly slower but very compatible
   - No performance penalty when optimized algorithms work
3. Tier 2 Fallback: CPU computation (last resort only)
   - Only triggered if MIOpen Find completely fails (both optimized and naive fail)
   - Slowest, but guaranteed to work

This ensures maximum performance while maintaining reliability.
"""

import torch
from contextlib import contextmanager
from typing import Callable, Any


class MIOpenFallbackHandler:
    """Handles MIOpen errors with CPU fallback (last resort only)."""

    _fallback_stats = {
        'cpu_attempts': 0,
        'cpu_successes': 0,
    }

    @classmethod
    def print_stats(cls):
        """Print fallback usage statistics."""
        if cls._fallback_stats['cpu_attempts'] > 0:
            print("\n=== MIOpen Fallback Statistics ===")
            print(f"CPU fallback attempts: {cls._fallback_stats['cpu_attempts']}, "
                  f"successes: {cls._fallback_stats['cpu_successes']}")
            print("===================================\n")


@contextmanager
def miopen_safe_execution(operation_name: str = "operation", enable_fallback: bool = True):
    """
    Context manager for safe execution with MIOpen fallback.

    Note: This is now mainly for logging. The actual fallback happens in the
    monkey-patched conv3d function. Naive algorithms are already in MIOpen's
    search space, so MIOpen will try them automatically.

    Usage:
        with miopen_safe_execution("VAE decode"):
            output = vae.decode(latents)

    Args:
        operation_name: Name of the operation for logging
        enable_fallback: Whether to enable fallback mechanisms
    """
    try:
        yield
    except RuntimeError as e:
        if not enable_fallback:
            raise

        error_msg = str(e)
        if "miopenStatusUnknownError" not in error_msg and "MIOpen" not in error_msg:
            raise

        # MIOpen error occurred - log it and re-raise
        # The monkey-patched conv3d will handle CPU fallback if needed
        print(f"[MIOpen Fallback] Error in {operation_name}: {error_msg}")
        raise


def with_miopen_fallback(func: Callable) -> Callable:
    """
    Decorator for automatic MIOpen fallback.

    Note: This is now mainly for error logging. The actual fallback (naive GPU, then CPU)
    happens automatically in the monkey-patched conv3d function.

    Usage:
        @with_miopen_fallback
        def my_vae_decode(latents, vae):
            return vae.decode(latents)
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            if "miopenStatusUnknownError" in error_msg or "MIOpen" in error_msg:
                print(f"[MIOpen Fallback] {func.__name__} encountered MIOpen error")
                print(f"[MIOpen Fallback] Error will be handled by conv3d fallback")
            raise

    return wrapper


def apply_miopen_monkey_patch():
    """
    Monkey-patch torch.nn.functional.conv3d to add automatic fallback.
    This is a global solution that doesn't require modifying PyTorch source.
    """
    import torch.nn.functional as F

    # Store original conv3d
    _original_conv3d = F.conv3d

    def patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """
        Patched conv3d with MIOpen fallback.

        Note: Naive convolution algorithms are added to MIOpen's search space at startup
        via MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1 in demo_gradio.py.

        MIOpen's Find phase will:
        1. Try optimized algorithms (XDLOPS, implicit GEMM, etc.)
        2. Automatically fall back to naive if optimized fails
        3. Only reach this CPU fallback if MIOpen Find completely fails

        This means the CPU fallback is truly last resort - triggered only when
        MIOpen cannot find ANY working GPU algorithm (including naive).
        """
        try:
            return _original_conv3d(input, weight, bias, stride, padding, dilation, groups)
        except RuntimeError as e:
            error_msg = str(e)
            if "miopenStatusUnknownError" not in error_msg and "MIOpen" not in error_msg:
                raise

            # MIOpen Find failed completely (all GPU algorithms including naive failed)
            print("[MIOpen Fallback] MIOpen algorithm search failed (optimized + naive algorithms)")
            print("[MIOpen Fallback] Using CPU computation as last resort...")
            MIOpenFallbackHandler._fallback_stats['cpu_attempts'] += 1

            device_orig = input.device
            dtype_orig = input.dtype

            input_cpu = input.cpu().float()
            weight_cpu = weight.cpu().float()
            bias_cpu = bias.cpu().float() if bias is not None else None

            result = _original_conv3d(input_cpu, weight_cpu, bias_cpu, stride, padding, dilation, groups)

            MIOpenFallbackHandler._fallback_stats['cpu_successes'] += 1
            print("[MIOpen Fallback] ✓ CPU computation succeeded!")
            return result.to(device=device_orig, dtype=dtype_orig)

    # Replace the function
    F.conv3d = patched_conv3d
    print("[MIOpen Fallback] Monkey-patch applied to torch.nn.functional.conv3d")


# Convenience function for initialization
def initialize_miopen_fallback(use_monkey_patch: bool = True, verbose: bool = True):
    """
    Initialize MIOpen fallback system.

    Args:
        use_monkey_patch: If True, applies global monkey-patch to conv3d
        verbose: If True, prints initialization message
    """
    if use_monkey_patch:
        apply_miopen_monkey_patch()

    if verbose:
        print("[MIOpen Fallback] Fallback system initialized")
        print("[MIOpen Fallback] Strategy: Optimized GPU → Naive GPU (auto) → CPU (last resort)")
        print("[MIOpen Fallback] Naive algorithms in search space - no performance penalty when not used")
