"""
Transformer Engine wrapper for text encoders (LlamaModel and CLIPTextModel).

This module provides functions to convert standard PyTorch Linear layers to
Transformer Engine Linear layers with FP8 support for AMD ROCm GPUs.

Usage:
    from te_text_encoder_wrapper import convert_model_to_te, te_encode_wrapper

    # Convert model
    text_encoder = convert_model_to_te(text_encoder, verbose=True)

    # Use with FP8 autocast
    with te_encode_wrapper(enabled=True):
        output = text_encoder(input_ids)
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional, Tuple

# Try to import Transformer Engine
HAS_TRANSFORMER_ENGINE = False
te = None
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TRANSFORMER_ENGINE = True
    print("✓ Transformer Engine available for FP8 optimization")
except ImportError:
    print("⚠ Transformer Engine not available - install with:")
    print("  pip install transformer_engine (see AMD ROCm installation guide)")
    print("  Falling back to standard PyTorch operations")


def get_fp8_recipe():
    """
    Create FP8 recipe for Transformer Engine.

    Returns:
        DelayedScaling recipe for FP8 training/inference
    """
    if not HAS_TRANSFORMER_ENGINE:
        return None

    # Use HYBRID format for best compatibility on AMD MI300
    # E4M3 for forward pass, E5M2 for gradients
    fp8_format = Format.HYBRID

    # Recipe configuration optimized for inference
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=16,  # Shorter history for inference
        amax_compute_algo="max",  # Use max for stability
        override_linear_precision=(False, False, False)  # Let TE decide precision
    )

    return fp8_recipe


@contextmanager
def te_encode_wrapper(enabled: bool = True, fp8_recipe=None):
    """
    Context manager for FP8 autocast during encoding.

    Args:
        enabled: Whether to enable FP8 (requires Transformer Engine)
        fp8_recipe: Optional FP8 recipe, creates default if None

    Yields:
        Context for model execution

    Example:
        with te_encode_wrapper(enabled=True):
            output = model(input)
    """
    if not HAS_TRANSFORMER_ENGINE or not enabled:
        # No-op context manager
        yield
        return

    if fp8_recipe is None:
        fp8_recipe = get_fp8_recipe()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        yield


def replace_linear_with_te(
    model: nn.Module,
    parent_name: str = "",
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Recursively replace nn.Linear layers with te.Linear layers.

    Args:
        model: PyTorch module to convert
        parent_name: Name prefix for nested modules (used for logging)
        verbose: Whether to print conversion details

    Returns:
        Tuple of (num_replaced, num_failed)
    """
    if not HAS_TRANSFORMER_ENGINE:
        if verbose:
            print("⚠ Transformer Engine not available - no conversion performed")
        return 0, 0

    num_replaced = 0
    num_failed = 0

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, nn.Linear):
            try:
                # Extract Linear layer properties
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias is not None
                device = module.weight.device
                dtype = module.weight.dtype

                # Create TE Linear layer
                # Note: TE Linear doesn't support all nn.Linear parameters
                # We use basic configuration suitable for inference
                te_linear = te.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    params_dtype=dtype,
                    device=device,
                    # Skip distributed parameters for single-GPU inference
                )

                # Copy weights and bias from original layer
                with torch.no_grad():
                    te_linear.weight.copy_(module.weight)
                    if bias:
                        te_linear.bias.copy_(module.bias)

                # Replace the module
                setattr(model, name, te_linear)
                num_replaced += 1

                if verbose:
                    print(f"  ✓ Replaced {full_name}: Linear({in_features}, {out_features}) -> te.Linear")

            except Exception as e:
                num_failed += 1
                if verbose:
                    print(f"  ✗ Failed to replace {full_name}: {e}")
        else:
            # Recursively process child modules
            child_replaced, child_failed = replace_linear_with_te(
                module,
                parent_name=full_name,
                verbose=verbose
            )
            num_replaced += child_replaced
            num_failed += child_failed

    return num_replaced, num_failed


def convert_model_to_te(
    model: nn.Module,
    verbose: bool = True,
    keep_original_device: bool = True
) -> nn.Module:
    """
    Convert a PyTorch model to use Transformer Engine Linear layers.

    This function replaces all nn.Linear layers with te.Linear layers
    that support FP8 quantization on AMD ROCm GPUs.

    Args:
        model: PyTorch model to convert (LlamaModel, CLIPTextModel, etc.)
        verbose: Whether to print conversion details
        keep_original_device: Keep model on original device after conversion

    Returns:
        Converted model with TE Linear layers

    Example:
        text_encoder = LlamaModel.from_pretrained(...)
        text_encoder = convert_model_to_te(text_encoder, verbose=True)

        # Use with FP8 autocast
        with te_encode_wrapper(enabled=True):
            output = text_encoder(input_ids)
    """
    if not HAS_TRANSFORMER_ENGINE:
        if verbose:
            print("\n" + "="*60)
            print("⚠ Transformer Engine not available")
            print("="*60)
            print("Install with: pip install transformer_engine")
            print("See: cache/TransformerEngine-dev/README.rst for AMD ROCm")
            print("Returning original model without conversion")
            print("="*60 + "\n")
        return model

    if verbose:
        print("\n" + "="*60)
        print("Converting model to Transformer Engine")
        print("="*60)
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print("="*60)

    # Store original device
    original_device = next(model.parameters()).device

    # Perform conversion
    num_replaced, num_failed = replace_linear_with_te(model, verbose=verbose)

    if verbose:
        print("="*60)
        print(f"Conversion complete:")
        print(f"  ✓ Replaced: {num_replaced} Linear layers")
        if num_failed > 0:
            print(f"  ✗ Failed: {num_failed} Linear layers")
        print("="*60 + "\n")

    # Restore original device if requested
    if keep_original_device and original_device.type != 'cpu':
        model = model.to(original_device)

    return model


def create_te_quantization_config():
    """
    Create a configuration dict for TE-based quantization.

    This is an alternative to BitsAndBytesConfig that uses
    Transformer Engine's FP8 quantization instead.

    Returns:
        Dict with TE configuration settings
    """
    if not HAS_TRANSFORMER_ENGINE:
        return None

    config = {
        'use_fp8': True,
        'fp8_recipe': get_fp8_recipe(),
        'description': 'Transformer Engine FP8 quantization (AMD ROCm)',
    }

    return config


def load_model_with_te(
    model_class,
    model_name: str,
    subfolder: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    convert_to_te: bool = True,
    verbose: bool = True
):
    """
    Load a model from HuggingFace and optionally convert to TE.

    This is a convenience function that combines model loading and TE conversion.

    Args:
        model_class: Model class (LlamaModel, CLIPTextModel, etc.)
        model_name: Model name/path for from_pretrained
        subfolder: Optional subfolder in the model repository
        dtype: Torch dtype for the model
        convert_to_te: Whether to convert Linear layers to TE
        verbose: Whether to print loading details

    Returns:
        Loaded model (with TE conversion if requested)

    Example:
        text_encoder = load_model_with_te(
            LlamaModel,
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder',
            dtype=torch.float16,
            convert_to_te=True
        )
    """
    if verbose:
        print(f"\nLoading {model_class.__name__} from {model_name}")
        if subfolder:
            print(f"  Subfolder: {subfolder}")
        print(f"  Dtype: {dtype}")
        print(f"  Convert to TE: {convert_to_te}")

    # Build loading kwargs
    load_kwargs = {
        "torch_dtype": dtype,
    }

    if subfolder:
        load_kwargs["subfolder"] = subfolder

    # Load model
    model = model_class.from_pretrained(model_name, **load_kwargs).cpu()

    if verbose:
        print(f"  ✓ Model loaded")

    # Convert to TE if requested
    if convert_to_te:
        model = convert_model_to_te(model, verbose=verbose)

    return model


# Example usage and testing
if __name__ == "__main__":
    print("Transformer Engine Text Encoder Wrapper")
    print("="*60)

    if HAS_TRANSFORMER_ENGINE:
        print("✓ Transformer Engine is available")
        print(f"  Version: {te.__version__ if hasattr(te, '__version__') else 'unknown'}")

        # Create a simple test model
        print("\nCreating test model...")
        test_model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        print(f"Original model has {sum(1 for m in test_model.modules() if isinstance(m, nn.Linear))} Linear layers")

        # Convert to TE
        print("\nConverting to Transformer Engine...")
        test_model = convert_model_to_te(test_model, verbose=True)

        # Test forward pass with FP8
        print("\nTesting forward pass with FP8 autocast...")
        test_input = torch.randn(2, 512, 768)

        with te_encode_wrapper(enabled=True):
            output = test_model(test_input)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

    else:
        print("✗ Transformer Engine is not available")
        print("  Install it to use FP8 quantization on AMD ROCm GPUs")
