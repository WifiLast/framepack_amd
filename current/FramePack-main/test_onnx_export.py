#!/usr/bin/env python3
"""
Minimal script to test ONNX export of HunyuanVideo VAE decoder.
"""

import os

# Set GPU as only visible device BEFORE importing torch
# This ensures ONNX export only sees the GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from diffusers import AutoencoderKLHunyuanVideo
import tempfile
import shutil

# Monkey-patch diffusers to fix device mismatch in prepare_causal_attention_mask
import diffusers.models.autoencoders.autoencoder_kl_hunyuan_video as hunyuan_vae_module

_original_prepare_causal_attention_mask = hunyuan_vae_module.prepare_causal_attention_mask
_global_export_device = None

def _patched_prepare_causal_attention_mask(*args, **kwargs):
    """Patched version that ensures all tensors are on the correct device."""
    global _global_export_device

    print(f"[PATCH] prepare_causal_attention_mask called with args={len(args)}, kwargs={list(kwargs.keys())}")

    # Handle both positional and keyword arguments
    # Actual signature from diffusers source:
    # prepare_causal_attention_mask(num_frames, height_width, dtype, device, batch_size=None)
    if len(args) >= 2:
        num_frames = args[0]
        height_width = args[1]  # This is already computed by the caller
        dtype = args[2] if len(args) > 2 else kwargs.get('dtype', torch.float32)
        device = args[3] if len(args) > 3 else kwargs.get('device', None)
        batch_size = args[4] if len(args) > 4 else kwargs.get('batch_size', None)
    else:
        num_frames = kwargs.get('num_frames', args[0] if len(args) > 0 else None)
        height_width = kwargs.get('height_width', args[1] if len(args) > 1 else None)
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', None)
        batch_size = kwargs.get('batch_size', None)

    print(f"[PATCH] Parsed args - num_frames={num_frames}, height_width={height_width}, batch_size={batch_size}, dtype={dtype}, device={device}")

    # Use global export device if set (during ONNX tracing)
    if _global_export_device is not None:
        device = _global_export_device
        print(f"[PATCH] Using global export device: {device}")
    else:
        # Force device to match the global export device type
        # If no global device set, use CUDA if available
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            print(f"[PATCH] Device was None, set to: {device}")
        elif isinstance(device, str):
            device = torch.device(device)
            print(f"[PATCH] Converted string device to: {device}")
        elif isinstance(device, torch.device):
            # Device is already a torch.device, use as-is
            print(f"[PATCH] Using provided device: {device}")

        # Always ensure device type matches CUDA availability
        if device.type == 'cpu' and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[PATCH] Forced CPU device to CUDA: {device}")

    # Create tensors directly on the target device
    # Ensure all numeric arguments are Python ints, not tensors
    if isinstance(num_frames, torch.Tensor):
        num_frames = int(num_frames.item())
        print(f"[PATCH] Converted num_frames tensor to int: {num_frames}")

    if isinstance(height_width, torch.Tensor):
        height_width = int(height_width.item())
        print(f"[PATCH] Converted height_width tensor to int: {height_width}")

    if isinstance(batch_size, torch.Tensor):
        batch_size = int(batch_size.item())
        print(f"[PATCH] Converted batch_size tensor to int: {batch_size}")

    print(f"[PATCH] Creating indices tensor on device: {device}, num_frames={num_frames}, height_width={height_width}")

    indices = torch.arange(num_frames, device=device, dtype=torch.long)
    print(f"[PATCH] indices device: {indices.device}, dtype: {indices.dtype}, shape: {indices.shape}")

    indices_blocks = indices.repeat_interleave(height_width)
    print(f"[PATCH] indices_blocks device: {indices_blocks.device}, shape: {indices_blocks.shape}")

    causal_mask = indices_blocks.unsqueeze(0) >= indices_blocks.unsqueeze(1)
    causal_mask = causal_mask.to(dtype=dtype, device=device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    print(f"[PATCH] Returning causal_mask on device: {causal_mask.device}")
    return causal_mask

hunyuan_vae_module.prepare_causal_attention_mask = _patched_prepare_causal_attention_mask


class _VaeDecodeWrapper(torch.nn.Module):
    """Wrap the Diffusers VAE to capture scaling inside the graph for ONNX export."""

    def __init__(self, vae: AutoencoderKLHunyuanVideo):
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(getattr(vae.config, 'scaling_factor', 0.18215))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.vae.decode(latents / self.scaling_factor).sample
        return decoded


def main():
    global _global_export_device

    print("="*80)
    print("Minimal ONNX Export Test (CPU Mode - TensorRT 10.0.1)")
    print("="*80)

    # Configuration - Force CPU for ONNX export
    device = torch.device("cpu")
    torch_dtype = torch.float32
    print(f"\n[INFO] Using {device} device for ONNX export")
    print("[INFO] Running on CPU to avoid CUDA memory issues")

    output_dir = "./test_onnx_output"
    onnx_path = os.path.join(output_dir, "vae_decoder_test.onnx")

    # Correct HunyuanVideo latent dimensions
    # Output video resolution: 544x704 (width x height)
    # VAE downsamples by 8x, so latent dimensions are: 68x88 (width x height)
    # PyTorch tensor format is (batch, channels, frames, height, width)
    batch_size = 1
    channels = 16  # VAE latent channels
    frames = 5     # Number of frames
    height = 88    # Latent height (704 / 8)
    width = 68     # Latent width (544 / 8)
    test_shape = (batch_size, channels, frames, height, width)

    print(f"\nDevice: {device}")
    print(f"Dtype: {torch_dtype}")
    print(f"Test shape: {test_shape}")
    print(f"Output path: {onnx_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load VAE
    print("\n[1/4] Loading VAE model...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch_dtype,
    )
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    vae.requires_grad_(False)
    print(f"VAE loaded on {device}")

    # Disable features that can cause issues
    print("\n[2/4] Configuring VAE for export...")
    if hasattr(vae, 'use_tiling'):
        vae.use_tiling = False
        print("Disabled tiling")
    if hasattr(vae, 'use_slicing'):
        vae.use_slicing = False
        print("Disabled slicing")
    if hasattr(vae, 'use_framewise_decoding'):
        vae.use_framewise_decoding = False
        print("Disabled framewise decoding")

    # Set global device for patched function
    _global_export_device = device
    print(f"Set global export device to: {device}")

    # Create wrapper and example input
    print("\n[3/4] Creating wrapper and example input...")
    wrapper = _VaeDecodeWrapper(vae).to(device=device, dtype=torch_dtype)
    example = torch.randn(test_shape, device=device, dtype=torch_dtype)
    print(f"Example input shape: {example.shape}")

    # Export to ONNX
    print("\n[4/4] Exporting to ONNX...")
    print("This may take a few minutes...")
    print("\n[ONNX] Using opset 17 for TensorRT 10.0.1 compatibility")
    print("[ONNX] TensorRT 10.0.1 has full support for modern ONNX operators")
    print("[ONNX] Model >2GB will use external data storage")

    try:
        # First export to a temporary file to avoid the 2GB limit error
        # Then convert to use external data if needed
        temp_dir = tempfile.mkdtemp()
        temp_onnx = os.path.join(temp_dir, "model.onnx")

        print("[ONNX] Exporting model (this may take several minutes)...")

        with torch.inference_mode():
            # Use temporary BytesIO first to avoid immediate protobuf limit
            torch.onnx.export(
                wrapper,
                example,
                temp_onnx,
                input_names=['latents'],
                output_names=['decoded'],
                dynamic_axes={
                    'latents': {2: 'frames'},
                    'decoded': {2: 'frames'},
                },
                opset_version=17,  # Use opset 17 for TensorRT 10.0.1
                do_constant_folding=True,
                verbose=False,
                dynamo=True,
            )

        print("[ONNX] Converting to external data format...")
        # Load the model and convert to external data format
        try:
            import onnx

            # Load model
            model = onnx.load(temp_onnx)

            # Convert to external data (stores weights in separate file)
            onnx.save_model(
                model,
                onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="vae_decoder_test.onnx.data",
                size_threshold=1024,  # Store tensors >1KB externally
            )

            print(f"[ONNX] Model saved with external data")

        except (ImportError, Exception) as e:
            # If onnx package not available or conversion fails, just copy the temp file
            print(f"[ONNX] Warning: Could not use external data ({e}), copying model as-is")
            shutil.copy2(temp_onnx, onnx_path)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        # Verify file was created
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024*1024)

            # Check for external data file
            data_file = onnx_path + ".data"
            if os.path.exists(data_file):
                data_size = os.path.getsize(data_file) / (1024*1024)
                print(f"\n{'='*80}")
                print("SUCCESS!")
                print(f"ONNX model exported to: {onnx_path}")
                print(f"Model file size: {file_size:.2f} MB")
                print(f"External data file: {data_file}")
                print(f"External data size: {data_size:.2f} MB")
                print(f"Total size: {file_size + data_size:.2f} MB")
                print("="*80)
            else:
                print(f"\n{'='*80}")
                print("SUCCESS!")
                print(f"ONNX model exported to: {onnx_path}")
                print(f"File size: {file_size:.2f} MB")
                print("="*80)
        else:
            print("\nERROR: ONNX file was not created!")

    except Exception as e:
        print(f"\nERROR during ONNX export:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clear global device
        _global_export_device = None
        print("\nCleaning up...")
        del wrapper
        del vae
        torch.cuda.empty_cache()
        print("Done!")


if __name__ == "__main__":
    main()
