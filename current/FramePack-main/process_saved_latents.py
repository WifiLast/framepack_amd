#!/usr/bin/env python3

import argparse
import os
from typing import List, Dict

import torch

from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.utils import save_bcthw_as_mp4, soft_append_bcthw
from diffusers_helper.hunyuan import vae_decode

# Separate cache directories for NVIDIA CUDA to prevent CUDA/ROCm interference
_cache_base = os.path.join(os.path.dirname(__file__), '.cache_cuda')
os.makedirs(_cache_base, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = os.path.join(_cache_base, 'triton')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_cache_base, 'torch_extensions')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_cache_base, 'inductor')

CHANNELS_LAST_3D = getattr(torch, 'channels_last_3d', torch.contiguous_format)


def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None or tensor.dim() != 5:
        return tensor
    return tensor.contiguous(memory_format=CHANNELS_LAST_3D)


def load_vae(device: torch.device, enable_tiling: bool = True, enable_slicing: bool = False) -> AutoencoderKLHunyuanVideo:
    """Load VAE model optimized for GPU inference."""
    print(f'Loading VAE on {device}...')
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16,
    )
    vae.to(device=device, dtype=torch.float16)
    vae.to(memory_format=CHANNELS_LAST_3D)

    if enable_tiling and hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        print('VAE tiling enabled')

    if enable_slicing and hasattr(vae, 'enable_slicing'):
        vae.enable_slicing()
        print('VAE slicing enabled (for lower VRAM usage)')

    vae.eval()
    vae.requires_grad_(False)
    print(f'VAE loaded successfully')
    return vae


@torch.no_grad()
def reconstruct_video_from_segments(
    segments: List[Dict[str, torch.Tensor]],
    metadata: Dict[str, int],
    vae: AutoencoderKLHunyuanVideo,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Reconstruct video from latent segments using GPU-accelerated VAE decoding.

    Args:
        segments: List of latent segments from the saved file
        metadata: Metadata dictionary with video parameters
        vae: VAE model already loaded on GPU
        output_path: Path to save the output video
        verbose: Whether to print progress information

    Returns:
        Path to the saved video file
    """
    if not segments:
        raise ValueError('No latent segments found in the provided file.')

    latent_window_size = int(metadata.get('latent_window_size', 9))
    height = int(metadata.get('height'))
    width = int(metadata.get('width'))
    fps = int(metadata.get('fps', 30))
    mp4_crf = int(metadata.get('mp4_crf', 16))
    total_segments = len(segments)

    if verbose:
        print(f'\nReconstruction Info:')
        print(f'  Total segments: {total_segments}')
        print(f'  Resolution: {width}x{height}')
        print(f'  FPS: {fps}')
        print(f'  Latent window size: {latent_window_size}')
        print(f'  Output: {output_path}\n')

    base_dtype = segments[0]['generated_latents'].dtype
    device = next(vae.parameters()).device

    history_latents = torch.zeros(
        size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
        dtype=base_dtype,
        device=device,
    )
    history_latents = _ensure_channels_last_3d(history_latents)

    history_pixels = None
    total_generated_latent_frames = 0
    overlap_frames = latent_window_size * 4 - 3

    for idx, segment in enumerate(segments, 1):
        if verbose:
            print(f'Processing segment {idx}/{total_segments}...')

        generated_latents = segment['generated_latents'].to(device=device, dtype=base_dtype)
        generated_latents = _ensure_channels_last_3d(generated_latents)
        segment_latent_frames = int(generated_latents.shape[2])

        # Match the original logic from demo_gradio.py
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        total_generated_latent_frames += segment_latent_frames
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        real_history_latents = _ensure_channels_last_3d(real_history_latents)

        if history_pixels is None:
            if verbose:
                print(f'  Decoding initial segment (latent frames: {real_history_latents.shape[2]})...')
            decoded = vae_decode(real_history_latents, vae)
            history_pixels = _ensure_channels_last_3d(decoded).cpu()
            if verbose:
                print(f'  Initial decode complete, pixel frames: {history_pixels.shape[2]}')
        else:
            is_last_section = bool(segment.get('is_last_section'))
            expected_section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            section_latent_frames = int(segment_latent_frames)
            if verbose and section_latent_frames != expected_section_latent_frames:
                print(
                    f'  Warning: segment latent frames {section_latent_frames} != expected {expected_section_latent_frames}; '
                    f'using actual segment length for decode'
                )
            current_latents = real_history_latents[:, :, :section_latent_frames, :, :]
            if verbose:
                print(f'  Decoding segment (latent frames: {current_latents.shape[2]}, last={is_last_section})...')
            current_pixels = vae_decode(current_latents, vae)
            current_pixels = _ensure_channels_last_3d(current_pixels).cpu()
            if verbose:
                print(f'  Before append: history_pixels={history_pixels.shape[2]} frames, current_pixels={current_pixels.shape[2]} frames')
            # soft_append_bcthw expects: soft_append_bcthw(history, current, overlap)
            # But we're appending NEW pixels to the FRONT, so we need to flip the arguments
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap_frames)
            history_pixels = _ensure_channels_last_3d(history_pixels)
            if verbose:
                print(f'  After append: history_pixels={history_pixels.shape[2]} frames')

        # Clear GPU cache periodically
        if idx % 2 == 0:
            torch.cuda.empty_cache()

    total_frames = history_pixels.shape[2]
    duration = total_frames / fps
    if verbose:
        print(f'\nDecoding complete!')
        print(f'  Total frames: {total_frames}')
        print(f'  Duration: {duration:.2f} seconds')
        print(f'  Saving video to {output_path}...')

    save_bcthw_as_mp4(history_pixels, output_path, fps=fps, crf=mp4_crf)

    if verbose:
        print(f'Video saved successfully!')

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Resume video generation from saved FramePack latents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process on NVIDIA GPU (default)
  python process_saved_latents.py --latents outputs/20250101_123456_latents.pt

  # Process on specific GPU with custom output
  python process_saved_latents.py --latents outputs/20250101_123456_latents.pt --device cuda:1 --output my_video.mp4

  # Process on CPU (slow, for testing)
  python process_saved_latents.py --latents outputs/20250101_123456_latents.pt --device cpu

  # Enable slicing for lower VRAM usage
  python process_saved_latents.py --latents outputs/20250101_123456_latents.pt --enable-slicing
        '''
    )
    parser.add_argument('--latents', required=True, help='Path to the saved *_latents.pt file.')
    parser.add_argument('--device', default='cuda:0', help='Device for VAE decoding (e.g., cuda:0, cuda:1, or cpu). Default: cuda:0')
    parser.add_argument('--output-dir', default='./outputs', help='Directory for the reconstructed video output. Default: ./outputs')
    parser.add_argument('--output', default=None, help='Optional filename for the output video (defaults to <job_id>_resume.mp4).')
    parser.add_argument('--enable-slicing', action='store_true', help='Enable VAE slicing for lower VRAM usage (slower but uses less memory).')
    parser.add_argument('--disable-tiling', action='store_true', help='Disable VAE tiling (not recommended, may cause OOM).')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f'Loading latent file: {args.latents}')
    latents_pkg = torch.load(args.latents, map_location='cpu')
    metadata = latents_pkg.get('metadata', {})
    segments = latents_pkg.get('segments', [])

    if not segments:
        print('Error: No segments found in latent file!')
        return

    version = metadata.get('version', 0)
    print(f'Latent file version: {version}')

    os.makedirs(args.output_dir, exist_ok=True)
    job_id = metadata.get('job_id', 'framepack')
    output_name = args.output or f'{job_id}_resume.mp4'
    output_path = os.path.join(args.output_dir, output_name)

    device = torch.device(args.device)
    print(f'Using device: {device}')

    if device.type == 'cuda' and not torch.cuda.is_available():
        print('Error: CUDA requested but not available! Falling back to CPU.')
        device = torch.device('cpu')

    vae = None
    try:
        vae = load_vae(device, enable_tiling=not args.disable_tiling, enable_slicing=args.enable_slicing)

        reconstructed_path = reconstruct_video_from_segments(
            segments, metadata, vae, output_path, verbose=not args.quiet
        )
        print(f'\n✓ Reconstructed video saved to: {reconstructed_path}')

    finally:
        # Clean up: unload VAE and clear CUDA cache
        if vae is not None:
            print('\nCleaning up GPU memory...')
            del vae

        # Clear all cached tensors
        if 'segments' in locals():
            del segments
        if 'latents_pkg' in locals():
            del latents_pkg

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if using CUDA
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print('GPU memory cleared')


if __name__ == '__main__':
    main()
