#!/usr/bin/env python3

import argparse
import os
from typing import List, Dict

import torch

from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.utils import save_bcthw_as_mp4, soft_append_bcthw
from diffusers_helper.hunyuan import vae_decode


CHANNELS_LAST_3D = getattr(torch, 'channels_last_3d', torch.contiguous_format)


def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None or tensor.dim() != 5:
        return tensor
    return tensor.contiguous(memory_format=CHANNELS_LAST_3D)


def load_vae(device: torch.device) -> AutoencoderKLHunyuanVideo:
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16,
    )
    vae.to(device=device, dtype=torch.float16)
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    vae.eval()
    return vae


@torch.no_grad()
def reconstruct_video_from_segments(
    segments: List[Dict[str, torch.Tensor]],
    metadata: Dict[str, int],
    vae: AutoencoderKLHunyuanVideo,
    output_path: str,
) -> str:
    if not segments:
        raise ValueError('No latent segments found in the provided file.')

    latent_window_size = int(metadata.get('latent_window_size', 9))
    height = int(metadata.get('height'))
    width = int(metadata.get('width'))
    fps = int(metadata.get('fps', 30))
    mp4_crf = int(metadata.get('mp4_crf', 16))

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

    for segment in segments:
        generated_latents = segment['generated_latents'].to(device)
        generated_latents = _ensure_channels_last_3d(generated_latents)
        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        real_history_latents = _ensure_channels_last_3d(real_history_latents)

        if history_pixels is None:
            decoded = vae_decode(real_history_latents, vae)
            history_pixels = _ensure_channels_last_3d(decoded).cpu()
        else:
            is_last_section = bool(segment.get('is_last_section'))
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            current_latents = real_history_latents[:, :, :section_latent_frames, :, :]
            decoded = vae_decode(current_latents, vae)
            decoded = _ensure_channels_last_3d(decoded).cpu()
            history_pixels = soft_append_bcthw(decoded, history_pixels, overlap_frames)
            history_pixels = _ensure_channels_last_3d(history_pixels)

    save_bcthw_as_mp4(history_pixels, output_path, fps=fps, crf=mp4_crf)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resume video generation from saved FramePack latents.')
    parser.add_argument('--latents', required=True, help='Path to the saved *_latents.pt file.')
    parser.add_argument('--device', default='cuda:0', help='Device for VAE decoding (e.g., cuda:0 or cpu).')
    parser.add_argument('--output-dir', default='./outputs', help='Directory for the reconstructed video output.')
    parser.add_argument('--output', default=None, help='Optional filename for the output video (defaults to <job_id>_resume.mp4).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latents_pkg = torch.load(args.latents, map_location='cpu')
    metadata = latents_pkg.get('metadata', {})
    segments = latents_pkg.get('segments', [])

    os.makedirs(args.output_dir, exist_ok=True)
    job_id = metadata.get('job_id', 'framepack')
    output_name = args.output or f'{job_id}_resume.mp4'
    output_path = os.path.join(args.output_dir, output_name)

    device = torch.device(args.device)
    vae = load_vae(device)

    reconstructed_path = reconstruct_video_from_segments(segments, metadata, vae, output_path)
    print(f'Reconstructed video saved to {reconstructed_path}')


if __name__ == '__main__':
    main()
