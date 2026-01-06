#!/usr/bin/env python3
"""
REST API server for processing saved FramePack latents.
Preloads the VAE model and accepts POST requests for video reconstruction.
"""

import argparse
import os
import gc
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


import torch
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

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


# ========================= Models & Schemas =========================

class ProcessRequest(BaseModel):
    """Request model for video processing"""
    latents_path: str = Field(..., description="Path to the saved *_latents.pt file")
    output_name: Optional[str] = Field(None, description="Optional output video filename")
    enable_slicing: bool = Field(False, description="Enable VAE slicing for lower VRAM")
    verbose: bool = Field(True, description="Enable verbose logging")


class ProcessResponse(BaseModel):
    """Response model for successful processing"""
    status: str
    output_path: str
    total_frames: int
    duration: float
    resolution: str
    fps: int
    processing_time: float


class ErrorResponse(BaseModel):
    """Response model for errors"""
    status: str = "error"
    message: str
    details: Optional[str] = None


# ========================= Global State =========================

class APIState:
    """Global state for the API server"""
    def __init__(self):
        self.vae: Optional[AutoencoderKLHunyuanVideo] = None
        self.device: Optional[torch.device] = None
        self.output_dir: str = "./outputs"
        self.enable_tiling: bool = True
        self.base_enable_slicing: bool = False
        self.is_ready: bool = False
        self.processing: bool = False


state = APIState()


# ========================= Helper Functions =========================

def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in channels_last_3d format"""
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
) -> tuple[str, int, float]:
    """
    Reconstruct video from latent segments using GPU-accelerated VAE decoding.

    Returns:
        Tuple of (output_path, total_frames, duration)
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

    # Debug: Check latent statistics
    first_latent = segments[0]['generated_latents']
    print(f'  DEBUG: First latent dtype={first_latent.dtype}, device={first_latent.device}')
    print(f'  DEBUG: First latent shape={first_latent.shape}')
    print(f'  DEBUG: First latent range: min={first_latent.min():.4f}, max={first_latent.max():.4f}, mean={first_latent.mean():.4f}')
    print(f'  DEBUG: Contains NaN: {torch.isnan(first_latent).any()}, Contains Inf: {torch.isinf(first_latent).any()}')

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

        # Ensure proper dtype and device - convert from CPU/ROCm to CUDA if needed
        generated_latents = segment['generated_latents']

        # Convert to proper dtype (fp32 -> fp16 if needed)
        if generated_latents.dtype != base_dtype:
            if verbose:
                print(f'  Converting latent dtype from {generated_latents.dtype} to {base_dtype}')
            generated_latents = generated_latents.to(dtype=base_dtype)

        # Move to GPU
        generated_latents = generated_latents.to(device=device)
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

    return output_path, total_frames, duration


# ========================= FastAPI Application =========================

app = FastAPI(
    title="FramePack Latent Processing API",
    description="REST API for processing saved FramePack latents with preloaded VAE model",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the VAE model on server startup"""
    print("=" * 60)
    print("Starting FramePack Latent Processing API Server")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(state.output_dir, exist_ok=True)
    print(f"Output directory: {state.output_dir}")

    # Load VAE model
    try:
        print(f"Device: {state.device}")
        state.vae = load_vae(
            state.device,
            enable_tiling=state.enable_tiling,
            enable_slicing=state.base_enable_slicing
        )
        state.is_ready = True
        print("=" * 60)
        print("✓ Server ready to accept requests")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: Failed to load VAE: {e}")
        state.is_ready = False
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown"""
    print("\nShutting down server...")
    if state.vae is not None:
        print("Cleaning up GPU memory...")
        del state.vae
        state.vae = None
        gc.collect()
        if state.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✓ Cleanup complete")


@app.get("/")
async def root():
    """Root endpoint - returns API status"""
    return {
        "service": "FramePack Latent Processing API",
        "version": "1.0.0",
        "status": "ready" if state.is_ready else "not_ready",
        "device": str(state.device),
        "processing": state.processing
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="VAE model not loaded")

    return {
        "status": "healthy",
        "vae_loaded": state.vae is not None,
        "device": str(state.device),
        "processing": state.processing
    }


@app.post("/process", response_model=ProcessResponse)
async def process_latents(request: ProcessRequest):
    """
    Process saved latent file and generate video.

    The VAE model is already preloaded, so this endpoint will process requests immediately.
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Server not ready - VAE model not loaded")

    if state.processing:
        raise HTTPException(status_code=409, detail="Server is currently processing another request. Please try again later.")

    start_time = datetime.now()
    state.processing = True

    try:
        # Validate latents file exists
        latents_path = Path(request.latents_path)
        if not latents_path.exists():
            raise HTTPException(status_code=404, detail=f"Latents file not found: {request.latents_path}")

        print(f"\n{'='*60}")
        print(f"Processing request: {latents_path.name}")
        print(f"{'='*60}")

        # Load latent file
        print(f'Loading latent file: {latents_path}')
        latents_pkg = torch.load(str(latents_path), map_location='cpu')
        metadata = latents_pkg.get('metadata', {})
        segments = latents_pkg.get('segments', [])

        if not segments:
            raise HTTPException(status_code=400, detail="No segments found in latent file")

        version = metadata.get('version', 0)
        print(f'Latent file version: {version}')

        # Determine output path
        job_id = metadata.get('job_id', 'framepack')
        output_name = request.output_name or f'{job_id}_resume.mp4'
        output_path = os.path.join(state.output_dir, output_name)

        # Enable slicing if requested
        if request.enable_slicing and not state.base_enable_slicing:
            if hasattr(state.vae, 'enable_slicing'):
                state.vae.enable_slicing()
                print('VAE slicing enabled for this request')

        # Process the video
        reconstructed_path, total_frames, duration = reconstruct_video_from_segments(
            segments, metadata, state.vae, output_path, verbose=request.verbose
        )

        # Cleanup
        del segments, latents_pkg
        gc.collect()
        if state.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        height = int(metadata.get('height'))
        width = int(metadata.get('width'))
        fps = int(metadata.get('fps', 30))

        print(f"\n{'='*60}")
        print(f"✓ Processing complete in {processing_time:.2f}s")
        print(f"{'='*60}\n")

        return ProcessResponse(
            status="success",
            output_path=reconstructed_path,
            total_frames=total_frames,
            duration=duration,
            resolution=f"{width}x{height}",
            fps=fps,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        state.processing = False


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a generated video file"""
    file_path = os.path.join(state.output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )


# ========================= CLI Entry Point =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='REST API server for processing FramePack latents with preloaded VAE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start server on default port with GPU
  python process_saved_latents_api.py

  # Start on custom port with specific GPU
  python process_saved_latents_api.py --port 8000 --device cuda:1

  # Start with VAE slicing enabled by default
  python process_saved_latents_api.py --enable-slicing

  # Start on CPU (for testing)
  python process_saved_latents_api.py --device cpu

Then send POST requests to:
  http://localhost:7860/process

  Example curl:
  curl -X POST "http://localhost:7860/process" \\
       -H "Content-Type: application/json" \\
       -d '{"latents_path": "outputs/job123_latents.pt", "verbose": true}'
        '''
    )
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server. Default: 0.0.0.0')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind the server. Default: 7860')
    parser.add_argument('--device', default='cuda:0', help='Device for VAE decoding (e.g., cuda:0, cuda:1, or cpu). Default: cuda:0')
    parser.add_argument('--output-dir', default='./outputs', help='Directory for output videos. Default: ./outputs')
    parser.add_argument('--enable-slicing', action='store_true', help='Enable VAE slicing by default for lower VRAM usage.')
    parser.add_argument('--disable-tiling', action='store_true', help='Disable VAE tiling (not recommended).')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes. Default: 1')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure global state
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA requested but not available! Falling back to CPU.')
        device = torch.device('cpu')

    state.device = device
    state.output_dir = args.output_dir
    state.enable_tiling = not args.disable_tiling
    state.base_enable_slicing = args.enable_slicing

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == '__main__':
    main()
