#!/usr/bin/env python3

import argparse
import gc
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover - TensorRT is required at runtime
    raise SystemExit(
        'TensorRT python bindings not found. Install NVIDIA TensorRT to use this script.'
    ) from exc

from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.utils import save_bcthw_as_mp4, soft_append_bcthw

# Monkey-patch diffusers to fix device mismatch in prepare_causal_attention_mask
import diffusers.models.autoencoders.autoencoder_kl_hunyuan_video as hunyuan_vae_module

_original_prepare_causal_attention_mask = hunyuan_vae_module.prepare_causal_attention_mask

# Store a global reference to track the current device during ONNX export
_global_export_device = None

def _patched_prepare_causal_attention_mask(*args, **kwargs):
    """Patched version that ensures all tensors are on the correct device."""
    global _global_export_device

    # Handle both positional and keyword arguments
    if len(args) >= 3:
        num_frames, num_channels_latents, batch_size = args[0], args[1], args[2]
        dtype = args[3] if len(args) > 3 else kwargs.get('dtype', torch.float32)
        device = args[4] if len(args) > 4 else kwargs.get('device', None)
    else:
        num_frames = kwargs.get('num_frames', args[0] if len(args) > 0 else None)
        num_channels_latents = kwargs.get('num_channels_latents', args[1] if len(args) > 1 else None)
        batch_size = kwargs.get('batch_size', args[2] if len(args) > 2 else None)
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', None)

    # Use global export device if set (during ONNX tracing)
    if _global_export_device is not None:
        device = _global_export_device
    # Otherwise force device to be CUDA if available
    elif device is None or (isinstance(device, torch.device) and device.type == 'cpu'):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Create tensors directly on the target device
    height_width = num_channels_latents // num_frames
    indices = torch.arange(num_frames, device=device, dtype=torch.long)
    indices_blocks = indices.repeat_interleave(height_width)

    causal_mask = indices_blocks.unsqueeze(0) >= indices_blocks.unsqueeze(1)
    causal_mask = causal_mask.to(dtype=dtype, device=device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return causal_mask

hunyuan_vae_module.prepare_causal_attention_mask = _patched_prepare_causal_attention_mask


# Separate cache directories for NVIDIA CUDA to prevent CUDA/ROCm interference
_cache_base = os.path.join(os.path.dirname(__file__), '.cache_cuda')
os.makedirs(_cache_base, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = os.path.join(_cache_base, 'triton')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_cache_base, 'torch_extensions')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_cache_base, 'inductor')

CHANNELS_LAST_3D = getattr(torch, 'channels_last_3d', torch.contiguous_format)

TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
}

# Add INT64 support if available (TensorRT 10.x+)
if hasattr(trt.DataType, 'INT64'):
    TRT_TO_TORCH_DTYPE[trt.DataType.INT64] = torch.int64


@dataclass
class TensorRTProfile:
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]


def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None or tensor.dim() != 5:
        return tensor
    return tensor.contiguous(memory_format=CHANNELS_LAST_3D)


def _device_index(device: torch.device) -> int:
    if device.type != 'cuda':
        raise ValueError('TensorRT decoding requires a CUDA device.')
    return device.index or 0


def _collect_latent_shape_profile(
    segments: List[Dict[str, torch.Tensor]],
    latent_window_size: int,
    override_max_frames: Optional[int] = None,
) -> TensorRTProfile:
    if not segments:
        raise ValueError('No segments provided for TensorRT profile calculation.')

    def _frames(tensor: torch.Tensor) -> int:
        return int(tensor.shape[2])

    first_frames = _frames(segments[0]['generated_latents'])
    sliding_window_frames = latent_window_size * 2 + 1
    observed_max = max(first_frames, sliding_window_frames)
    for segment in segments:
        observed_max = max(observed_max, _frames(segment['generated_latents']))

    # Cap the maximum number of frames if requested (prevents enormous TensorRT profiles)
    if override_max_frames is not None:
        override_cap = max(int(override_max_frames), sliding_window_frames)
        observed_max = min(observed_max, override_cap)

    observed_max = max(observed_max, sliding_window_frames)

    min_frames = max(1, min(sliding_window_frames, observed_max))
    first_frames_clamped = min(first_frames, observed_max)
    opt_frames = min(observed_max, max(sliding_window_frames, first_frames_clamped))
    max_frames = observed_max

    channels = int(segments[0]['generated_latents'].shape[1])
    height = int(segments[0]['generated_latents'].shape[3])
    width = int(segments[0]['generated_latents'].shape[4])

    return TensorRTProfile(
        min_shape=(1, channels, min_frames, height, width),
        opt_shape=(1, channels, opt_frames, height, width),
        max_shape=(1, channels, max_frames, height, width),
    )


def _infer_max_frames_from_engine_name(engine_path: str) -> Optional[int]:
    match = re.search(r'_f(\d+)\.engine$', os.path.basename(engine_path))
    if match:
        return int(match.group(1))
    return None


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


class _VaeDecodeWrapper(torch.nn.Module):
    """Wrap the Diffusers VAE to capture scaling inside the graph for ONNX export."""

    def __init__(self, vae: AutoencoderKLHunyuanVideo):
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(getattr(vae.config, 'scaling_factor', 0.18215))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.vae.decode(latents / self.scaling_factor).sample
        return decoded


def export_vae_to_onnx(
    vae: AutoencoderKLHunyuanVideo,
    onnx_path: str,
    profile: TensorRTProfile,
    opset: int = 17,
) -> None:
    global _global_export_device

    print('[ONNX Export] Starting VAE decoder ONNX export process...')
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    if os.path.exists(onnx_path):
        print(f'[ONNX Export] Skipping: found existing file at {onnx_path}')
        return

    print(f'[ONNX Export] VAE device: {vae.device}, dtype: {vae.dtype}')
    print(f'[ONNX Export] Target shape - min: {profile.min_shape}, opt: {profile.opt_shape}, max: {profile.max_shape}')

    # Temporarily disable tiling and slicing during ONNX export to avoid device mismatch issues
    tiling_enabled = getattr(vae, 'use_tiling', False)
    slicing_enabled = getattr(vae, 'use_slicing', False)
    framewise_enabled = getattr(vae, 'use_framewise_decoding', False)

    print(f'[ONNX Export] VAE features - tiling: {tiling_enabled}, slicing: {slicing_enabled}, framewise: {framewise_enabled}')

    if tiling_enabled and hasattr(vae, 'disable_tiling'):
        print('[ONNX Export] Disabling VAE tiling for export...')
        vae.disable_tiling()
    if slicing_enabled and hasattr(vae, 'disable_slicing'):
        print('[ONNX Export] Disabling VAE slicing for export...')
        vae.disable_slicing()
    # Disable framewise decoding to avoid temporal tiling during export
    if hasattr(vae, 'use_framewise_decoding'):
        print('[ONNX Export] Disabling framewise decoding for export...')
        vae.use_framewise_decoding = False

    # Set global device for the patched prepare_causal_attention_mask function
    _global_export_device = vae.device
    print(f'[ONNX Export] Set global export device to: {_global_export_device}')

    try:
        print('[ONNX Export] Creating VAE wrapper...')
        wrapper = _VaeDecodeWrapper(vae).to(device=vae.device, dtype=vae.dtype)

        print('[ONNX Export] Generating example input tensor...')
        sample_shape = profile.opt_shape
        example = torch.randn(sample_shape, device=vae.device, dtype=vae.dtype)
        example = _ensure_channels_last_3d(example)
        print(f'[ONNX Export] Example shape: {example.shape}, device: {example.device}, dtype: {example.dtype}')

        print(f'[ONNX Export] Starting torch.onnx.export to {onnx_path}')
        print(f'[ONNX Export] ONNX opset version: {opset}')
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    example,
                    onnx_path,
                    input_names=['latents'],
                    output_names=['decoded'],
                    dynamic_axes={
                        'latents': {2: 'frames'},
                        'decoded': {2: 'frames'},
                    },
                    opset_version=opset,
                    do_constant_folding=True,
                )
            if os.path.exists(onnx_path):
                file_size = os.path.getsize(onnx_path) / (1024*1024)
                print(f'[ONNX Export] Export successful! File saved to: {onnx_path}')
                print(f'[ONNX Export] ONNX file size: {file_size:.2f} MB')
            else:
                raise RuntimeError(f'ONNX export completed but file was not created at {onnx_path}')
        except Exception as e:
            print(f'[ONNX Export] ERROR during export: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            raise
    finally:
        # Clear global device
        print('[ONNX Export] Clearing global export device...')
        _global_export_device = None

        # Restore original settings
        if tiling_enabled and hasattr(vae, 'enable_tiling'):
            print('[ONNX Export] Re-enabling VAE tiling...')
            vae.enable_tiling()
        if slicing_enabled and hasattr(vae, 'enable_slicing'):
            print('[ONNX Export] Re-enabling VAE slicing...')
            vae.enable_slicing()
        if hasattr(vae, 'use_framewise_decoding'):
            print(f'[ONNX Export] Restoring framewise decoding to: {framewise_enabled}')
            vae.use_framewise_decoding = framewise_enabled
        print('[ONNX Export] Cleanup complete')


def _create_timing_cache(builder: trt.Builder, cache_path: Optional[str]) -> Optional[trt.ITimingCache]:
    if not cache_path:
        return None

    # Check if timing cache is supported in this TensorRT version
    if not hasattr(builder, 'create_timing_cache'):
        print('[TensorRT] Timing cache not supported in this TensorRT version, skipping...')
        return None

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            blob = cache_file.read()
        return builder.create_timing_cache(blob)
    return builder.create_timing_cache(b'')


def _serialize_engine_blob(blob) -> bytes:
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if hasattr(blob, 'tobytes'):
        return blob.tobytes()
    return bytes(memoryview(blob))


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    profile: TensorRTProfile,
    use_fp16: bool = True,
    workspace_gb: float = 6.0,
    timing_cache_path: Optional[str] = None,
) -> None:
    print('[TensorRT] Starting TensorRT engine build process...')
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    print('[TensorRT] Creating TensorRT builder and network...')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    print(f'[TensorRT] TensorRT version: {trt.__version__}')

    print(f'[TensorRT] Parsing ONNX model from: {onnx_path}')
    with open(onnx_path, 'rb') as onnx_file:
        onnx_data = onnx_file.read()
        print(f'[TensorRT] ONNX file size: {len(onnx_data) / (1024*1024):.2f} MB')
        if not parser.parse(onnx_data):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            print(f'[TensorRT] ERROR: Failed to parse ONNX. {len(errors)} error(s):')
            for err in errors:
                print(f'[TensorRT]   - {err}')
            raise RuntimeError('Failed to parse ONNX for TensorRT engine build.')
    print(f'[TensorRT] ONNX parsing successful')
    print(f'[TensorRT] Network inputs: {network.num_inputs}, outputs: {network.num_outputs}')

    print('[TensorRT] Configuring builder...')
    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    print(f'[TensorRT] Workspace size: {workspace_gb} GB ({workspace_bytes} bytes)')
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        print('[TensorRT] Using set_memory_pool_limit for workspace')
    else:  # pragma: no cover
        config.max_workspace_size = workspace_bytes
        print('[TensorRT] Using max_workspace_size for workspace')

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('[TensorRT] FP16 mode enabled')
    else:
        print(f'[TensorRT] FP16 mode disabled (use_fp16={use_fp16}, platform_has_fast_fp16={builder.platform_has_fast_fp16})')

    print('[TensorRT] Setting up optimization profile...')
    profile_obj = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    print(f'[TensorRT] Input name: {input_name}')
    print(f'[TensorRT] Profile shapes:')
    print(f'[TensorRT]   - min: {profile.min_shape}')
    print(f'[TensorRT]   - opt: {profile.opt_shape}')
    print(f'[TensorRT]   - max: {profile.max_shape}')
    profile_obj.set_shape(input_name, profile.min_shape, profile.opt_shape, profile.max_shape)
    config.add_optimization_profile(profile_obj)

    timing_cache = _create_timing_cache(builder, timing_cache_path)
    if timing_cache is not None and hasattr(config, 'set_timing_cache'):
        config.set_timing_cache(timing_cache, False)
        print(f'[TensorRT] Timing cache loaded from: {timing_cache_path}')
    else:
        print('[TensorRT] No timing cache available')

    print('[TensorRT] Building engine... (this may take several minutes)')
    print('[TensorRT] Please wait - TensorRT is optimizing the model for your GPU')

    # Handle different TensorRT API versions
    if hasattr(builder, 'build_serialized_network'):
        # TensorRT 10.x and newer
        print('[TensorRT] Using build_serialized_network (TensorRT 10.x+)')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print('[TensorRT] ERROR: Engine build returned None')
            raise RuntimeError('TensorRT engine build returned None.')
        print('[TensorRT] Engine build successful!')
        serialized_bytes = _serialize_engine_blob(serialized)
        print(f'[TensorRT] Serialized engine size: {len(serialized_bytes) / (1024*1024):.2f} MB')
    elif hasattr(builder, 'build_engine'):
        # TensorRT 8.x and older
        print('[TensorRT] Using build_engine (TensorRT 8.x)')
        engine = builder.build_engine(network, config)
        if engine is None:
            print('[TensorRT] ERROR: Engine build returned None')
            raise RuntimeError('TensorRT engine build returned None.')
        print('[TensorRT] Engine build successful!')
        print('[TensorRT] Serializing engine...')
        serialized = engine.serialize()
        serialized_bytes = _serialize_engine_blob(serialized)
        print(f'[TensorRT] Serialized engine size: {len(serialized_bytes) / (1024*1024):.2f} MB')
    else:
        raise RuntimeError('Unsupported TensorRT version: neither build_serialized_network nor build_engine available')

    print(f'[TensorRT] Saving engine to: {engine_path}')
    with open(engine_path, 'wb') as engine_file:
        engine_file.write(serialized_bytes)
    print(f'[TensorRT] Engine saved successfully')

    if timing_cache_path and hasattr(config, 'get_timing_cache'):
        print('[TensorRT] Updating timing cache...')
        try:
            with open(timing_cache_path, 'wb') as cache_file:
                cache_file.write(config.get_timing_cache().serialize())
            print(f'[TensorRT] Timing cache saved to: {timing_cache_path}')
        except Exception as e:
            print(f'[TensorRT] Warning: Could not save timing cache: {e}')

    print('[TensorRT] TensorRT engine build complete!')


class TensorRTVaeDecoder:
    """Execute ONNX-converted VAE decoder via TensorRT."""

    def __init__(self, engine_path: str, device: torch.device, max_frames: Optional[int] = None):
        self.device = device
        self.device_index = _device_index(device)
        self.max_frames = max_frames
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as engine_file:
            serialized_engine = engine_file.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine from {engine_path}')

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Failed to create TensorRT execution context.')

        self.stream = torch.cuda.Stream(device=self.device_index)
        self._bindings = [0] * self.engine.num_bindings
        self.input_binding = None
        self.output_binding = None

        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_binding = idx
            else:
                self.output_binding = idx

        if self.input_binding is None or self.output_binding is None:
            raise RuntimeError('TensorRT engine missing expected input/output bindings.')

        self.input_dtype = TRT_TO_TORCH_DTYPE[self.engine.get_binding_dtype(self.input_binding)]
        self.output_dtype = TRT_TO_TORCH_DTYPE[self.engine.get_binding_dtype(self.output_binding)]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(self.device_index):
            if latents.device != self.device or latents.dtype != self.input_dtype:
                latents = latents.to(
                    device=self.device,
                    dtype=self.input_dtype,
                    non_blocking=True,
                )
            latents = _ensure_channels_last_3d(latents)
            latents = latents.contiguous()
            shape = tuple(int(dim) for dim in latents.shape)
            if not self.context.set_binding_shape(self.input_binding, shape):
                raise RuntimeError(f'Failed to set binding shape to {shape}')

            output_shape = tuple(int(dim) for dim in self.context.get_binding_shape(self.output_binding))
            output = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)

            self._bindings[self.input_binding] = latents.data_ptr()
            self._bindings[self.output_binding] = output.data_ptr()

            self.context.execute_async_v2(
                bindings=self._bindings,
                stream_handle=self.stream.cuda_stream,
            )
            self.stream.synchronize()

        return _ensure_channels_last_3d(output)

    def close(self) -> None:
        del self.context
        del self.engine
        del self.runtime
        del self.stream


def _decode_latents_chunked(decoder: TensorRTVaeDecoder, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents while respecting the TensorRT engine's maximum supported frame count."""
    max_frames = decoder.max_frames
    total_frames = int(latents.shape[2])

    if not max_frames or max_frames <= 0 or total_frames <= max_frames:
        return decoder.decode(latents)

    decoded_chunks: List[torch.Tensor] = []
    for start in range(0, total_frames, max_frames):
        end = min(start + max_frames, total_frames)
        chunk = latents[:, :, start:end, :, :]
        chunk = _ensure_channels_last_3d(chunk)
        decoded_chunks.append(decoder.decode(chunk))

    return torch.cat(decoded_chunks, dim=2)


def ensure_trt_engine(
    device: torch.device,
    metadata: Dict[str, int],
    segments: List[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> Tuple[str, Optional[int]]:
    print('\n' + '='*80)
    print('[TensorRT Setup] Ensuring TensorRT engine is ready...')
    print('='*80)

    cache_dir = os.path.abspath(args.trt_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f'[TensorRT Setup] Cache directory: {cache_dir}')

    latent_window_size = int(metadata.get('latent_window_size', 9))
    print(f'[TensorRT Setup] Latent window size: {latent_window_size}')

    print('[TensorRT Setup] Analyzing latent segments to determine shape profile...')
    profile = _collect_latent_shape_profile(
        segments,
        latent_window_size=latent_window_size,
        override_max_frames=args.max_trt_frames,
    )
    print(f'[TensorRT Setup] Shape profile determined:')
    print(f'[TensorRT Setup]   - min: {profile.min_shape}')
    print(f'[TensorRT Setup]   - opt: {profile.opt_shape}')
    print(f'[TensorRT Setup]   - max: {profile.max_shape}')

    height = int(metadata.get('height'))
    width = int(metadata.get('width'))
    precision = 'fp32' if args.fp32_trt else 'fp16'
    profile_tag = f'f{profile.max_shape[2]}'
    engine_name = f'hunyuan_vae_{height}x{width}_{precision}_{profile_tag}.engine'
    onnx_name = f'hunyuan_vae_{height}x{width}_{precision}_{profile_tag}.onnx'
    engine_path = os.path.join(cache_dir, engine_name)
    onnx_path = os.path.join(cache_dir, onnx_name)
    timing_cache_path = os.path.join(cache_dir, 'vae_timing.cache')

    print(f'[TensorRT Setup] Configuration:')
    print(f'[TensorRT Setup]   - Resolution: {width}x{height}')
    print(f'[TensorRT Setup]   - Precision: {precision}')
    print(f'[TensorRT Setup]   - Max frames: {profile.max_shape[2]}')
    print(f'[TensorRT Setup]   - Engine file: {engine_name}')
    print(f'[TensorRT Setup]   - ONNX file: {onnx_name}')

    if os.path.exists(engine_path) and not args.force_trt_build:
        print(f'[TensorRT Setup] Found cached TensorRT engine: {engine_path}')
        cached_frames = _infer_max_frames_from_engine_name(engine_path)
        if cached_frames:
            print(f'[TensorRT Setup] Cached engine max frames (inferred): {cached_frames}')
        else:
            print('[TensorRT Setup] Could not infer max frames from cached engine filename. Assuming unlimited.')
        print('[TensorRT Setup] Skipping build, using cached engine')
        print('='*80 + '\n')
        return engine_path, cached_frames

    print('[TensorRT Setup] No cached engine found, building new engine...')
    print(f'[TensorRT Setup] Force rebuild: {args.force_trt_build}')

    # Check if pre-generated ONNX model exists from test_onnx_export.py
    pregenerated_onnx = './test_onnx_output/vae_decoder_test.onnx'

    if not os.path.exists(onnx_path):
        # TensorRT 10.0.1 supports modern ONNX operators, use pre-generated ONNX if available
        use_pregenerated = os.path.exists(pregenerated_onnx)

        if use_pregenerated:
            print(f'[TensorRT Setup] Found pre-generated ONNX model at: {pregenerated_onnx}')
            print(f'[TensorRT Setup] Copying to: {onnx_path}')
            import shutil
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            shutil.copy2(pregenerated_onnx, onnx_path)

            # Also copy external data file if it exists
            pregenerated_data = pregenerated_onnx + '.data'
            if os.path.exists(pregenerated_data):
                onnx_data_path = onnx_path + '.data'
                print(f'[TensorRT Setup] Copying external data file to: {onnx_data_path}')
                shutil.copy2(pregenerated_data, onnx_data_path)

            print('[TensorRT Setup] ONNX model copied successfully')
        else:
            # Fallback to generating ONNX if pre-generated doesn't exist
            print('[TensorRT Setup] No pre-generated ONNX found, loading VAE for export...')
            vae = load_vae(
                device=device,
                enable_tiling=False,
                enable_slicing=args.enable_slicing,
            )
            try:
                print('\n[TensorRT Setup] Step 1/2: Converting VAE to ONNX...')
                export_vae_to_onnx(vae, onnx_path, profile=profile)
            finally:
                print('[TensorRT Setup] Cleaning up VAE model from memory...')
                del vae
                torch.cuda.empty_cache()
                print('[TensorRT Setup] Cleanup complete')
    else:
        print(f'[TensorRT Setup] Using existing ONNX model: {onnx_path}')

    print('\n[TensorRT Setup] Step 2/2: Building TensorRT engine from ONNX...')
    build_trt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        profile=profile,
        use_fp16=not args.fp32_trt,
        workspace_gb=args.trt_workspace_gb,
        timing_cache_path=timing_cache_path,
    )

    print('\n' + '='*80)
    print('[TensorRT Setup] TensorRT engine ready!')
    print('='*80 + '\n')
    return engine_path, profile.max_shape[2]


@torch.no_grad()
def reconstruct_video_from_segments(
    segments: List[Dict[str, torch.Tensor]],
    metadata: Dict[str, int],
    decoder: TensorRTVaeDecoder,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Reconstruct video from latent segments using TensorRT-accelerated VAE decoding.
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

    target_dtype = torch.float16
    device = decoder.device
    latent_channels = int(segments[0]['generated_latents'].shape[1])
    latent_height = int(segments[0]['generated_latents'].shape[3])
    latent_width = int(segments[0]['generated_latents'].shape[4])

    history_latents = torch.zeros(
        size=(1, latent_channels, 1 + 2 + 16, latent_height, latent_width),
        dtype=target_dtype,
        device=device,
    )
    history_latents = _ensure_channels_last_3d(history_latents)

    history_pixels = None
    total_generated_latent_frames = 0
    overlap_frames = latent_window_size * 4 - 3

    for idx, segment in enumerate(segments, 1):
        if verbose:
            print(f'Processing segment {idx}/{total_segments}...')

        generated_latents = segment['generated_latents'].to(device=device, dtype=target_dtype)
        generated_latents = _ensure_channels_last_3d(generated_latents)
        segment_latent_frames = int(generated_latents.shape[2])

        history_latents = torch.cat([generated_latents, history_latents], dim=2)
        total_generated_latent_frames += segment_latent_frames
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        real_history_latents = _ensure_channels_last_3d(real_history_latents)

        if history_pixels is None:
            if verbose:
                print(f'  Decoding initial segment (latent frames: {real_history_latents.shape[2]})...')
            decoded = _decode_latents_chunked(decoder, real_history_latents)
            history_pixels = _ensure_channels_last_3d(decoded).cpu()
            if verbose:
                print(f'  Initial decode complete, pixel frames: {history_pixels.shape[2]}')
        else:
            is_last_section = bool(segment.get('is_last_section'))
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            current_latents = real_history_latents[:, :, :section_latent_frames, :, :]
            if verbose:
                print(f'  Decoding segment (latent frames: {current_latents.shape[2]}, last={is_last_section})...')
            current_pixels = _decode_latents_chunked(decoder, current_latents)
            current_pixels = _ensure_channels_last_3d(current_pixels).cpu()
            if verbose:
                print(f'  Before append: history_pixels={history_pixels.shape[2]} frames, current_pixels={current_pixels.shape[2]} frames')
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap_frames)
            history_pixels = _ensure_channels_last_3d(history_pixels)
            if verbose:
                print(f'  After append: history_pixels={history_pixels.shape[2]} frames')

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
        description='Resume video generation from saved FramePack latents using NVIDIA TensorRT-accelerated VAE decoding.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process with TensorRT using default GPU
  python process_saved_latents_tensorrt.py --latents outputs/20250101_123456_latents.pt

  # Force rebuild of the TensorRT engine and store it under a custom directory
  python process_saved_latents_tensorrt.py --latents my_latents.pt --force-trt-build --trt-cache-dir ./trt_models

  # Use FP32 TensorRT engine (for older GPUs) and custom workspace size
  python process_saved_latents_tensorrt.py --latents my_latents.pt --fp32-trt --trt-workspace-gb 12
        ''',
    )
    parser.add_argument('--latents', required=True, help='Path to the saved *_latents.pt file.')
    parser.add_argument('--device', default='cuda:0', help='CUDA device for TensorRT inference (e.g., cuda:0).')
    parser.add_argument('--output-dir', default='./outputs', help='Directory for the reconstructed video output. Default: ./outputs')
    parser.add_argument('--output', default=None, help='Optional filename for the output video (defaults to <job_id>_resume.mp4).')
    parser.add_argument('--enable-slicing', action='store_true', help='Enable VAE slicing for lower VRAM usage during export.')
    parser.add_argument('--disable-tiling', action='store_true', help='Disable VAE tiling during export (may increase VRAM usage).')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    parser.add_argument('--trt-cache-dir', default='./trt_cache', help='Directory for ONNX files, TensorRT engines, and timing caches.')
    parser.add_argument('--force-trt-build', action='store_true', help='Force rebuilding the TensorRT engine even if one already exists.')
    parser.add_argument('--fp32-trt', action='store_true', help='Force TensorRT engine to run in FP32 (default uses FP16 when supported).')
    parser.add_argument('--max-trt-frames', type=int, default=96, help='Upper bound for latent frames inside the TensorRT engine profile (defaults to 96). Larger clips are decoded in chunks automatically.')
    parser.add_argument('--trt-workspace-gb', type=float, default=8.0, help='Workspace size (GB) to allocate when building the TensorRT engine.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print('\n' + '='*80)
    print('FramePack - TensorRT Video Reconstruction')
    print('='*80)

    if not torch.cuda.is_available():
        raise SystemExit('ERROR: CUDA is required for TensorRT decoding but no CUDA device was found.')

    print(f'\n[Main] Loading latent file: {args.latents}')
    latents_pkg = torch.load(args.latents, map_location='cpu')
    metadata = latents_pkg.get('metadata', {})
    segments = latents_pkg.get('segments', [])

    if not segments:
        print('[Main] ERROR: No segments found in latent file!')
        return

    version = metadata.get('version', 0)
    print(f'[Main] Latent file version: {version}')
    print(f'[Main] Number of segments: {len(segments)}')

    # Display metadata
    print(f'[Main] Metadata:')
    for key, value in metadata.items():
        print(f'[Main]   - {key}: {value}')

    os.makedirs(args.output_dir, exist_ok=True)
    job_id = metadata.get('job_id', 'framepack')
    output_name = args.output or f'{job_id}_resume.mp4'
    output_path = os.path.join(args.output_dir, output_name)
    print(f'[Main] Output path: {output_path}')

    device = torch.device(args.device)
    device_index = _device_index(device)
    torch.cuda.set_device(device_index)
    print(f'[Main] Using CUDA device: {device} (index: {device_index})')

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_memory = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        print(f'[Main] GPU: {gpu_name}')
        print(f'[Main] GPU Memory: {gpu_memory:.2f} GB')

    print('\n[Main] Initializing TensorRT engine...')
    engine_path, max_frames = ensure_trt_engine(device, metadata, segments, args)

    print('[Main] Loading TensorRT decoder...')
    decoder = TensorRTVaeDecoder(engine_path, device=device, max_frames=max_frames)
    print('[Main] TensorRT decoder ready')

    try:
        print('\n[Main] Starting video reconstruction...')
        reconstructed_path = reconstruct_video_from_segments(
            segments, metadata, decoder, output_path, verbose=not args.quiet
        )
        print(f'\n' + '='*80)
        print(f'SUCCESS: Video reconstructed!')
        print(f'Output: {reconstructed_path}')
        print('='*80 + '\n')
    finally:
        print('[Main] Cleaning up TensorRT decoder...')
        decoder.close()

        print('[Main] Cleaning up memory...')
        del segments
        del latents_pkg
        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('[Main] GPU memory cleared')
        print('[Main] Done!')


if __name__ == '__main__':
    main()
