import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask

CHANNELS_LAST_3D = getattr(torch, 'channels_last_3d', torch.contiguous_format)


def _ensure_channels_last_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None or tensor.dim() != 5:
        return tensor
    return tensor.contiguous(memory_format=CHANNELS_LAST_3D)


def _module_device(module: torch.nn.Module) -> torch.device:
    if hasattr(module, 'device'):
        return module.device
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')


@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images


@torch.no_grad()
def vae_decode(latents, vae, image_mode=False):
    import sys
    import os

    device = _module_device(vae)
    print(f"[VAE Decode] Starting - latents shape: {latents.shape}, device: {latents.device}", file=sys.stderr, flush=True)
    print(f"[VAE Decode] VAE device: {device}, VAE dtype: {vae.dtype}", file=sys.stderr, flush=True)

    old_find_mode = os.environ.get('MIOPEN_FIND_MODE', '')
    old_find_enforce = os.environ.get('MIOPEN_FIND_ENFORCE', '')

    print(f"[VAE Decode] Setting MIOPEN to IMMEDIATE mode to prevent Find phase hanging", file=sys.stderr, flush=True)
    os.environ['MIOPEN_FIND_MODE'] = 'NORMAL'
    os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD'] = '1'
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD'] = '1'
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW'] = '1'

    latents = _ensure_channels_last_3d(latents / vae.config.scaling_factor)
    latents = latents.to(device=device, dtype=vae.dtype)

    try:
        if not image_mode:
            print(f"[VAE Decode] Attempting GPU decode on {device}", file=sys.stderr, flush=True)
            image = vae.decode(latents).sample
        else:
            print(f"[VAE Decode] Attempting GPU image mode decode on {device}", file=sys.stderr, flush=True)
            latents_list = latents.unbind(2)
            image = [vae.decode(l.unsqueeze(2)).sample for l in latents_list]
            image = torch.cat(image, dim=2)

        image = _ensure_channels_last_3d(image)
        print(f"[VAE Decode] GPU decode SUCCESS - result device: {image.device}, shape: {image.shape}", file=sys.stderr, flush=True)
        return image
    except RuntimeError as exc:
        print(f"[VAE Decode] GPU decode FAILED with error: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if old_find_mode:
            os.environ['MIOPEN_FIND_MODE'] = old_find_mode
        if old_find_enforce:
            os.environ['MIOPEN_FIND_ENFORCE'] = old_find_enforce


@torch.no_grad()
def vae_encode(image, vae):
    import sys
    device = _module_device(vae)
    print(f"[VAE Encode] Starting - image shape: {image.shape}, device: {image.device}", file=sys.stderr, flush=True)
    print(f"[VAE Encode] VAE device: {device}, VAE dtype: {vae.dtype}", file=sys.stderr, flush=True)

    image = _ensure_channels_last_3d(image)
    image = image.to(device=device, dtype=vae.dtype)

    try:
        print(f"[VAE Encode] Attempting GPU encode on {device}", file=sys.stderr, flush=True)
        latents = vae.encode(image).latent_dist.sample()
        latents = _ensure_channels_last_3d(latents * vae.config.scaling_factor)
        print(f"[VAE Encode] GPU encode SUCCESS - latents device: {latents.device}, shape: {latents.shape}", file=sys.stderr, flush=True)
        return latents
    except RuntimeError as exc:
        print(f"[VAE Encode] GPU encode FAILED with error: {exc}", file=sys.stderr, flush=True)
        raise
