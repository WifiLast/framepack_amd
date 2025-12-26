import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask


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
    import signal
    from contextlib import contextmanager

    print(f"[VAE Decode] Starting - latents shape: {latents.shape}, device: {latents.device}", file=sys.stderr, flush=True)
    print(f"[VAE Decode] VAE device: {vae.device}, VAE dtype: {vae.dtype}", file=sys.stderr, flush=True)

    # Force MIOpen to use IMMEDIATE mode for VAE (prevents hanging in Find phase)
    old_find_mode = os.environ.get('MIOPEN_FIND_MODE', '')
    old_find_enforce = os.environ.get('MIOPEN_FIND_ENFORCE', '')

    print(f"[VAE Decode] Setting MIOpen to IMMEDIATE mode to prevent Find phase hanging", file=sys.stderr, flush=True)
    os.environ['MIOPEN_FIND_MODE'] = 'NORMAL'  # Use NORMAL but with immediate fallback
    os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'  # Don't enforce Find, use immediate if Find fails

    # Also ensure naive convolution is still available
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD'] = '1'
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD'] = '1'
    os.environ['MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW'] = '1'

    latents = latents / vae.config.scaling_factor

    try:
        # Try GPU decode first
        if not image_mode:
            print(f"[VAE Decode] Attempting GPU decode on {vae.device}", file=sys.stderr, flush=True)
            torch.cuda.synchronize()  # Ensure everything is ready before decode
            image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
            torch.cuda.synchronize()  # Ensure decode completed
            print(f"[VAE Decode] GPU decode SUCCESS - result device: {image.device}, shape: {image.shape}", file=sys.stderr, flush=True)
        else:
            print(f"[VAE Decode] Attempting GPU image mode decode on {vae.device}", file=sys.stderr, flush=True)
            latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
            image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
            image = torch.cat(image, dim=2)
            torch.cuda.synchronize()  # Ensure decode completed
            print(f"[VAE Decode] GPU image mode decode SUCCESS - result device: {image.device}, shape: {image.shape}", file=sys.stderr, flush=True)

        # Restore original MIOpen settings
        if old_find_mode:
            os.environ['MIOPEN_FIND_MODE'] = old_find_mode
        if old_find_enforce:
            os.environ['MIOPEN_FIND_ENFORCE'] = old_find_enforce

        return image
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[VAE Decode] GPU decode FAILED with error: {error_msg}", file=sys.stderr, flush=True)

        # Fallback to CPU for MIOpen errors (ZLUDA 3D convolution issues)
        if "miopenStatus" in error_msg or "MIOpen" in error_msg or "convolution" in error_msg.lower():
            print(f"[VAE Decode] Detected MIOpen/convolution error - falling back to CPU", file=sys.stderr, flush=True)

            # Move VAE to CPU
            original_device = vae.device
            print(f"[VAE Decode] Moving VAE from {original_device} to CPU...", file=sys.stderr, flush=True)
            vae.to('cpu')
            print(f"[VAE Decode] VAE moved to CPU successfully", file=sys.stderr, flush=True)

            # Decode on CPU
            if not image_mode:
                print(f"[VAE Decode] Running CPU decode...", file=sys.stderr, flush=True)
                image = vae.decode(latents.to(device='cpu', dtype=vae.dtype)).sample
                print(f"[VAE Decode] CPU decode SUCCESS - result shape: {image.shape}", file=sys.stderr, flush=True)
            else:
                print(f"[VAE Decode] Running CPU image mode decode...", file=sys.stderr, flush=True)
                latents = latents.to(device='cpu', dtype=vae.dtype).unbind(2)
                image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
                image = torch.cat(image, dim=2)
                print(f"[VAE Decode] CPU image mode decode SUCCESS - result shape: {image.shape}", file=sys.stderr, flush=True)

            # Keep VAE on CPU for future operations (don't move back to avoid repeated failures)
            print(f"[VAE Decode] Keeping VAE on CPU for future operations", file=sys.stderr, flush=True)
            return image
        else:
            # Re-raise if it's a different error
            print(f"[VAE Decode] Non-MIOpen error - re-raising exception", file=sys.stderr, flush=True)
            raise


@torch.no_grad()
def vae_encode(image, vae):
    import sys
    print(f"[VAE Encode] Starting - image shape: {image.shape}, device: {image.device}", file=sys.stderr, flush=True)
    print(f"[VAE Encode] VAE device: {vae.device}, VAE dtype: {vae.dtype}", file=sys.stderr, flush=True)

    try:
        # Try GPU encode first
        print(f"[VAE Encode] Attempting GPU encode on {vae.device}", file=sys.stderr, flush=True)
        latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        print(f"[VAE Encode] GPU encode SUCCESS - latents device: {latents.device}, shape: {latents.shape}", file=sys.stderr, flush=True)
        return latents
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[VAE Encode] GPU encode FAILED with error: {error_msg}", file=sys.stderr, flush=True)

        # Fallback to CPU for MIOpen errors (ZLUDA 3D convolution issues)
        if "miopenStatus" in error_msg or "MIOpen" in error_msg or "convolution" in error_msg.lower():
            print(f"[VAE Encode] Detected MIOpen/convolution error - falling back to CPU", file=sys.stderr, flush=True)

            # Move VAE to CPU
            original_device = vae.device
            print(f"[VAE Encode] Moving VAE from {original_device} to CPU...", file=sys.stderr, flush=True)
            vae.to('cpu')
            print(f"[VAE Encode] VAE moved to CPU successfully", file=sys.stderr, flush=True)

            # Encode on CPU
            print(f"[VAE Encode] Running CPU encode...", file=sys.stderr, flush=True)
            latents = vae.encode(image.to(device='cpu', dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            print(f"[VAE Encode] CPU encode SUCCESS - latents shape: {latents.shape}", file=sys.stderr, flush=True)

            # Keep VAE on CPU for future operations
            print(f"[VAE Encode] Keeping VAE on CPU for future operations", file=sys.stderr, flush=True)
            return latents
        else:
            # Re-raise if it's a different error
            print(f"[VAE Encode] Non-MIOpen error - re-raising exception", file=sys.stderr, flush=True)
            raise
