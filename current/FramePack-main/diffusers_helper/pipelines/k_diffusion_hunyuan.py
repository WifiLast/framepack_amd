import torch
import math
import time

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


@torch.inference_mode()
def sample_hunyuan(
        transformer,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        **kwargs,
):
    # === START: Initialization Logging ===
    overall_start = time.time()
    print(f"\n[sample_hunyuan] ===== Starting Sampling =====")
    print(f"[sample_hunyuan] Configuration:")
    print(f"  - Resolution: {width}x{height}, Frames: {frames}")
    print(f"  - Steps: {num_inference_steps}, Sampler: {sampler}")
    print(f"  - CFG Scale: {real_guidance_scale}, Distilled Guidance: {distilled_guidance_scale}")
    print(f"  - Device: {device or transformer.device}, Dtype: {dtype}")

    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    # Initialize latents
    init_start = time.time()
    latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)
    init_time = time.time() - init_start
    print(f"[sample_hunyuan] Latent initialization: {init_time:.3f}s")

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4
    print(f"[sample_hunyuan] Latent shape: {latents.shape}, Sequence length: {seq_length}")

    # Calculate sigma schedule
    schedule_start = time.time()
    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
    schedule_time = time.time() - schedule_start
    print(f"[sample_hunyuan] Sigma schedule calculation: {schedule_time:.3f}s (mu={mu:.4f})")
    print(f"[sample_hunyuan] Sigma range: [{sigmas.min().item():.4f}, {sigmas.max().item():.4f}]")

    # Wrap transformer
    wrapper_start = time.time()
    k_model = fm_wrapper(transformer)
    wrapper_time = time.time() - wrapper_start
    print(f"[sample_hunyuan] Model wrapper creation: {wrapper_time:.3f}s")

    # Handle initial latent (img2img style)
    if initial_latent is not None:
        blend_start = time.time()
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma
        blend_time = time.time() - blend_start
        print(f"[sample_hunyuan] Initial latent blending: {blend_time:.3f}s (strength={strength})")

    # Handle concat latent
    if concat_latent is not None:
        concat_start = time.time()
        concat_latent = concat_latent.to(latents)
        concat_time = time.time() - concat_start
        print(f"[sample_hunyuan] Concat latent preparation: {concat_time:.3f}s")

    # Prepare guidance and embeddings
    prep_start = time.time()
    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)
    prep_time = time.time() - prep_start
    print(f"[sample_hunyuan] Embeddings/guidance preparation: {prep_time:.3f}s")

    # Build sampler arguments
    kwargs_start = time.time()
    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )
    kwargs_time = time.time() - kwargs_start
    print(f"[sample_hunyuan] Sampler kwargs construction: {kwargs_time:.3f}s")

    # === START: Main Sampling ===
    print(f"\n[sample_hunyuan] ===== Starting {sampler.upper()} Sampling =====")
    print(f"[sample_hunyuan] This will perform {num_inference_steps} denoising steps...")

    sampling_start = time.time()
    if sampler == 'unipc':
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')
    sampling_time = time.time() - sampling_start

    # === END: Completion Logging ===
    total_time = time.time() - overall_start
    print(f"\n[sample_hunyuan] ===== Sampling Complete =====")
    print(f"[sample_hunyuan] Timing breakdown:")
    print(f"  - Sampling ({num_inference_steps} steps): {sampling_time:.2f}s ({sampling_time/num_inference_steps:.2f}s/step)")
    print(f"  - Preparation overhead: {total_time - sampling_time:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"[sample_hunyuan] Output shape: {results.shape}")
    print(f"[sample_hunyuan] ==============================\n")

    return results
