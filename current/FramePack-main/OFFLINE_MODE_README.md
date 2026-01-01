# FramePack Offline Mode Configuration

This document describes the offline mode setup for running FramePack without any internet connection.

## What Was Changed

The following modifications were made to enable completely offline operation:

### 1. Environment Variables (Lines 14-23)
Added the following environment variables at the top of `demo_gradio.py`:
```python
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
# Note: HF_HUB_OFFLINE is NOT set - it breaks tokenizer loading
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
```

**Important**: We do NOT set `HF_HUB_OFFLINE=1` because it causes the transformers library to fail when loading tokenizers due to a bug where it checks for Mistral models before respecting the `local_files_only` parameter. Instead, we rely on `local_files_only=True` in all model loading calls.

### 2. Disabled HuggingFace Login (Lines 88-90)
Commented out the HF login import to prevent authentication attempts:
```python
# from diffusers_helper.hf_login import login
print("✓ Skipping HuggingFace login (offline mode)")
```

### 3. Model Loading with `local_files_only=True`
Modified all model loading calls to use local cache only:

- **`load_model_with_fallback()` function** (Line 769): Added `local_files_only=True`
- **Tokenizers** (Lines 844-845): Added `local_files_only=True`
- **VAE** (Line 853): Added `local_files_only=True`
- **Feature Extractor** (Line 856): Added `local_files_only=True`
- **Transformer** (Line 862): Added `local_files_only=True`

### 4. Gradio Configuration (Lines 2055-2061)
Disabled Gradio sharing and API documentation:
```python
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=False,  # Disable online sharing
    inbrowser=args.inbrowser,
    show_api=False,  # Disable API documentation
)
```

## Prerequisites for Offline Mode

Before running in offline mode, you **MUST** have already downloaded all required models while online. The models should be cached in:

- `./hf_download/` - HuggingFace models cache (configured via `HF_HOME`)
- `./.cache_rocm/` - ROCm-specific caches

### Required Models to Download (Online First)

Run the script once with internet connection to download:

1. **Text Encoders**:
   - `hunyuanvideo-community/HunyuanVideo` (text_encoder subfolder)
   - `hunyuanvideo-community/HunyuanVideo` (text_encoder_2 subfolder)

2. **Image Encoder**:
   - `lllyasviel/flux_redux_bfl` (image_encoder subfolder)

3. **Tokenizers**:
   - `hunyuanvideo-community/HunyuanVideo` (tokenizer subfolder)
   - `hunyuanvideo-community/HunyuanVideo` (tokenizer_2 subfolder)

4. **VAE**:
   - `hunyuanvideo-community/HunyuanVideo` (vae subfolder)

5. **Transformer**:
   - `lllyasviel/FramePackI2V_HY`

6. **Feature Extractor**:
   - `lllyasviel/flux_redux_bfl` (feature_extractor subfolder)

## Running in Offline Mode

Once all models are cached, you can run completely offline:

```bash
# Disconnect from internet or set firewall rules
python demo_gradio.py --server 127.0.0.1 --inbrowser
```

## Verification

You can verify offline mode is working by:

1. Check the startup logs for "offline mode" messages
2. Disconnect your network adapter before running
3. Monitor network traffic - should see zero HTTP requests to:
   - `huggingface.co`
   - `gradio.app`
   - `googleapis.com`

## Troubleshooting

### Error: "OfflineModeIsEnabled: Cannot reach https://huggingface.co/..."
**Cause**: You have `HF_HUB_OFFLINE=1` set in your environment, which breaks tokenizer loading.

**Solution**:
```bash
# Remove these environment variables if set
unset HF_HUB_OFFLINE
unset HUGGINGFACE_HUB_OFFLINE
```

The code already avoids setting these variables. If you see this error, check if they're set globally in your shell profile (`.bashrc`, `.zshrc`, etc.).

### Error: "Can't load model - no internet connection"
**Solution**: You haven't downloaded the models yet. Connect to internet and run once to cache models.

### Error: "HuggingFace token required"
**Solution**: The HF login has been disabled. Make sure models were downloaded with proper authentication before going offline.

### Gradio still trying to connect online
**Solution**: Check that `GRADIO_ANALYTICS_ENABLED=False` is set before Gradio import.

### Models still downloading despite offline mode
**Solution**:
1. Verify all model loading calls have `local_files_only=True`
2. Check that models are actually cached in `./hf_download/`
3. Try running with `--verbose` to see which model is being downloaded

## Performance Environment Variables

The following HIP/ROCm performance variables are now configured (Lines 36-37):

```python
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'  # Reduce kernel launch latency by 2-3μs
os.environ['HIP_MEM_POOL_SUPPORT'] = '1'   # Improve memory allocation performance
```

These were identified as beneficial from the AMD ROCm documentation for:
- Reducing kernel launch overhead
- Improving memory pool allocation speed

## Notes

- **Transformer Engine**: If using TE, cached model conversions are stored in `.cache_rocm/te_models/`
- **Triton kernels**: Cached in `.cache_rocm/triton/`
- **Torch extensions**: Cached in `.cache_rocm/torch_extensions/`
- **TunableOp results**: Saved to `tunableop_results.csv` (if enabled)

All these caches work offline once initially populated.

## Summary

With these changes, FramePack will:
- ✅ Load all models from local cache
- ✅ Skip HuggingFace authentication
- ✅ Disable Gradio analytics and sharing
- ✅ Use only local compilation caches
- ✅ Work completely offline (air-gapped environments)

The application can now run in secure, offline, or air-gapped environments without any external network dependencies.
