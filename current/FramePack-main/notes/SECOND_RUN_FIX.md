# Fix for Second Run Issues in demo_gradio.py

## Problem Description
The demo_gradio.py script cannot run a second generation after the first one completes.

## Root Causes Identified

### 1. Stream State Management Issue
**Problem**: When starting a second generation, a new `AsyncStream()` is created, but the old worker thread might still be running and referencing the old stream.

**Fix Applied**:
- Added cleanup code to signal the old stream to end before creating a new one
- Added 0.5 second delay to allow old worker thread to finish
- Added `GeneratorExit` exception handling to cleanup when Gradio cancels the generator

### 2. GPU Memory Not Being Released
**Problem**: GPU memory from the first run might not be fully released, causing OOM or allocation failures on the second run.

**Fix Applied**:
- Added explicit `torch.cuda.empty_cache()` call before starting new generation
- Added ROCm memory pool trim (`flush_rocm_allocator`) before new generation
- This ensures maximum memory is available for the second run

### 3. Potential Remaining Issues

If the second run still doesn't work, the issue might be:

#### A. Torch Compile Cache Issues (if USE_TORCH_COMPILE=1)
**Symptom**: Second run hangs or crashes during model forward passes
**Solution**: Disable torch.compile temporarily to test:
```bash
export FRAMEPACK_USE_TORCH_COMPILE=0
python demo_gradio.py
```

#### B. MIOpen Find Database Corruption
**Symptom**: Second run hangs during convolution operations (especially VAE)
**Solution**: Clear MIOpen cache before each run:
```bash
rm -rf ~/.config/miopen/
```

Or add to the script before the second run.

#### C. Gradio Queue Not Properly Resetting
**Symptom**: Second generation button doesn't respond
**Solution**: This requires restarting the Gradio server. The current fix should help, but if not, you may need to refresh the web page.

#### D. Thread Listener Not Cleaning Up Tasks
**Symptom**: Old generation continues running in background
**Investigation needed**: The `Listener` class in `thread_utils.py` uses a daemon thread that never cleans up its task queue. Old tasks might still be queued.

## Testing the Fix

1. Start the server:
```bash
python demo_gradio.py
```

2. Run a first generation with a small video length (1-2 seconds)

3. Wait for it to complete fully

4. Try to run a second generation

## If Issues Persist

### Enable Debug Logging
Add this at the start of `process()` function:
```python
print(f"[DEBUG] Starting generation, current stream: {id(stream)}")
```

And in `worker()` function:
```python
print(f"[DEBUG] Worker started, stream id: {id(stream)}")
```

This will help identify if multiple workers are running simultaneously.

### Check for Memory Leaks
Monitor GPU memory between runs:
```bash
watch -n 1 rocm-smi
# or
watch -n 1 nvidia-smi
```

Memory should drop back to baseline after first run completes.

### Check Thread Count
```python
import threading
print(f"Active threads: {threading.active_count()}")
```

Add this before and after each generation to see if threads are accumulating.

## Changes Made to demo_gradio.py

1. **Line 365**: Changed `stream = AsyncStream()` to `stream = None`
2. **Lines 700-713**: Added cleanup code before starting new generation
3. **Lines 723-747**: Wrapped main loop in try/except to handle `GeneratorExit`
4. **Line 742**: Added null check to `end_process()`

## Additional Recommendations

If the issue persists, consider:

1. **Reset models between runs**: Add code to move models back to CPU and clear their internal states
2. **Clear torch compile cache**: Add `torch._dynamo.reset()` if using torch.compile
3. **Restart Gradio server**: Add a "Restart Server" button that clears all state
4. **Limit concurrent generations**: Use a lock to prevent multiple generations from overlapping
