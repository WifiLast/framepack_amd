# hipBLASLt Path Fix

## Issue

If you see this error:
```
rocblaslt error: Cannot read /opt/rocm/lib/TensileLibrary_lazy_gfx1100.dat: No such file or directory
```

## Root Cause

The Tensile library files are located in `/opt/rocm/lib/rocblas/library/` but hipBLASLt by default looks in `/opt/rocm/lib/`.

## Solution ✅

The correct path has been set in demo_gradio.py (line 18):

```python
os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/rocblas/library'
```

## Verification

You can verify the Tensile libraries are in the correct location:

```bash
ls -1 /opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx*
```

**Expected output for your system (gfx1100):**
```
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1030.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1100.dat  ← Your GPU
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1101.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1102.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1151.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1200.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx1201.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx908.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx90a.dat
/opt/rocm/lib/rocblas/library/TensileLibrary_lazy_gfx942.dat
```

## Your GPU

Based on the available libraries, you have a **gfx1100** GPU (likely RX 7900 XT/XTX).

## Alternative: Manual Environment Variable

If you need to set this outside of demo_gradio.py:

```bash
export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
python demo_gradio.py
```

## Verification After Fix

When you run demo_gradio.py, you should see:

```
✓ hipBLASLt enabled for fused GEMM operations (20-40% speedup expected)
```

And **NO** error messages about missing TensileLibrary files.

## Additional Notes

- This path is specific to ROCm 6.x installations
- Different ROCm versions may have different paths
- The fix has been applied to all documentation files
- hipBLASLt will now correctly find and use the optimized kernels for your gfx1100 GPU

---

**Status:** ✅ Fixed in demo_gradio.py line 18
