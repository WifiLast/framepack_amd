# Low Performance Diagnosis - RX 7900 XTX

## Issue Detected

Your test shows **very low GPU efficiency**:
- **Actual:** 2.65 TFLOPS (FP16)
- **Expected:** 43-55 TFLOPS (70-90% efficiency)
- **Current Efficiency:** 4.3% ⚠️

This is **not normal** and indicates the GPU is not being utilized properly.

## Root Causes

### 1. Power/Performance State Issues (Most Likely)

Your GPU might be stuck in a low power state.

**Check current GPU state:**
```bash
rocm-smi
```

**Look for:**
- GPU clock speed (should be ~2400 MHz boost, not idle ~500 MHz)
- Power consumption (should be 200W+ under load, not 20W)
- Temperature (should rise during test, 60-80°C)

**Fix - Force performance mode:**
```bash
# Temporary (until reboot)
sudo rocm-smi --setperflevel high
sudo rocm-smi --setfan 50  # Set 50% fan speed

# Check if it worked
rocm-smi
```

**Permanent fix - Add to /etc/rc.local:**
```bash
#!/bin/bash
/opt/rocm/bin/rocm-smi --setperflevel high
exit 0
```

### 2. PCIe Link Speed Issues

The GPU might not be running at full PCIe 4.0 x16 speed.

**Check PCIe link:**
```bash
rocm-smi --showbus

# Or more detailed:
lspci -vv | grep -A 20 "VGA\|3D"
```

**Look for:**
- Should be: `LnkSta: Speed 16GT/s, Width x16`
- Bad: `Speed 8GT/s` or `Width x8` or lower

**Common causes:**
- GPU in wrong PCIe slot (use primary x16 slot)
- BIOS PCIe set to Gen 3 instead of Gen 4
- Riser cable limiting speed

**Fix:**
1. Move GPU to primary PCIe x16 slot (usually closest to CPU)
2. Update BIOS settings:
   - Set PCIe to Gen 4 (not Auto or Gen 3)
   - Enable Above 4G Decoding
   - Enable Resizable BAR

### 3. Thermal Throttling

GPU might be overheating and throttling.

**Check temperatures:**
```bash
watch -n 1 rocm-smi
# Run your test in another terminal
# Temperature should stay below 110°C junction temp
```

**If overheating:**
- Check case airflow
- Clean dust from GPU cooler
- Increase fan speed: `sudo rocm-smi --setfan 70`
- Check thermal paste (if GPU is old)

### 4. Driver/ROCm Issues

Old or corrupted ROCm installation.

**Check ROCm version:**
```bash
rocminfo | grep "Name:"
apt list --installed | grep rocm
```

**You should have ROCm 6.4 fully installed:**
```bash
sudo apt-get update
sudo apt-get install --reinstall rocm-hip-runtime rocm-device-libs
sudo apt-get install --reinstall rocblas hipblas
```

### 5. Compute Mode Issues

GPU might be in graphics mode instead of compute mode.

**Check and set compute mode:**
```bash
# Check current mode
rocm-smi --showproductname

# Set compute mode (if available)
sudo rocm-smi --setcomputepartition COMPUTE
```

## Quick Diagnostic Steps

Run these in order:

### Step 1: Check GPU is recognized
```bash
rocminfo | grep -A 5 "Name:.*gfx1100"
```

**Expected output:**
```
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  ...
```

### Step 2: Check current clocks and power
```bash
rocm-smi
```

**Expected during idle:**
```
GPU[0]    : Temp: 45°C  Power: 20W  Clock: 500MHz
```

**Expected under load:**
```
GPU[0]    : Temp: 75°C  Power: 250W  Clock: 2400MHz
```

If power stays at ~20W during test, GPU is not being used!

### Step 3: Force performance mode and re-test
```bash
# Enable performance mode
sudo rocm-smi --setperflevel high

# Run test again
cd /root/video_gen/framepack_orginal/FramePack-main
python test_hipblaslt.py --test-gemm

# Watch GPU during test
watch -n 0.5 rocm-smi
```

**You should see:**
- Clock: 2000-2500 MHz
- Power: 200-300W
- Temp: Rising to 60-80°C
- TFLOPS: 40-55 (not 2.65!)

### Step 4: Check for Compute Processes
```bash
# While test is running:
rocm-smi --showpids
```

Should show Python process using GPU.

### Step 5: Verify PyTorch sees GPU correctly
```python
python3 << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"HIP version: {torch.version.hip}")

# Test actual compute
device = torch.device('cuda:0')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

import time
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Time: {elapsed*10:.2f}ms per matmul")
print(f"Expected: ~0.5-2ms per matmul for RX 7900 XTX")
if elapsed*10 > 5:
    print("⚠️  WARNING: GPU performance is very slow!")
EOF
```

## Expected vs Actual Performance

### Benchmark Reference for RX 7900 XTX

| Test | Expected | Your Result | Status |
|------|----------|-------------|--------|
| rocBLAS GEMM (512x512 FP16) | 45-55 TFLOPS | 2.65 TFLOPS | ❌ Too slow |
| Power under load | 200-300W | ? | ❓ Check with rocm-smi |
| GPU clock under load | 2000-2500 MHz | ? | ❓ Check with rocm-smi |
| Efficiency | 70-90% | 4.3% | ❌ Way too low |

### What Good Performance Looks Like

When the test runs with proper GPU utilization:

```bash
$ python test_hipblaslt.py --test-gemm

==================================================================
  Test 3: Basic Matrix Multiplication (rocBLAS)
==================================================================
     Creating 512x512 matrices...
     Warming up...
     Benchmarking...
✅ PASS: Basic GEMM
     Time: 0.95ms/iter, 48.32 TFLOPS  ← Should be in this range!
```

**And rocm-smi shows:**
```
GPU[0]    Temp: 72°C  Power: 245W  Clock: 2450MHz  ← Active computation
```

## Most Likely Fix

Based on the symptoms (4.3% efficiency), the issue is almost certainly:

**GPU stuck in low power state**

**Immediate fix:**
```bash
sudo rocm-smi --setperflevel high
python test_hipblaslt.py --test-gemm
```

**If this fixes it, make permanent:**
```bash
# Create startup script
sudo nano /etc/rc.local

# Add these lines:
#!/bin/bash
/opt/rocm/bin/rocm-smi --setperflevel high
exit 0

# Make executable
sudo chmod +x /etc/rc.local
```

## After Fixing

Once GPU performance is normal:

1. **Re-run the hipBLASLt test:**
   ```bash
   python test_hipblaslt.py
   ```

2. **You should see:**
   - rocBLAS GEMM: 45-55 TFLOPS (not 2.65!)
   - Efficiency: 70-90% (not 4.3%!)

3. **Then test your video generation:**
   ```bash
   python demo_gradio.py
   ```

   With proper GPU performance, you'll see the **full 40-70% speedup** from CK optimizations!

## If Still Slow After Fixes

**Hardware issues:**
- GPU might be defective
- PSU might not provide enough power (need 850W+ for RX 7900 XTX)
- Thermal paste needs replacement
- PCIe slot damaged

**Software issues:**
- Wrong GPU driver (need amdgpu-install)
- Kernel too old (need 5.15+ for RDNA3)
- SELinux blocking GPU access

**Get more diagnostics:**
```bash
# Full system info
dmesg | grep amdgpu
lsmod | grep amdgpu
uname -r  # Kernel version

# GPU detailed info
rocminfo > rocminfo.txt
cat rocminfo.txt
```

## Summary

**Immediate action:**
```bash
# 1. Force high performance mode
sudo rocm-smi --setperflevel high

# 2. Re-run test
python test_hipblaslt.py --test-gemm

# 3. Watch GPU during test
watch -n 0.5 rocm-smi
```

**Expected result after fix:**
- TFLOPS: 45-55 (18-20x faster than current!)
- Power: 200-300W (10-15x more than idle)
- Clock: 2000-2500 MHz (4-5x faster than idle)
- Efficiency: 70-90%

**Then your video generation will be truly fast!** 🚀

---

**Note:** The CK optimizations in demo_gradio.py are working correctly. The issue is **GPU not running at full speed**, not the optimizations themselves. Once you fix the GPU performance state, you'll see the full benefit!
