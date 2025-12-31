# TransformerEngine Benchmark Guide

## Overview

The [benchmark_transformer_engine.py](benchmark_transformer_engine.py) script provides comprehensive performance comparisons between TransformerEngine and standard PyTorch operations.

## Quick Start

### Basic Benchmark
```bash
python benchmark_transformer_engine.py
```

This runs:
- Linear layer benchmarks (various sizes)
- MLP benchmarks (two linear layers + GELU)
- Matrix multiplication benchmarks
- Batch size variations
- Layer size variations

### Quick Benchmark (Faster)
```bash
python benchmark_transformer_engine.py --quick
```

Runs fewer iterations and skips extended tests (~30 seconds vs ~5 minutes).

## Command Line Options

```bash
python benchmark_transformer_engine.py [OPTIONS]

Options:
  --iterations N      Number of iterations per benchmark (default: 100)
  --warmup N         Number of warmup iterations (default: 10)
  --no-warmup        Skip warmup iterations
  --verbose, -v      Show detailed timing for each iteration
  --quick            Quick mode (20 iterations, fewer tests)
```

## Examples

### High Precision Benchmark
```bash
python benchmark_transformer_engine.py --iterations 500
```

### Debug Mode
```bash
python benchmark_transformer_engine.py --quick --verbose
```

### No Warmup (Cold Start Performance)
```bash
python benchmark_transformer_engine.py --no-warmup
```

## What It Tests

### 1. Linear Layer Performance
Tests various linear layer configurations:
- Small: 768 → 768
- Expansion: 768 → 3072 (typical MLP expansion)
- Reduction: 3072 → 768 (typical MLP projection)
- Large: 1024 → 4096

### 2. MLP Performance
Full MLP block with:
- Linear layer 1
- GELU activation
- Linear layer 2

This represents a typical transformer feedforward network.

### 3. Matrix Multiplication
Direct GEMM operations to measure raw compute performance.

### 4. Batch Size Scaling
Tests how performance scales with batch sizes: 1, 4, 8, 16, 32, 64

### 5. Layer Size Variations
Tests different hidden dimensions to find optimal use cases.

## Understanding Results

### Sample Output

```
SUMMARY
================================================================================

Benchmark                                PyTorch (ms)    TE (ms)         Speedup
--------------------------------------------------------------------------------
Linear 768->3072                        2.45            1.89            1.30x
MLP 768->3072->768                      4.82            3.67            1.31x
MatMul (2048x768)@(768x768)            0.52            0.51            1.02x
--------------------------------------------------------------------------------
OVERALL                                  7.79            6.07            1.28x

VERDICT
================================================================================
✅ TransformerEngine is 1.28x FASTER than PyTorch
   Recommendation: Use TransformerEngine for production
```

### Interpreting Speedup

- **> 1.1x**: TransformerEngine is significantly faster ✅
- **0.9x - 1.1x**: Similar performance (either is fine) ⚖️
- **< 0.9x**: PyTorch is faster (don't use TE) ⚠️

### Factors Affecting Performance

1. **GPU Architecture**
   - MI300 series: Best TE performance (native FP8)
   - MI200 series: Good TE performance
   - RX 7900 series: May show less benefit (no FP8)

2. **Operation Size**
   - Larger operations typically benefit more
   - Small operations may have overhead

3. **Batch Size**
   - Larger batches typically show better speedup
   - Small batches may not benefit

4. **hipBLASLt Status**
   - If hipBLASLt works: Better TE performance
   - If hipBLASLt fails: TE falls back to rocBLAS

## Expected Results by GPU

### RX 7900 XTX (gfx1100)
- **Without hipBLASLt**: ~0.9x - 1.1x (similar to PyTorch)
- **With hipBLASLt**: ~1.2x - 1.5x faster
- **Verdict**: TE provides marginal benefit unless hipBLASLt works

### MI200 Series (gfx90a)
- **Expected**: ~1.2x - 1.5x faster
- **With FP8**: ~1.5x - 2.0x faster
- **Verdict**: TE recommended

### MI300 Series (gfx942)
- **Expected**: ~1.5x - 2.0x faster
- **With FP8**: ~2.0x - 3.0x faster
- **Verdict**: TE highly recommended

## Troubleshooting

### "HIPBLASLT Error: 3"
TransformerEngine tried to use hipBLASLt and failed.

**Solution**: The benchmark will catch this and report TE as slower/broken.

### "Data types for parameters must match"
TE layer dtype doesn't match input dtype.

**Solution**: This is already fixed in the benchmark (line 60: `params_dtype=torch.float16`).

### "Invalid backend" in attention
Flash Attention backend selection failed.

**Solution**: The benchmark doesn't test attention (that's separate). This won't affect benchmark results.

### Benchmark is too slow
Use `--quick` mode:
```bash
python benchmark_transformer_engine.py --quick
```

### Results are inconsistent
Increase iterations for more stable averages:
```bash
python benchmark_transformer_engine.py --iterations 500
```

## Benchmark Workflow

1. **First run**: Quick benchmark to check if TE works
   ```bash
   python benchmark_transformer_engine.py --quick
   ```

2. **If TE works**: Full benchmark for accurate comparison
   ```bash
   python benchmark_transformer_engine.py
   ```

3. **If TE is faster**: Enable TE in your application
   - Set `HAS_TRANSFORMER_ENGINE = True` in demo_gradio.py

4. **If TE is slower**: Stick with PyTorch
   - Keep TE disabled (default behavior)

## Integration with demo_gradio.py

Based on benchmark results:

### If TE is Faster (>1.1x)
TransformerEngine is already automatically enabled if it works! Check startup output:
```
✓ Transformer Engine available - FP8 optimization enabled
```

### If TE is Slower (<0.9x)
Disable it by preventing import:
```bash
export DISABLE_TRANSFORMER_ENGINE=1
python demo_gradio.py
```

Or modify demo_gradio.py to skip TE import entirely.

## Advanced Usage

### Benchmark Specific Operation
Edit the script to comment out unwanted tests:

```python
# In main():
all_results = []

# Only test what you care about
all_results.append(benchmark_linear_layer(config, ...))
# all_results.append(benchmark_mlp(config, ...))  # Skip this
# all_results.append(benchmark_matmul(config, ...))  # Skip this
```

### Custom Layer Sizes
Add your specific layer sizes to `layer_configs`:

```python
layer_configs = [
    (768, 768),      # Your config here
    (2048, 2048),    # Another config
]
```

### Test Different Data Types
Modify `BenchmarkConfig`:

```python
class BenchmarkConfig:
    def __init__(self, args):
        self.dtype = torch.bfloat16  # Try bfloat16
        # Or
        self.dtype = torch.float32   # Try float32
```

## Performance Tips

1. **Close other GPU applications** before benchmarking
2. **Let GPU warm up** - first run may be slower
3. **Run multiple times** and average results
4. **Check GPU temperature** - thermal throttling affects results
5. **Monitor GPU utilization** - should be near 100% during tests

## Comparison with Other Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| benchmark_transformer_engine.py | TE vs PyTorch | Decide if TE is worth using |
| test_te_minimal.py | TE functionality test | Check if TE works at all |
| test_hipblaslt.py | hipBLASLt compatibility | Debug hipBLASLt issues |
| diagnose_rocm.py | System diagnostics | Check ROCm installation |

## Summary

The benchmark script helps you make an **informed decision** about whether TransformerEngine provides real performance benefits on your specific hardware.

**Key takeaway**: If TE is >1.1x faster, use it. Otherwise, stick with PyTorch.

Run the benchmark, check the verdict, and make your choice! 🚀
