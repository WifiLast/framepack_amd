# TransformerEngine Benchmark - Quick Start Guide

## Overview

This benchmark suite comprehensively compares TransformerEngine performance against standard PyTorch for transformer operations. It includes **AMD/ROCm compatibility fixes** to handle platform-specific limitations gracefully.

## Quick Start

### Basic Benchmark (Default)
```bash
python benchmark_transformer_engine.py
```

**Runs**: Linear, MLP, MatMul, LayerNorm, RMSNorm, Attention, Backward pass
**Duration**: ~2-3 minutes
**Iterations**: 100 per test

### Quick Test
```bash
python benchmark_transformer_engine.py --quick
```

**Runs**: Same as default but faster
**Duration**: ~30 seconds
**Iterations**: 20 per test

### Comprehensive Benchmark
```bash
python benchmark_transformer_engine.py --comprehensive
```

**Runs**: All tests including scaling studies and memory benchmarks
**Duration**: ~15-20 minutes
**Includes**: Batch size scaling, layer size scaling, sequence length scaling, memory usage

## Run Specific Tests

### Individual Component Tests
```bash
# Test only linear layers
python benchmark_transformer_engine.py --test linear

# Test only attention mechanism
python benchmark_transformer_engine.py --test attention

# Test only LayerNorm
python benchmark_transformer_engine.py --test layernorm

# Test only RMSNorm
python benchmark_transformer_engine.py --test rmsnorm

# Test backward pass
python benchmark_transformer_engine.py --test backward

# Test full transformer block
python benchmark_transformer_engine.py --test transformer

# Test training step
python benchmark_transformer_engine.py --test training
```

### Scaling Studies
```bash
# Test different batch sizes
python benchmark_transformer_engine.py --test batch_sizes

# Test different layer sizes
python benchmark_transformer_engine.py --test layer_sizes

# Test different sequence lengths
python benchmark_transformer_engine.py --test seq_lengths

# Test memory usage
python benchmark_transformer_engine.py --test memory
```

## Advanced Options

### Custom Iterations
```bash
# More iterations for more accurate results
python benchmark_transformer_engine.py --iterations 200

# Fewer iterations for quick validation
python benchmark_transformer_engine.py --iterations 50
```

### Warmup Control
```bash
# Custom warmup iterations
python benchmark_transformer_engine.py --warmup 20

# Skip warmup entirely
python benchmark_transformer_engine.py --no-warmup
```

### Verbose Output
```bash
# Show detailed timing for each iteration
python benchmark_transformer_engine.py --verbose
```

### Combined Options
```bash
# Quick test with verbose output
python benchmark_transformer_engine.py --quick --verbose

# Specific test with custom iterations
python benchmark_transformer_engine.py --test attention --iterations 500

# Comprehensive test with more iterations
python benchmark_transformer_engine.py --comprehensive --iterations 200
```

## Understanding the Output

### Per-Test Results
```
======================================================================
Benchmark: LayerNorm (hidden_size=768)
  Input shape: (16, 128, 768)
======================================================================
  PyTorch LayerNorm: 0.45ms/iter (100 iterations)
  TE LayerNorm: 0.38ms/iter (100 iterations)
LayerNorm (d=768):
  PyTorch:  0.45ms
  TE:       0.38ms
  Speedup:  1.18x
```

### Summary Table
```
Benchmark                                PyTorch (ms)    TE (ms)         Speedup
--------------------------------------------------------------------------------
Linear 768->3072                         2.15            1.89            1.14x
MLP 768->3072->768                       4.32            3.91            1.10x
MatMul (2048x768)@(768x768)             1.23            1.23            1.00x
LayerNorm (d=768)                        0.45            0.38            1.18x
RMSNorm (d=768)                          0.41            0.35            1.17x
Attention (h=12, d=768)                  5.67            5.67            1.00x
Fwd+Bwd Linear 768->3072                 6.43            6.43            1.00x
--------------------------------------------------------------------------------
OVERALL                                  20.66           19.86           1.04x
```

### Verdict
```
======================================================================
VERDICT
======================================================================
✅ TransformerEngine is 1.04x FASTER than PyTorch
   Recommendation: Use TransformerEngine for production
```

## AMD/ROCm Compatibility

### What You'll See on AMD

When running on AMD hardware, you may see warning messages:

```
⚠️  TE Attention not supported on this platform: fused attn configs not supported
  Using PyTorch implementation for comparison

⚠️  TE Backward pass not supported on this platform: Unable to find any suitable
  Using PyTorch implementation for comparison
```

**This is normal!** The benchmark automatically falls back to PyTorch when TE features aren't supported.

### Expected Behavior on AMD

- ✅ **Linear/LayerNorm/RMSNorm** will show actual TE performance (forward pass only)
- ⚠️ **Attention** will show PyTorch performance (TE attention not supported)
- ⚠️ **Backward/Training** will show PyTorch performance (TE gradients not supported)
- ⚠️ **Transformer Block** will use hybrid: TE Linear/LayerNorm + PyTorch Attention

### Why Some Tests Show 1.00x Speedup

When you see `1.00x` speedup, it means:
- TransformerEngine feature is not available on this platform
- Benchmark fell back to PyTorch for both measurements
- Both times are identical because they're using the same implementation

## Test Categories

### Available Tests (`--test` option)

| Test Name | Description |
|-----------|-------------|
| `linear` | Single linear layer |
| `mlp` | 2-layer MLP with GELU |
| `matmul` | Matrix multiplication |
| `layernorm` | LayerNorm normalization |
| `rmsnorm` | RMSNorm normalization |
| `attention` | Multi-head attention |
| `dropout` | Dropout regularization |
| `backward` | Forward + backward pass |
| `activations` | GELU, ReLU, SiLU, Tanh |
| `transformer` | Complete transformer block |
| `training` | Full training step with optimizer |
| `memory` | GPU memory usage |
| `batch_sizes` | Scaling with batch size |
| `layer_sizes` | Scaling with layer dimensions |
| `seq_lengths` | Scaling with sequence length |

## Common Use Cases

### 1. Quick Validation
```bash
python benchmark_transformer_engine.py --quick
```
Use when: Quickly checking if TE is working

### 2. Detailed Performance Analysis
```bash
python benchmark_transformer_engine.py --comprehensive --iterations 200
```
Use when: Making production decisions about which library to use

### 3. Debug Specific Operation
```bash
python benchmark_transformer_engine.py --test attention --verbose
```
Use when: Investigating performance of a specific component

### 4. Memory Profiling
```bash
python benchmark_transformer_engine.py --test memory
```
Use when: Checking GPU memory requirements

### 5. Scaling Study
```bash
python benchmark_transformer_engine.py --test seq_lengths --iterations 50
```
Use when: Understanding how performance changes with input size

## Interpreting Results

### Performance Recommendations

| Speedup | Recommendation |
|---------|---------------|
| > 1.1x faster | ✅ Use TransformerEngine for production |
| 0.9x - 1.1x | ⚖️ Either framework works fine |
| < 0.9x (slower) | ⚠️ Stick with standard PyTorch |

### What Affects Results

1. **Hardware**: NVIDIA GPUs show better TE support than AMD
2. **Data Type**: FP16 typically shows more improvement than FP32
3. **Problem Size**: Larger models benefit more from TE optimizations
4. **ROCm Version**: Newer ROCm versions may have better TE support

## Troubleshooting

### Issue: All tests show 1.00x speedup
**Cause**: TransformerEngine features not supported on your platform
**Solution**: This is expected on AMD. Check forward-pass tests (Linear, LayerNorm) for actual TE performance

### Issue: CUDA out of memory
**Cause**: Batch size or model size too large for GPU
**Solution**: Reduce problem size or use `--quick` mode

### Issue: Import error for transformer_engine
**Cause**: TransformerEngine not installed
**Solution**: `pip install transformer-engine`

### Issue: Very slow on first run
**Cause**: CUDA kernel compilation
**Solution**: Normal behavior, subsequent runs will be faster

## Next Steps

After running benchmarks:

1. **Review BENCHMARK_TESTS_SUMMARY.md** for detailed information on each test
2. **Check the summary table** to see which operations benefit most from TE
3. **Make an informed decision** about using TE in your project
4. **Consider platform-specific behavior** when deploying to AMD vs NVIDIA

## More Information

- See [BENCHMARK_TESTS_SUMMARY.md](BENCHMARK_TESTS_SUMMARY.md) for complete technical details
- Check TransformerEngine docs for latest compatibility information
- Report issues or suggest improvements via GitHub issues
