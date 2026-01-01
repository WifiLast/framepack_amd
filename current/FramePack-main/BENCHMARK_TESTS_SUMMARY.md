# TransformerEngine Benchmark Tests - Summary

## Overview

The `benchmark_transformer_engine.py` script has been significantly enhanced with comprehensive tests to evaluate TransformerEngine performance vs standard PyTorch across various transformer operations.

## New Benchmark Tests Added

### 1. **LayerNorm Benchmark** (`benchmark_layernorm`)
- **Purpose**: Compare LayerNorm performance
- **Tests**: PyTorch LayerNorm vs TE LayerNorm
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size

### 2. **RMSNorm Benchmark** (`benchmark_rmsnorm`)
- **Purpose**: Test RMSNorm (used in modern LLMs like LLaMA)
- **Tests**: Custom PyTorch RMSNorm vs TE RMSNorm
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size

### 3. **Multi-Head Attention Benchmark** (`benchmark_attention`)
- **Purpose**: Compare attention mechanism performance
- **Tests**: PyTorch MultiheadAttention vs TE MultiheadAttention
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size, 12 heads
- **Note**: Includes AMD compatibility handling - falls back to PyTorch if TE attention is not supported

### 4. **Dropout Benchmark** (`benchmark_dropout`)
- **Purpose**: Test dropout overhead
- **Tests**: PyTorch Dropout vs TE context dropout
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size, p=0.1

### 5. **Backward Pass Benchmark** (`benchmark_backward_pass`)
- **Purpose**: Test forward + backward propagation (critical for training)
- **Tests**: Complete gradient computation for Linear layers
- **Default config**: 16 batch, 128 seq_len, 768→3072

### 6. **Activation Functions Benchmark** (`benchmark_activation_functions`)
- **Purpose**: Compare different activation functions
- **Tests**: GELU, ReLU, SiLU, Tanh
- **Default config**: 16 batch, 128 seq_len, 3072 hidden_size

### 7. **Sequence Length Variations** (`benchmark_sequence_lengths`)
- **Purpose**: Test performance scaling with sequence length
- **Tests**: 7 different lengths: 32, 64, 128, 256, 512, 1024, 2048
- **Shows**: How performance scales with longer sequences

### 8. **Full Transformer Block Benchmark** (`benchmark_transformer_block`)
- **Purpose**: Test complete transformer layer (most realistic benchmark)
- **Components**: Attention + FFN + LayerNorm + Dropout + Residuals
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size, 12 heads, 3072 FFN
- **Note**: Uses PyTorch attention with TE Linear/LayerNorm on AMD for compatibility

### 9. **Training Step Benchmark** (`benchmark_training_step`)
- **Purpose**: End-to-end training performance
- **Tests**: Forward + Backward + Optimizer step
- **Includes**: Loss calculation, gradient computation, weight updates
- **Default config**: 16 batch, 128 seq_len, 768 hidden_size, AdamW optimizer

### 10. **Memory Usage Benchmark** (`benchmark_memory_usage`)
- **Purpose**: Compare GPU memory consumption
- **Tests**: Small and large layer configurations
- **Metrics**: Peak memory usage in MB
- **Configurations**:
  - Small: 768→768
  - Large: 2048→8192

## AMD Compatibility Fixes

### Issues Identified

TransformerEngine has several compatibility issues on AMD hardware (ROCm):

1. **Fused Attention Error**:
   ```
   RuntimeError: fused attn configs not supported in ck_fused_attn fwd pass.
   ```

2. **Backward Pass GEMM Error**:
   ```
   RuntimeError: Unable to find any suitable algorithms
   ```

### Solutions Implemented

Added graceful error handling with fallback mechanisms throughout the benchmark suite:

1. **Attention Benchmark**: Wraps TE attention in try-except, falls back to PyTorch if unsupported
2. **Backward Pass Benchmark**: Catches GEMM errors during backward pass, falls back to PyTorch
3. **Transformer Block**: Uses PyTorch attention with TE Linear/LayerNorm layers for compatibility
4. **Training Step Benchmark**: Wraps entire training loop in try-except to handle backward pass failures
5. **User-friendly messages**: Shows clear warnings when TE features are unavailable on the platform

### What Works on AMD

- ✅ **Linear layers** (forward pass only)
- ✅ **LayerNorm** (forward pass only)
- ✅ **RMSNorm** (forward pass only)
- ✅ **Matrix multiplication** (forward pass)
- ❌ **Attention** (fused attention not supported)
- ❌ **Backward pass** (GEMM algorithms not available)
- ❌ **Training** (requires backward pass)

### Impact

The benchmarks will now run successfully on AMD hardware without crashes. When TE features are unavailable, the benchmark:
- Shows a clear warning message
- Falls back to PyTorch implementation
- Reports identical times for both (since TE isn't being used)
- Continues with remaining tests instead of crashing

## Command-Line Interface

### New Arguments

```bash
--comprehensive          # Run all comprehensive benchmarks
--test <name>           # Run specific test only
--quick                 # Quick run (fewer iterations)
--iterations <N>        # Number of iterations (default: 100)
--warmup <N>            # Warmup iterations (default: 10)
--no-warmup            # Skip warmup
--verbose              # Verbose output
```

### Available Test Names

- `linear` - Linear layer benchmark
- `mlp` - MLP (2-layer FFN) benchmark
- `matmul` - Matrix multiplication benchmark
- `layernorm` - LayerNorm benchmark
- `rmsnorm` - RMSNorm benchmark
- `attention` - Multi-head attention benchmark
- `dropout` - Dropout benchmark
- `backward` - Backward pass benchmark
- `activations` - Activation functions benchmark
- `transformer` - Full transformer block benchmark
- `training` - Training step benchmark
- `memory` - Memory usage benchmark
- `batch_sizes` - Varying batch sizes benchmark
- `layer_sizes` - Varying layer sizes benchmark
- `seq_lengths` - Varying sequence lengths benchmark

## Usage Examples

### Default Run (Basic + Core Tests)
```bash
python benchmark_transformer_engine.py
```
Runs: Linear, MLP, MatMul, LayerNorm, RMSNorm, Attention, Backward pass

### Comprehensive Benchmark
```bash
python benchmark_transformer_engine.py --comprehensive
```
Runs all tests including scaling studies and memory benchmarks

### Quick Test
```bash
python benchmark_transformer_engine.py --quick
```
Reduced iterations (20 instead of 100) for faster results

### Specific Test
```bash
# Test only attention performance
python benchmark_transformer_engine.py --test attention

# Test only transformer block
python benchmark_transformer_engine.py --test transformer

# Test memory usage
python benchmark_transformer_engine.py --test memory
```

### Custom Iterations
```bash
# More accurate results with 200 iterations
python benchmark_transformer_engine.py --iterations 200

# Test specific component with custom iterations
python benchmark_transformer_engine.py --test training --iterations 50
```

## Output Format

The benchmark provides detailed output including:

1. **Per-test results**: PyTorch time, TE time, speedup
2. **Summary table**: All benchmarks with comparison
3. **Overall verdict**: Recommendation based on overall performance
4. **Progress indicators**: Shows what's being benchmarked

Example output:
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

## Testing Strategy

### Default Mode (No flags)
- **Coverage**: Core operations
- **Duration**: ~2-3 minutes
- **Use case**: Quick validation

### Comprehensive Mode (`--comprehensive`)
- **Coverage**: All operations + scaling studies
- **Duration**: ~15-20 minutes
- **Use case**: Full performance characterization

### Quick Mode (`--quick`)
- **Coverage**: Core operations only
- **Duration**: ~30 seconds
- **Use case**: Rapid iteration during development

## Platform Notes

### AMD/ROCm Specific
- TransformerEngine fused attention is NOT supported
- Benchmarks automatically fall back to PyTorch attention
- Linear and LayerNorm operations work correctly with TE
- Memory benchmarks require CUDA-compatible calls (torch.cuda.*)

### NVIDIA/CUDA
- All TE features should work
- Fused attention can provide significant speedups
- Full TE transformer blocks are supported

## Performance Insights

The benchmarks help answer:
1. **Is TE faster than PyTorch?** - Overall speedup comparison
2. **Which operations benefit most?** - Per-operation analysis
3. **How does it scale?** - Batch size, sequence length, layer size scaling
4. **What's the memory overhead?** - Memory usage comparison
5. **Training performance?** - Real-world training step timing

## Recommendations

Based on benchmark results:
- **>1.1x speedup**: Use TransformerEngine for production
- **0.9x-1.1x**: Either framework works, TE may offer other benefits
- **<0.9x slower**: Stick with standard PyTorch

## Future Enhancements

Potential additions:
- [ ] FP8 precision benchmarks (MI300 specific)
- [ ] Multi-GPU benchmarks
- [ ] Gradient accumulation benchmarks
- [ ] Mixed precision training benchmarks (AMP)
- [ ] Flash Attention comparison (when available on AMD)
- [ ] Different model architectures (GPT, BERT, T5)
