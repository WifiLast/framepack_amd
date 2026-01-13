"""Benchmark script for DuckDB Storage with PyTorch tensors.

This script measures performance improvements from the optimized GPU→DuckDB pipeline.
Compares old path (GPU→numpy→PyArrow→DuckDB) vs new path (GPU→torch.save→DuckDB).
"""

import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duckdb_storage import DuckDBStorage


def benchmark_tensor_storage():
    """Benchmark tensor storage performance."""
    
    print("=" * 70)
    print("DuckDB Storage Benchmark - PyTorch Tensor Operations")
    print("=" * 70)
    
    # Test configurations
    test_sizes = [
        (100, 100, "Small (10K elements)"),
        (1000, 1000, "Medium (1M elements)"),
        (2000, 2000, "Large (4M elements)"),
        (5000, 5000, "Very Large (25M elements)"),
    ]
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n" + "-" * 70)
    
    # Create storage instance (in-memory for benchmarking)
    storage = DuckDBStorage(":memory:")
    
    for height, width, desc in test_sizes:
        print(f"\n{desc}: {height}x{width}")
        print("-" * 70)
        
        # Create test tensor
        tensor = torch.randn(height, width, device=device, dtype=torch.float32)
        tensor_size_mb = tensor.element_size() * tensor.numel() / (1024**2)
        print(f"Tensor size: {tensor_size_mb:.2f} MB")
        
        # Benchmark OLD path: GPU→CPU→numpy→PyArrow→DuckDB
        print("\n[OLD PATH] GPU→numpy→PyArrow→DuckDB:")
        start = time.perf_counter()
        try:
            # This simulates the old approach
            cpu_tensor = tensor.cpu()
            numpy_array = cpu_tensor.numpy()
            storage.store_array(f"old_test_{height}", numpy_array)
            old_time = time.perf_counter() - start
            old_speed = tensor_size_mb / old_time
            print(f"  Time: {old_time:.4f}s | Speed: {old_speed:.2f} MB/s")
        except Exception as e:
            print(f"  Error: {e}")
            old_time = float('inf')
            old_speed = 0
        
        # Benchmark NEW path: GPU→torch.save→DuckDB (direct, no numpy)
        print("\n[NEW PATH] GPU→torch.save→DuckDB (direct, no numpy/pyarrow):")
        
        # Test without pinned memory
        start = time.perf_counter()
        try:
            storage.store_tensor(f"new_test_{height}_regular", tensor, use_pinned=False)
            new_time_regular = time.perf_counter() - start
            new_speed_regular = tensor_size_mb / new_time_regular
            print(f"  Regular: {new_time_regular:.4f}s | Speed: {new_speed_regular:.2f} MB/s")
        except Exception as e:
            print(f"  Error: {e}")
            new_time_regular = float('inf')
            new_speed_regular = 0
        
        # Test WITH pinned memory (if GPU available)
        if device == 'cuda':
            start = time.perf_counter()
            try:
                storage.store_tensor(f"new_test_{height}_pinned", tensor, use_pinned=True)
                new_time_pinned = time.perf_counter() - start
                new_speed_pinned = tensor_size_mb / new_time_pinned
                print(f"  Pinned:  {new_time_pinned:.4f}s | Speed: {new_speed_pinned:.2f} MB/s")
            except Exception as e:
                print(f"  Error: {e}")
                new_time_pinned = float('inf')
                new_speed_pinned = 0
            
            best_new_time = min(new_time_regular, new_time_pinned)
            best_new_speed = max(new_speed_regular, new_speed_pinned)
        else:
            best_new_time = new_time_regular
            best_new_speed = new_speed_regular
        
        # Calculate speedup
        if old_time != float('inf') and best_new_time != float('inf'):
            speedup = old_time / best_new_time
            print(f"\n✓ Speedup: {speedup:.2f}x faster ({old_speed:.2f} → {best_new_speed:.2f} MB/s)")
        
        # Verify data integrity
        print("\nVerifying data integrity...")
        try:
            retrieved_new = storage.get_tensor(f"new_test_{height}_regular", device=device)
            max_diff = torch.max(torch.abs(tensor - retrieved_new)).item()
            print(f"  Max difference: {max_diff:.2e} (should be ~0)")
            
            if max_diff < 1e-6:
                print("  ✓ Data integrity verified")
            else:
                print("  ⚠ Warning: Data mismatch detected")
        except Exception as e:
            print(f"  Error during verification: {e}")
    
    # Benchmark batched operations
    print("\n" + "=" * 70)
    print("Batched Operations Benchmark")
    print("=" * 70)
    
    # Create a mock model state dict
    state_dict = {
        f"layer_{i}.weight": torch.randn(512, 512, device=device)
        for i in range(10)
    }
    
    total_size_mb = sum(
        t.element_size() * t.numel() / (1024**2) 
        for t in state_dict.values()
    )
    
    print(f"\nStoring {len(state_dict)} tensors ({total_size_mb:.2f} MB total)")
    
    # Individual stores
    print("\n[Individual stores]:")
    start = time.perf_counter()
    for name, tensor in state_dict.items():
        storage.store_tensor(f"individual.{name}", tensor, use_pinned=True)
    individual_time = time.perf_counter() - start
    individual_speed = total_size_mb / individual_time
    print(f"  Time: {individual_time:.4f}s | Speed: {individual_speed:.2f} MB/s")
    
    # Batched store
    print("\n[Batched store]:")
    start = time.perf_counter()
    storage.store_tensors_batched("batched", state_dict, use_pinned=True)
    batched_time = time.perf_counter() - start
    batched_speed = total_size_mb / batched_time
    print(f"  Time: {batched_time:.4f}s | Speed: {batched_speed:.2f} MB/s")
    
    if individual_time > 0 and batched_time > 0:
        speedup = individual_time / batched_time
        print(f"\n✓ Batched speedup: {speedup:.2f}x faster")
    
    # Storage statistics
    print("\n" + "=" * 70)
    print("Storage Statistics")
    print("=" * 70)
    stats = storage.get_statistics()
    print(f"\nTotal tensors stored: {stats['tensor_count']}")
    print(f"Total arrays stored: {stats['array_count']}")
    print(f"Total size: {stats['total_size_bytes'] / (1024**2):.2f} MB")
    
    storage.close()
    print("\n✓ Benchmark completed")


def test_basic_functionality():
    """Test basic functionality of tensor storage."""
    
    print("\n" + "=" * 70)
    print("Basic Functionality Tests")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    storage = DuckDBStorage(":memory:")
    
    # Test 1: Store and retrieve
    print("\n[Test 1] Store and retrieve tensor")
    tensor = torch.randn(100, 100, device=device)
    storage.store_tensor("test1", tensor)
    retrieved = storage.get_tensor("test1", device=device)
    assert torch.allclose(tensor, retrieved), "Tensor mismatch!"
    print("  ✓ PASSED")
    
    # Test 2: Different devices
    print("\n[Test 2] Cross-device storage")
    cpu_tensor = torch.randn(50, 50, device='cpu')
    storage.store_tensor("test2_cpu", cpu_tensor)
    retrieved_cpu = storage.get_tensor("test2_cpu", device='cpu')
    assert torch.allclose(cpu_tensor, retrieved_cpu), "CPU tensor mismatch!"
    print("  ✓ PASSED")
    
    # Test 3: Different dtypes
    print("\n[Test 3] Different dtypes")
    for dtype in [torch.float32, torch.float16, torch.int64, torch.bool]:
        if dtype == torch.bool:
            t = torch.randint(0, 2, (10, 10), dtype=dtype, device=device)
        elif dtype == torch.int64:
            t = torch.randint(0, 100, (10, 10), dtype=dtype, device=device)
        else:
            t = torch.randn(10, 10, dtype=dtype, device=device)
        
        storage.store_tensor(f"test3_{dtype}", t)
        retrieved_t = storage.get_tensor(f"test3_{dtype}", device=device)
        assert t.dtype == retrieved_t.dtype, f"dtype mismatch for {dtype}"
        print(f"  ✓ {dtype} - PASSED")
    
    # Test 4: Batched operations
    print("\n[Test 4] Batched operations")
    state_dict = {
        "weight": torch.randn(50, 50, device=device),
        "bias": torch.randn(50, device=device),
    }
    storage.store_tensors_batched("model", state_dict)
    retrieved_dict = storage.get_tensors_batched("model", device=device)
    
    assert set(state_dict.keys()) == set(retrieved_dict.keys()), "Key mismatch!"
    for key in state_dict.keys():
        assert torch.allclose(state_dict[key], retrieved_dict[key]), f"Mismatch in {key}!"
    print("  ✓ PASSED")
    
    # Test 5: Metadata
    print("\n[Test 5] Metadata preservation")
    tensor_with_grad = torch.randn(20, 20, device=device, requires_grad=True)
    storage.store_tensor("test5_grad", tensor_with_grad)
    
    tensors_list = storage.list_tensors()
    found = False
    for t_info in tensors_list:
        if t_info['key'] == 'test5_grad':
            found = True
            assert t_info['requires_grad'] == True, "requires_grad not preserved!"
            print(f"  ✓ Metadata: {t_info}")
            break
    assert found, "Tensor not found in list!"
    print("  ✓ PASSED")
    
    storage.close()
    print("\n✓ All basic functionality tests PASSED")


if __name__ == "__main__":
    print("\n🚀 DuckDB Storage Optimization Benchmark\n")
    
    try:
        test_basic_functionality()
        benchmark_tensor_storage()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
