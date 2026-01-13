"""Example: Using DuckDB Persistent Storage with memory.py

This example demonstrates how to use the DuckDB integration in memory.py
for persistent model offloading to reduce CPU RAM usage.
"""

import torch
import torch.nn as nn
from diffusers_helper.memory import (
    MemoryOptimizationConfig,
    initialize_duckdb_storage,
    offload_model_to_duckdb,
    restore_model_from_duckdb,
    is_model_in_duckdb,
    clear_duckdb_cache,
    gpu, cpu
)


def example_basic_offload():
    """Basic example: Offload a model to DuckDB."""
    
    print("=" * 70)
    print("Example 1: Basic Model Offloading to DuckDB")
    print("=" * 70)
    
    # Create optimization config with DuckDB enabled
    config = MemoryOptimizationConfig(
        use_duckdb_storage=True,
        duckdb_path="my_model_cache.duckdb",
        duckdb_memory_limit="4GB",
        persist_on_offload=True,
        clear_cpu_after_duckdb=True,  # Free CPU RAM after storing
        use_pinned_memory=True  # Faster GPU transfers
    )
    
    # Initialize DuckDB storage
    initialize_duckdb_storage(config)
    
    # Create a large model on GPU
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    ).to(gpu)
    
    print(f"\nModel created on {gpu}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Offload model to DuckDB (frees CPU RAM)
    success = offload_model_to_duckdb(model, optim_config=config)
    
    if success:
        print("\n✓ Model offloaded to DuckDB successfully")
        print("  CPU RAM is now free (only model structure in memory)")
        
        # Later: Restore model to GPU
        print("\nRestoring model to GPU...")
        restore_model_from_duckdb(model, target_device=gpu)
        print("✓ Model restored to GPU")
        
        # Verify model still works
        x = torch.randn(10, 1000, device=gpu)
        y = model(x)
        print(f"✓ Model inference successful: output shape = {y.shape}")


def example_multiple_models():
    """Example: Offload multiple models with custom keys."""
    
    print("\n" + "=" * 70)
    print("Example 2: Multiple Model Management")
    print("=" * 70)
    
    config = MemoryOptimizationConfig(use_duckdb_storage=True)
    initialize_duckdb_storage(config)
    
    # Create multiple models
    encoder = nn.Linear(784, 256).to(gpu)
    decoder = nn.Linear(256, 784).to(gpu)
    
    # Offload with custom keys
    offload_model_to_duckdb(encoder, model_key="vae_encoder_v1", optim_config=config)
    offload_model_to_duckdb(decoder, model_key="vae_decoder_v1", optim_config=config)
    
    print("\n✓ Both models offloaded")
    
    # Check if models are in DuckDB
    print(f"Encoder in DuckDB: {is_model_in_duckdb(encoder, 'vae_encoder_v1')}")
    print(f"Decoder in DuckDB: {is_model_in_duckdb(decoder, 'vae_decoder_v1')}")
    
    # Restore specific model
    restore_model_from_duckdb(encoder, target_device=gpu, model_key="vae_encoder_v1")
    print("\n✓ Encoder restored")


def example_memory_hierarchy():
    """Example: GPU → CPU → DuckDB memory hierarchy."""
    
    print("\n" + "=" * 70)
    print("Example 3: Memory Hierarchy (GPU → CPU → DuckDB)")
    print("=" * 70)
    
    config = MemoryOptimizationConfig(
        use_duckdb_storage=True,
        clear_cpu_after_duckdb=True  # Enable hierarchy: disk-backed only
    )
    initialize_duckdb_storage(config)
    
    # Simulate multiple large models
    models = {
        "model_a": nn.Linear(5000, 5000).to(gpu),
        "model_b": nn.Linear(5000, 5000).to(gpu),
        "model_c": nn.Linear(5000, 5000).to(gpu),
    }
    
    print("\nCreated 3 large models on GPU")
    
    # Offload all to DuckDB (frees GPU and CPU)
    for name, model in models.items():
        print(f"\nOffloading {name}...")
        offload_model_to_duckdb(model, model_key=name, optim_config=config)
    
    print("\n✓ All models in DuckDB")
    print("✓ GPU memory freed")
    print("✓ CPU memory freed (clear_cpu_after_duckdb=True)")
    
    # Restore one model to GPU
    print("\nRestoring model_a to GPU...")
    restore_model_from_duckdb(models["model_a"], target_device=gpu, model_key="model_a")
    print("✓ Only model_a active in GPU")


def example_persistent_cache():
    """Example: Persistent cache across sessions."""
    
    print("\n" + "=" * 70)
    print("Example 4: Persistent Cache Across Sessions")
    print("=" * 70)
    
    # Session 1: Store model
    print("\n[Session 1] Storing model...")
    config = MemoryOptimizationConfig(
        use_duckdb_storage=True,
        duckdb_path="persistent_cache.duckdb"
    )
    initialize_duckdb_storage(config)
    
    model = nn.Linear(1000, 1000).to(gpu)
    offload_model_to_duckdb(model, model_key="my_persistent_model", optim_config=config)
    print("✓ Model stored to persistent_cache.duckdb")
    
    # Clear cache simulation (close storage)
    clear_duckdb_cache()
    
    # Session 2: Load model from disk
    print("\n[Session 2] Loading from persistent storage...")
    initialize_duckdb_storage(config)
    
    new_model = nn.Linear(1000, 1000)  # CPU model
    restore_model_from_duckdb(new_model, target_device=gpu, model_key="my_persistent_model")
    print("✓ Model loaded from disk to GPU")


if __name__ == "__main__":
    print("\n🚀 DuckDB Memory Integration Examples\n")
    
    try:
        example_basic_offload()
        example_multiple_models()
        example_memory_hierarchy()
        example_persistent_cache()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
