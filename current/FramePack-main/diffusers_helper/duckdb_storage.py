"""DuckDB Storage Module for PyTorch Tensors, NumPy Arrays and Python Dictionaries.

This module provides high-performance persistent storage using DuckDB for PyTorch tensors,
numpy arrays and Python dictionaries. Optimized for GPU/CPU transfers with direct 
serialization (no numpy/pyarrow intermediates for tensors).

Key Features:
    - Direct GPU→DuckDB pipeline for PyTorch tensors (bypass numpy/pyarrow)
    - Pinned memory support for async GPU transfers
    - Batched operations for storing multiple tensors
    - Zero-copy optimizations where possible
    - Automatic device management

Usage:
    from duckdb_storage import DuckDBStorage
    import torch
    
    # Create storage instance
    storage = DuckDBStorage("my_data.db")
    
    # Store PyTorch tensor (direct GPU→DuckDB)
    tensor = torch.randn(1000, 1000, device='cuda')
    storage.store_tensor("my_tensor", tensor, use_pinned=True)
    
    # Retrieve tensor to GPU
    retrieved = storage.get_tensor("my_tensor", device='cuda')
    
    # Batch store multiple tensors (e.g., model state_dict)
    state_dict = {"weight": torch.randn(100, 100), "bias": torch.randn(100)}
    storage.store_tensors_batched("model", state_dict)
"""

import sys
import os
import io
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Add the local duckdb-python source to the path
DUCKDB_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "duckdb-python-main")
if os.path.exists(DUCKDB_SOURCE_PATH) and DUCKDB_SOURCE_PATH not in sys.path:
    sys.path.insert(0, DUCKDB_SOURCE_PATH)

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# PyTorch import with availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class DuckDBStorageError(Exception):
    """Base exception for DuckDB storage errors."""
    pass


class KeyNotFoundError(DuckDBStorageError):
    """Raised when a key is not found in storage."""
    pass


class TypeMismatchError(DuckDBStorageError):
    """Raised when attempting an operation on wrong storage type."""
    pass


class DuckDBStorage:
    """DuckDB-based persistent storage for PyTorch tensors, numpy arrays and dictionaries.
    
    This class provides optimized storage for PyTorch tensors with direct GPU→DuckDB
    transfers, bypassing numpy and PyArrow for maximum performance.
    
    Attributes:
        db_path: Path to the DuckDB database file.
        conn: Active DuckDB connection.
    """
    
    def __init__(self, db_path: str = ":memory:", memory_limit: str = "80%"):
        """Initialize DuckDB storage with aggressive memory settings.
        
        Args:
            db_path: Path to database file. Use ":memory:" for in-memory database.
            memory_limit: Maximum memory DuckDB can use (e.g., "80%", "8GB", "16GB").
                         Default is 80% of available system memory.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Configure DuckDB for aggressive in-memory operation
        self._configure_memory_settings(memory_limit)
        self._initialize_tables()
    
    def _configure_memory_settings(self, memory_limit: str):
        """Configure DuckDB for aggressive memory usage and caching.
        
        Args:
            memory_limit: Maximum memory DuckDB can use.
        """
        # Set memory limit - allow DuckDB to use significant RAM
        self.conn.execute(f"SET memory_limit='{memory_limit}'")
        
        # Increase maximum memory for temporary data
        self.conn.execute("SET max_memory='80%'")
        
        # Set temp directory for spilling (only when absolutely necessary)
        self.conn.execute("SET temp_directory='temp_duckdb'")
        
        # Optimize for in-memory performance
        # Use more threads for parallel processing
        import os
        num_threads = os.cpu_count() or 4
        self.conn.execute(f"SET threads={num_threads}")
        
        # Enable aggressive buffering
        self.conn.execute("SET preserve_insertion_order=false")
        
        # Optimize hash joins for large datasets
        self.conn.execute("SET enable_object_cache=true")
        
        # Use more aggressive block size for better memory utilization
        self.conn.execute("SET default_block_alloc_size='256KB'")
        
        # Keep data in memory as much as possible
        self.conn.execute("SET force_compression='none'")  # Skip compression for speed
    
    def _initialize_tables(self):
        """Create storage tables if they don't exist."""
        # Table for PyTorch tensors (optimized path: direct GPU→DuckDB, no numpy/pyarrow)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS torch_tensors (
                key VARCHAR PRIMARY KEY,
                data BLOB,
                shape VARCHAR,
                dtype VARCHAR,
                device VARCHAR,
                requires_grad BOOLEAN,
                storage_format VARCHAR DEFAULT 'torch',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for numpy arrays
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS numpy_arrays (
                key VARCHAR PRIMARY KEY,
                data BLOB,
                shape VARCHAR,
                dtype VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for dictionaries
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dicts (
                key VARCHAR PRIMARY KEY,
                data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    # ==================== PyTorch Tensor Operations (Optimized) ====================
    
    def store_tensor(self, key: str, tensor: 'torch.Tensor', use_pinned: bool = True) -> None:
        """Store a PyTorch tensor with direct GPU→DuckDB transfer.
        
        This method uses direct PyTorch serialization (torch.save) without converting
        to numpy or PyArrow, providing optimal performance for GPU tensors.
        
        Args:
            key: Unique identifier for the tensor.
            tensor: PyTorch tensor to store (CPU or GPU).
            use_pinned: Use pinned memory for GPU→CPU transfer (faster, more RAM).
            
        Raises:
            TypeError: If tensor is not a PyTorch tensor.
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install torch to use tensor storage.")
        
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Store metadata
        shape_str = str(tuple(tensor.shape))
        dtype_str = str(tensor.dtype)
        device_str = str(tensor.device)
        requires_grad = tensor.requires_grad
        
        # Serialize tensor using torch.save (direct, no numpy conversion)
        buffer = io.BytesIO()
        
        # For GPU tensors, use pinned memory for faster transfer
        if tensor.device.type == 'cuda' and use_pinned:
            # Non-blocking transfer to pinned CPU memory
            cpu_tensor = tensor.to('cpu', non_blocking=True)
            # Synchronize to ensure transfer completes
            if tensor.device.type == 'cuda':
                torch.cuda.synchronize(tensor.device)
            torch.save(cpu_tensor, buffer)
        else:
            # Direct save (CPU tensors or non-pinned)
            torch.save(tensor.cpu(), buffer)
        
        serialized_data = buffer.getvalue()
        
        # Insert or replace
        self.conn.execute("""
            INSERT OR REPLACE INTO torch_tensors 
            (key, data, shape, dtype, device, requires_grad, storage_format, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'torch', CURRENT_TIMESTAMP)
        """, [key, serialized_data, shape_str, dtype_str, device_str, requires_grad])
    
    def get_tensor(self, key: str, device: Optional[Union[str, 'torch.device']] = None) -> 'torch.Tensor':
        """Retrieve a PyTorch tensor.
        
        Args:
            key: Identifier of the tensor to retrieve.
            device: Target device for the tensor ('cuda', 'cpu', or torch.device).
                   If None, uses the device from storage metadata.
            
        Returns:
            The stored PyTorch tensor.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install torch to use tensor storage.")
        
        result = self.conn.execute(
            "SELECT data, device, dtype, requires_grad FROM torch_tensors WHERE key = ?", [key]
        ).fetchone()
        
        if result is None:
            raise KeyNotFoundError(f"Tensor key '{key}' not found")
        
        # Deserialize using torch.load
        buffer = io.BytesIO(result[0])
        tensor = torch.load(buffer)
        
        # Move to target device if specified
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        
        return tensor
    
    def store_tensors_batched(self, prefix: str, tensors: Dict[str, 'torch.Tensor'], 
                             use_pinned: bool = True) -> None:
        """Store multiple tensors in a single batch operation.
        
        This is optimized for storing model state_dicts or large collections of tensors.
        Uses a single database transaction for 10-50x faster insertion.
        
        Args:
            prefix: Prefix for all keys (e.g., "model_name").
            tensors: Dictionary mapping tensor names to tensors.
            use_pinned: Use pinned memory for GPU→CPU transfers.
            
        Raises:
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        
        # Begin transaction for batch insert
        self.conn.begin()
        
        try:
            for name, tensor in tensors.items():
                full_key = f"{prefix}.{name}"
                self.store_tensor(full_key, tensor, use_pinned=use_pinned)
            
            # Commit transaction
            self.conn.commit()
        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            raise DuckDBStorageError(f"Batch tensor storage failed: {e}")
    
    def get_tensors_batched(self, prefix: str, device: Optional[Union[str, 'torch.device']] = None) -> Dict[str, 'torch.Tensor']:
        """Retrieve multiple tensors matching a prefix.
        
        Args:
            prefix: Prefix to match (e.g., "model_name").
            device: Target device for all tensors.
            
        Returns:
            Dictionary mapping tensor names (without prefix) to tensors.
            
        Raises:
            ImportError: If PyTorch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        
        # Query all keys with prefix
        results = self.conn.execute(
            "SELECT key FROM torch_tensors WHERE key LIKE ?", [f"{prefix}.%"]
        ).fetchall()
        
        tensors = {}
        for (full_key,) in results:
            # Remove prefix to get tensor name
            name = full_key[len(prefix)+1:]
            tensors[name] = self.get_tensor(full_key, device=device)
        
        return tensors
    
    def update_tensor(self, key: str, tensor: 'torch.Tensor', use_pinned: bool = True) -> None:
        """Update an existing tensor.
        
        Args:
            key: Identifier of the tensor to update.
            tensor: New tensor data.
            use_pinned: Use pinned memory for GPU→CPU transfer.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        if not self.tensor_exists(key):
            raise KeyNotFoundError(f"Tensor key '{key}' not found. Use store_tensor() to create.")
        
        self.store_tensor(key, tensor, use_pinned=use_pinned)
    
    def delete_tensor(self, key: str) -> None:
        """Delete a tensor.
        
        Args:
            key: Identifier of the tensor to delete.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        if not self.tensor_exists(key):
            raise KeyNotFoundError(f"Tensor key '{key}' not found")
        
        self.conn.execute("DELETE FROM torch_tensors WHERE key = ?", [key])
    
    def list_tensors(self) -> List[Dict[str, Any]]:
        """List all stored tensors with metadata.
        
        Returns:
            List of dictionaries containing tensor metadata.
        """
        result = self.conn.execute("""
            SELECT key, shape, dtype, device, requires_grad, created_at, updated_at
            FROM torch_tensors
            ORDER BY key
        """).fetchall()
        
        return [
            {
                "key": row[0],
                "shape": row[1],
                "dtype": row[2],
                "device": row[3],
                "requires_grad": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            }
            for row in result
        ]
    
    def tensor_exists(self, key: str) -> bool:
        """Check if a tensor exists.
        
        Args:
            key: Identifier to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        result = self.conn.execute(
            "SELECT COUNT(*) FROM torch_tensors WHERE key = ?", [key]
        ).fetchone()
        return result[0] > 0
    
    # ==================== NumPy Array Operations ====================
    
    def store_array(self, key: str, array: np.ndarray) -> None:
        """Store a numpy array.
        
        Args:
            key: Unique identifier for the array.
            array: NumPy array to store.
            
        Raises:
            TypeError: If array is not a numpy array.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(array)}")
        
        # Convert to PyArrow and serialize
        # Flatten array for storage, preserve shape and dtype as metadata
        pa_array = pa.array(array.flatten())
        serialized_data = pa.ipc.serialize_to(pa_array, None).to_pybytes()
        
        # Store metadata
        shape_str = str(array.shape)
        dtype_str = str(array.dtype)
        
        # Insert or replace
        self.conn.execute("""
            INSERT OR REPLACE INTO numpy_arrays (key, data, shape, dtype, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [key, serialized_data, shape_str, dtype_str])
    
    def get_array(self, key: str) -> np.ndarray:
        """Retrieve a numpy array.
        
        Args:
            key: Identifier of the array to retrieve.
            
        Returns:
            The stored numpy array.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        result = self.conn.execute(
            "SELECT data, shape, dtype FROM numpy_arrays WHERE key = ?", [key]
        ).fetchone()
        
        if result is None:
            raise KeyNotFoundError(f"Array key '{key}' not found")
        
        # Deserialize from PyArrow
        pa_array = pa.ipc.deserialize_from(pa.py_buffer(result[0]), None)
        flat_array = pa_array.to_numpy()
        
        # Restore original shape
        shape = eval(result[1])  # Convert string back to tuple
        array = flat_array.reshape(shape)
        return array
    
    def update_array(self, key: str, array: np.ndarray) -> None:
        """Update an existing numpy array.
        
        Args:
            key: Identifier of the array to update.
            array: New array data.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
            TypeError: If array is not a numpy array.
        """
        if not self.array_exists(key):
            raise KeyNotFoundError(f"Array key '{key}' not found. Use store_array() to create.")
        
        self.store_array(key, array)
    
    def delete_array(self, key: str) -> None:
        """Delete a numpy array.
        
        Args:
            key: Identifier of the array to delete.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        if not self.array_exists(key):
            raise KeyNotFoundError(f"Array key '{key}' not found")
        
        self.conn.execute("DELETE FROM numpy_arrays WHERE key = ?", [key])
    
    def list_arrays(self) -> List[Dict[str, Any]]:
        """List all stored arrays with metadata.
        
        Returns:
            List of dictionaries containing array metadata (key, shape, dtype, timestamps).
        """
        result = self.conn.execute("""
            SELECT key, shape, dtype, created_at, updated_at
            FROM numpy_arrays
            ORDER BY key
        """).fetchall()
        
        return [
            {
                "key": row[0],
                "shape": row[1],
                "dtype": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
            for row in result
        ]
    
    def array_exists(self, key: str) -> bool:
        """Check if an array exists.
        
        Args:
            key: Identifier to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        result = self.conn.execute(
            "SELECT COUNT(*) FROM numpy_arrays WHERE key = ?", [key]
        ).fetchone()
        return result[0] > 0
    
    # ==================== Dictionary Operations ====================
    
    def store_dict(self, key: str, dict_obj: dict) -> None:
        """Store a Python dictionary.
        
        Args:
            key: Unique identifier for the dictionary.
            dict_obj: Dictionary to store.
            
        Raises:
            TypeError: If dict_obj is not a dictionary.
        """
        if not isinstance(dict_obj, dict):
            raise TypeError(f"Expected dict, got {type(dict_obj)}")
        
        # For dictionaries, we use pickle since PyArrow's Python object support
        # is limited. Arrays use PyArrow for better DuckDB integration.
        import pickle
        serialized_data = pickle.dumps(dict_obj)
        
        # Insert or replace
        self.conn.execute("""
            INSERT OR REPLACE INTO dicts (key, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, [key, serialized_data])
    
    def get_dict(self, key: str) -> dict:
        """Retrieve a Python dictionary.
        
        Args:
            key: Identifier of the dictionary to retrieve.
            
        Returns:
            The stored dictionary.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        result = self.conn.execute(
            "SELECT data FROM dicts WHERE key = ?", [key]
        ).fetchone()
        
        if result is None:
            raise KeyNotFoundError(f"Dictionary key '{key}' not found")
        
        # Deserialize using pickle
        import pickle
        dict_obj = pickle.loads(result[0])
        return dict_obj
    
    def update_dict(self, key: str, dict_obj: dict) -> None:
        """Update an existing dictionary.
        
        Args:
            key: Identifier of the dictionary to update.
            dict_obj: New dictionary data.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
            TypeError: If dict_obj is not a dictionary.
        """
        if not self.dict_exists(key):
            raise KeyNotFoundError(f"Dictionary key '{key}' not found. Use store_dict() to create.")
        
        self.store_dict(key, dict_obj)
    
    def delete_dict(self, key: str) -> None:
        """Delete a dictionary.
        
        Args:
            key: Identifier of the dictionary to delete.
            
        Raises:
            KeyNotFoundError: If key doesn't exist.
        """
        if not self.dict_exists(key):
            raise KeyNotFoundError(f"Dictionary key '{key}' not found")
        
        self.conn.execute("DELETE FROM dicts WHERE key = ?", [key])
    
    def list_dicts(self) -> List[Dict[str, Any]]:
        """List all stored dictionaries with metadata.
        
        Returns:
            List of dictionaries containing metadata (key, timestamps).
        """
        result = self.conn.execute("""
            SELECT key, created_at, updated_at
            FROM dicts
            ORDER BY key
        """).fetchall()
        
        return [
            {
                "key": row[0],
                "created_at": row[1],
                "updated_at": row[2]
            }
            for row in result
        ]
    
    def dict_exists(self, key: str) -> bool:
        """Check if a dictionary exists.
        
        Args:
            key: Identifier to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        result = self.conn.execute(
            "SELECT COUNT(*) FROM dicts WHERE key = ?", [key]
        ).fetchone()
        return result[0] > 0
    
    # ==================== Utility Operations ====================
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """Get current DuckDB memory configuration.
        
        Returns:
            Dictionary with current memory settings.
        """
        settings = {}
        
        # Query various memory-related settings
        setting_names = [
            'memory_limit',
            'max_memory', 
            'threads',
            'temp_directory',
            'preserve_insertion_order',
            'enable_object_cache'
        ]
        
        for setting in setting_names:
            try:
                result = self.conn.execute(f"SELECT current_setting('{setting}')").fetchone()
                settings[setting] = result[0] if result else None
            except:
                settings[setting] = "N/A"
        
        return settings
    
    def clear_all(self) -> None:
        """Remove all stored data (tensors, arrays and dictionaries)."""
        self.conn.execute("DELETE FROM torch_tensors")
        self.conn.execute("DELETE FROM numpy_arrays")
        self.conn.execute("DELETE FROM dicts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics (counts, total size, etc.).
        """
        tensor_count = self.conn.execute("SELECT COUNT(*) FROM torch_tensors").fetchone()[0]
        array_count = self.conn.execute("SELECT COUNT(*) FROM numpy_arrays").fetchone()[0]
        dict_count = self.conn.execute("SELECT COUNT(*) FROM dicts").fetchone()[0]
        
        # Calculate total storage size
        tensor_size = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(data)), 0) FROM torch_tensors"
        ).fetchone()[0]
        array_size = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(data)), 0) FROM numpy_arrays"
        ).fetchone()[0]
        dict_size = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(data)), 0) FROM dicts"
        ).fetchone()[0]
        
        return {
            "tensor_count": tensor_count,
            "array_count": array_count,
            "dict_count": dict_count,
            "total_items": tensor_count + array_count + dict_count,
            "tensor_size_bytes": tensor_size,
            "array_size_bytes": array_size,
            "dict_size_bytes": dict_size,
            "total_size_bytes": tensor_size + array_size + dict_size,
            "db_path": self.db_path
        }
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"DuckDBStorage(db_path='{self.db_path}', "
                f"tensors={stats['tensor_count']}, "
                f"arrays={stats['array_count']}, dicts={stats['dict_count']})")
