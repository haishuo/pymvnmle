"""
Backend registry and auto-selection for PyMVNMLE
UPDATED: Now prioritizes finite difference compatibility for R matching

CRITICAL DISCOVERY (January 2025):
R uses finite differences, not analytical gradients. We adjust backend selection
to prioritize configurations that work well with finite differences.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .base import BackendInterface, BackendNotAvailableError
from .numpy_backend import NumPyBackend


# Global registry of available backend classes
_BACKEND_CLASSES = {
    'numpy': NumPyBackend,
}

# Global cache of initialized backends
_BACKEND_INSTANCES = {}


def _import_optional_backends():
    """Import optional backends with graceful fallback."""
    # Try to import CuPy backend
    try:
        from .cupy_backend import CuPyBackend
        _BACKEND_CLASSES['cupy'] = CuPyBackend
    except ImportError:
        pass  # CuPy not available
    
    # Try to import Metal backend  
    try:
        from .metal_backend import MetalBackend
        _BACKEND_CLASSES['metal'] = MetalBackend
    except ImportError:
        pass  # PyTorch MPS not available
    
    # Try to import JAX backend
    try:
        from .jax_backend import JAXBackend
        _BACKEND_CLASSES['jax'] = JAXBackend
    except ImportError:
        pass  # JAX not available


def get_available_backends() -> List[str]:
    """Get list of backends available on this system."""
    _import_optional_backends()
    
    available = ['numpy']  # Always available
    
    # Try other backends
    for name, backend_class in _BACKEND_CLASSES.items():
        if name == 'numpy':
            continue  # Already included
            
        try:
            backend = backend_class()
            if backend.is_available:
                available.append(name)
        except Exception:
            continue
    
    return available


def select_optimal_backend(data_shape: Tuple[int, int], 
                          available_backends: Optional[List[str]] = None) -> str:
    """
    Select optimal backend with preference for finite difference compatibility.
    
    UPDATED: Now considers finite difference performance in selection logic.
    """
    if available_backends is None:
        available_backends = get_available_backends()
    
    n_obs, n_vars = data_shape
    
    # UPDATED LOGIC: For finite differences, CPU often performs better 
    # for small-medium problems due to reduced GPU transfer overhead
    
    # Small problems: CPU is best (finite differences don't benefit from GPU)
    if n_vars <= 10 or n_obs <= 100:
        return 'numpy'
    
    # Medium problems: Still prefer CPU for finite differences unless problem is large
    if n_vars <= 50 and n_obs <= 2000:
        # Only use GPU if significant computational benefit expected
        if n_obs > 1000 and n_vars > 20:
                # Large enough to justify GPU overhead
            gpu_preference = _get_gpu_backend_preference()
            for backend in gpu_preference:
                if backend in available_backends:
                    return backend
        
        # Default to CPU for medium problems with finite differences
        return 'numpy'
    
    # Large problems: GPU can help despite finite difference overhead
    elif n_vars > 50 or n_obs > 2000:
        gpu_preference = _get_gpu_backend_preference()
        for backend in gpu_preference:
            if backend in available_backends:
                return backend
        return 'numpy'
    
    return 'numpy'  # Safe fallback


def _get_gpu_backend_preference() -> List[str]:
    """Get ordered preference for GPU backends."""
    # For finite differences, preference order is:
    # 1. JAX - good XLA optimization even for finite differences
    # 2. CuPy - mature, stable 
    # 3. Metal - Apple Silicon support
    return ['jax', 'cupy', 'metal']


def get_backend(backend_name: Union[str, BackendInterface]) -> BackendInterface:
    """Get a backend instance by name or return existing instance."""
    # If already a backend instance, return it
    if isinstance(backend_name, BackendInterface):
        return backend_name
    
    # Handle auto-selection
    if backend_name == 'auto':
        available = get_available_backends()
        if not available:
            raise BackendNotAvailableError("No backends available")
        backend_name = available[0]  # Default to first available (numpy)
    
    # Validate backend name
    if backend_name not in ['numpy', 'cupy', 'metal', 'jax']:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    # Check if backend is cached
    if backend_name in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend_name]
    
    # Import optional backends if needed
    _import_optional_backends()
    
    # Check if backend class is available
    if backend_name not in _BACKEND_CLASSES:
        raise BackendNotAvailableError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {get_available_backends()}"
        )
    
    # Instantiate backend
    try:
        backend_class = _BACKEND_CLASSES[backend_name]
        backend = backend_class()
        
        if not backend.is_available:
            raise BackendNotAvailableError(
                f"Backend '{backend_name}' failed initialization check"
            )
        
        # Cache the instance
        _BACKEND_INSTANCES[backend_name] = backend
        return backend
        
    except Exception as e:
        if isinstance(e, BackendNotAvailableError):
            raise
        else:
            raise BackendNotAvailableError(
                f"Failed to initialize backend '{backend_name}': {e}"
            )


def get_backend_with_fallback(backend_name: str, 
                             data_shape: Optional[Tuple[int, int]] = None,
                             verbose: bool = False) -> BackendInterface:
    """
    Get backend with intelligent fallback to CPU if requested backend fails.
    
    UPDATED: Considers finite difference performance in auto-selection.
    """
    original_backend_name = backend_name
    
    # Handle auto-selection with finite difference preference
    if backend_name == 'auto':
        if data_shape is None:
            raise ValueError("data_shape required for backend='auto'")
        
        available = get_available_backends()
        backend_name = select_optimal_backend(data_shape, available)
        
        if verbose:
            n_obs, n_vars = data_shape
            print(f"Auto-selected '{backend_name}' backend for dataset ({n_obs}×{n_vars})")
            print("(Optimized for finite difference computation)")
    
    # Try to get the requested/selected backend
    try:
        return get_backend(backend_name)
        
    except BackendNotAvailableError as e:
        # Handle fallback logic
        if original_backend_name != 'auto' and backend_name != 'numpy':
            # User explicitly requested this backend - give helpful error
            available = get_available_backends()
            raise BackendNotAvailableError(
                f"Requested backend '{backend_name}' is not available. "
                f"Available backends: {available}. "
                f"Original error: {e}"
            )
        
        elif backend_name != 'numpy':
            # Auto-selected backend failed, fall back to CPU
            if verbose:
                print(f"Warning: Backend '{backend_name}' failed, falling back to CPU")
                print("(CPU often performs well for finite difference computation)")
            
            try:
                return get_backend('numpy')
            except BackendNotAvailableError:
                # Even CPU failed - this is bad!
                raise BackendNotAvailableError(
                    f"All backends failed. Original backend '{backend_name}' error: {e}. "
                    f"CPU fallback also failed. Please check your Python environment."
                )
        
        else:
            # CPU itself failed - something is very wrong
            raise BackendNotAvailableError(
                f"NumPy backend failed to initialize: {e}. "
                f"Please check your NumPy/SciPy installation."
            )


def benchmark_backends(matrix_size: int = 1000, 
                      operation: str = 'cholesky') -> Dict[str, float]:
    """Benchmark all available backends for a specific operation."""
    available_backends = get_available_backends()
    results = {}
    
    for backend_name in available_backends:
        try:
            backend = get_backend(backend_name)
            time_taken = backend.benchmark_operation(operation, matrix_size)
            results[backend_name] = time_taken
        except Exception as e:
            results[backend_name] = float('inf')
            warnings.warn(f"Backend '{backend_name}' failed benchmarking: {e}")
    
    return results


def print_backend_summary():
    """Print a summary of available backends and their capabilities."""
    print("PyMVNMLE Backend Summary")
    print("=" * 50)
    print("UPDATED: Optimized for finite difference computation")
    
    available = get_available_backends()
    
    if not available:
        print("❌ No backends available!")
        return
    
    for name in ['numpy', 'cupy', 'metal', 'jax']:
        if name in available:
            backend = get_backend(name)
            info = backend.get_device_info()
            device_type = info.get('device_type', 'unknown')
            
            if name == 'numpy':
                blas_lib = info.get('blas_info', {}).get('library', 'unknown')
                cpu_count = info.get('processor_count', '?')
                print(f"✅ {name:6s} - CPU ({cpu_count} cores, BLAS: {blas_lib})")
                print(f"          Excellent for finite differences")
            
            elif name == 'cupy':
                gpu_count = info.get('device_count', '?')
                devices = info.get('devices', [])
                if devices and len(devices) > 0:
                    memory_gb = devices[0]['memory_gb']
                    print(f"✅ {name:6s} - NVIDIA GPU ({gpu_count} devices, {memory_gb}GB)")
                else:
                    print(f"✅ {name:6s} - NVIDIA GPU ({gpu_count} devices)")
                print(f"          Good for large problems with finite differences")
            
            elif name == 'metal':
                print(f"✅ {name:6s} - Apple Silicon GPU (unified memory)")
                print(f"          Efficient for medium-large problems")
            
            elif name == 'jax':
                device_count = info.get('device_count', '?')
                print(f"✅ {name:6s} - {device_type.upper()} ({device_count} devices)")
                print(f"          XLA optimization helps finite differences")
        
        else:
            if name == 'numpy':
                print(f"❌ {name:6s} - Not available (this should never happen!)")
            elif name == 'cupy':
                print(f"⚪ {name:6s} - NVIDIA GPU (install: pip install cupy)")
            elif name == 'metal':
                print(f"⚪ {name:6s} - Apple Silicon GPU (install: pip install torch)")
            elif name == 'jax':
                print(f"⚪ {name:6s} - TPU/GPU (install: pip install jax)")
    
    print()
    
    # Show auto-selection for different problem sizes
    small_backend = select_optimal_backend((50, 5), available)
    medium_backend = select_optimal_backend((1000, 50), available) 
    large_backend = select_optimal_backend((5000, 100), available)
    
    print("Auto-selection for finite differences:")
    print(f"  Small  (50×5):     {small_backend}")
    print(f"  Medium (1000×50):  {medium_backend}")  
    print(f"  Large  (5000×100): {large_backend}")
    print()
    print("NOTE: Selection optimized for finite difference performance")
    print("CPU often preferred due to reduced GPU transfer overhead")


# Initialize the module
_import_optional_backends()