"""
Backend registry and auto-selection for PyMVNMLE
Handles detection, initialization, and intelligent selection of computational backends
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
    """
    Import optional backends with graceful fallback.
    
    This function attempts to import GPU backends (CuPy, Metal, JAX)
    and registers them if available. If imports fail, they're silently
    skipped and the backend is marked as unavailable.
    """
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


def _check_cupy_compatibility() -> dict:
    """
    Check CuPy compatibility with available GPUs.
    
    Returns
    -------
    dict
        Compatibility information:
        - 'available': bool - Whether CuPy can be imported
        - 'gpu_compatible': bool - Whether CuPy works with detected GPUs
        - 'reason': str - Why CuPy might not work
        - 'compute_caps': list - Compute capabilities of detected GPUs
    """
    try:
        import cupy as cp
        
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            return {
                'available': False,
                'gpu_compatible': False, 
                'reason': 'No CUDA devices detected',
                'compute_caps': []
            }
        
        # Check compute capabilities
        compute_caps = []
        incompatible_gpus = []
        
        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            compute_cap = float(f"{props['major']}.{props['minor']}")
            gpu_name = props['name'].decode()
            compute_caps.append(compute_cap)
            
            # Test basic operation on this GPU
            try:
                with cp.cuda.Device(i):
                    # Test if linear algebra works
                    test_matrix = cp.eye(5, dtype=cp.float32)
                    _ = cp.linalg.cholesky(test_matrix)
            except Exception as e:
                if "no kernel image" in str(e) or "no binary for gpu" in str(e).lower():
                    incompatible_gpus.append((gpu_name, compute_cap))
        
        if incompatible_gpus:
            gpu_list = ", ".join([f"{name} (CC {cc})" for name, cc in incompatible_gpus])
            return {
                'available': True,
                'gpu_compatible': False,
                'reason': f'GPU compute capability not supported: {gpu_list}',
                'compute_caps': compute_caps
            }
        
        return {
            'available': True,
            'gpu_compatible': True,
            'reason': 'CuPy compatible with all detected GPUs',
            'compute_caps': compute_caps
        }
        
    except ImportError:
        return {
            'available': False,
            'gpu_compatible': False,
            'reason': 'CuPy not installed',
            'compute_caps': []
        }
    except Exception as e:
        return {
            'available': False,
            'gpu_compatible': False,
            'reason': f'CuPy initialization failed: {e}',
            'compute_caps': []
        }


def get_available_backends() -> List[str]:
    """
    Get list of backends available on this system.
    
    Returns
    -------
    List[str]
        Names of backends that can be instantiated successfully
        
    Notes
    -----
    This function actually attempts to instantiate each backend to verify
    it works, not just that the library is importable. For example, JAX
    might be importable but fail if no compatible hardware is detected.
    """
    # Import optional backends (may add to _BACKEND_CLASSES)
    _import_optional_backends()
    
    available = ['numpy']  # Always available
    
    # Check CuPy with compatibility testing
    cupy_info = _check_cupy_compatibility()
    if cupy_info['gpu_compatible']:
        available.append('cupy')
    
    # Try other backends
    for name, backend_class in _BACKEND_CLASSES.items():
        if name in ['numpy', 'cupy']:
            continue  # Already handled above
            
        try:
            # Try to actually instantiate the backend
            backend = backend_class()
            if backend.is_available:
                available.append(name)
        except Exception:
            # Backend failed to initialize, skip it
            continue
    
    return available


def get_backend_info() -> Dict[str, Dict]:
    """
    Get detailed information about all available backends.
    
    Returns
    -------
    Dict[str, Dict]
        Mapping from backend name to device information
        
    Examples
    --------
    >>> info = get_backend_info()
    >>> print(info['numpy']['device_type'])  # 'cpu'
    >>> print(info['cupy']['device_type'])   # 'cuda' (if available)
    """
    info = {}
    available_backends = get_available_backends()
    
    for name in available_backends:
        backend = get_backend(name)
        info[name] = backend.get_device_info()
    
    return info


def select_optimal_backend(data_shape: Tuple[int, int], 
                          available_backends: Optional[List[str]] = None) -> str:
    """
    Intelligently select the optimal backend based on problem characteristics.
    
    Parameters
    ----------
    data_shape : Tuple[int, int]
        Shape of data matrix (n_observations, n_variables)
    available_backends : List[str], optional
        Backends to consider. If None, uses all available backends.
        
    Returns
    -------
    str
        Name of the optimal backend for this problem
        
    Notes
    -----
    Selection logic:
    1. Small problems (n≤100, p≤10): CPU (avoid GPU overhead)
    2. Medium problems (n≤1000, p≤50): Prefer GPU if available
    3. Large problems (n>1000, p>50): Strongly prefer GPU
    4. Among GPUs: 
       - JAX preferred for compute capability ≥ 12.0 (newer GPUs)
       - CuPy preferred for compute capability < 12.0 (older GPUs)
       - Metal for Apple Silicon
    5. For very large problems (p>200): JAX may be better due to memory management
    """
    if available_backends is None:
        available_backends = get_available_backends()
    
    n_obs, n_vars = data_shape
    
    # Small problems: CPU overhead is minimal, GPU overhead significant
    if n_vars <= 10 or n_obs <= 100:
        return 'numpy'
    
    # For GPU selection, check what's actually compatible
    gpu_preference = _get_gpu_backend_preference()
    
    # Medium problems: GPU starts to pay off
    if n_vars <= 50 and n_obs <= 1000:
        # Use intelligent GPU preference
        for backend in gpu_preference:
            if backend in available_backends:
                return backend
        return 'numpy'
    
    # Large problems: GPU strongly preferred
    elif n_vars > 50 or n_obs > 1000:
        # For very large problems, prefer JAX > CuPy > Metal
        if n_vars > 200 and 'jax' in available_backends:
            return 'jax'  # JAX best for massive problems
        
        # Otherwise use intelligent GPU preference
        for backend in gpu_preference:
            if backend in available_backends:
                return backend
        return 'numpy'
    
    return 'numpy'  # Safe fallback


def _get_gpu_backend_preference() -> List[str]:
    """
    Get ordered preference for GPU backends based on hardware compatibility.
    
    Returns
    -------
    List[str]
        Ordered list of preferred GPU backends
    """
    # Check if we have NVIDIA GPU and its compute capability
    cupy_info = _check_cupy_compatibility()
    
    if cupy_info['available'] and cupy_info['compute_caps']:
        max_compute_cap = max(cupy_info['compute_caps'])
        
        if max_compute_cap >= 12.0:
            # Newer GPU (RTX 5000 series) - prefer JAX over CuPy
            if cupy_info['gpu_compatible']:
                # CuPy works on this new GPU
                return ['cupy', 'jax', 'metal']
            else:
                # CuPy doesn't work on this new GPU, prefer JAX
                return ['jax', 'metal', 'cupy']
        else:
            # Older GPU (RTX 4000 and below) - CuPy should work well
            return ['cupy', 'jax', 'metal']
    
    # No NVIDIA GPU detected, or CuPy unavailable
    return ['jax', 'metal', 'cupy']


def get_backend(backend_name: Union[str, BackendInterface]) -> BackendInterface:
    """
    Get a backend instance by name or return existing instance.
    
    Parameters
    ----------
    backend_name : str or BackendInterface
        Name of backend ('numpy', 'cupy', 'metal', 'jax', 'auto') 
        or existing backend instance
        
    Returns
    -------
    BackendInterface
        Initialized backend instance
        
    Raises
    ------
    BackendNotAvailableError
        If requested backend is not available on this system
    ValueError
        If backend_name is not recognized
        
    Examples
    --------
    >>> backend = get_backend('numpy')
    >>> backend = get_backend('auto')  # Intelligent selection
    >>> backend = get_backend(existing_backend)  # Returns same instance
    """
    # If already a backend instance, return it
    if isinstance(backend_name, BackendInterface):
        return backend_name
    
    # Handle auto-selection
    if backend_name == 'auto':
        # For auto-selection, we need data shape, so this will be called
        # from higher-level functions with shape information
        available = get_available_backends()
        if not available:
            raise BackendNotAvailableError("No backends available")
        # Default to first available (numpy should always be available)
        backend_name = available[0]
    
    # Validate backend name
    if backend_name not in ['numpy', 'cupy', 'metal', 'jax']:
        raise ValueError(f"Unknown backend: {backend_name}. "
                        f"Available: {list(_BACKEND_CLASSES.keys())}")
    
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
    
    Parameters
    ----------
    backend_name : str
        Requested backend name or 'auto'
    data_shape : Tuple[int, int], optional
        Shape for auto-selection. Required if backend_name is 'auto'.
    verbose : bool, default=False
        Whether to print fallback messages
        
    Returns
    -------
    BackendInterface
        Backend instance (may be different from requested if fallback occurred)
        
    Notes
    -----
    Fallback strategy:
    1. If user specified exact backend and it fails → informative error
    2. If auto-selected backend fails → silent fallback to CPU with warning
    3. If CPU fallback also fails → detailed error with system info
    """
    original_backend_name = backend_name
    
    # Handle auto-selection
    if backend_name == 'auto':
        if data_shape is None:
            raise ValueError("data_shape required for backend='auto'")
        
        available = get_available_backends()
        backend_name = select_optimal_backend(data_shape, available)
        
        if verbose:
            n_obs, n_vars = data_shape
            print(f"Auto-selected '{backend_name}' backend for dataset ({n_obs}×{n_vars})")
    
    # Try to get the requested/selected backend
    try:
        return get_backend(backend_name)
        
    except BackendNotAvailableError as e:
        # Handle fallback logic
        if original_backend_name != 'auto' and backend_name != 'numpy':
            # User explicitly requested this backend - give them helpful error
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
    """
    Benchmark all available backends for a specific operation.
    
    Parameters
    ----------
    matrix_size : int, default=1000
        Size of test matrices for benchmarking
    operation : str, default='cholesky'
        Operation to benchmark ('cholesky', 'matmul', 'inv', 'slogdet')
        
    Returns
    -------
    Dict[str, float]
        Mapping from backend name to execution time in seconds
        
    Examples
    --------
    >>> times = benchmark_backends(matrix_size=2000, operation='cholesky')
    >>> print(f"NumPy: {times['numpy']:.3f}s")
    >>> if 'cupy' in times:
    ...     speedup = times['numpy'] / times['cupy']
    ...     print(f"CuPy speedup: {speedup:.1f}x")
    """
    available_backends = get_available_backends()
    results = {}
    
    for backend_name in available_backends:
        try:
            backend = get_backend(backend_name)
            time_taken = backend.benchmark_operation(operation, matrix_size)
            results[backend_name] = time_taken
        except Exception as e:
            # Backend failed during benchmarking
            results[backend_name] = float('inf')  # Mark as failed
            warnings.warn(f"Backend '{backend_name}' failed benchmarking: {e}")
    
    return results


def print_backend_summary():
    """Print a summary of available backends and their capabilities."""
    print("PyMVNMLE Backend Summary")
    print("=" * 50)
    
    available = get_available_backends()
    backend_info = get_backend_info()
    
    if not available:
        print("❌ No backends available!")
        return
    
    # Get CuPy compatibility info for better messaging
    cupy_info = _check_cupy_compatibility()
    
    for name in ['numpy', 'cupy', 'metal', 'jax']:
        if name in available:
            info = backend_info[name]
            device_type = info.get('device_type', 'unknown')
            
            if name == 'numpy':
                blas_lib = info.get('blas_info', {}).get('library', 'unknown')
                cpu_count = info.get('processor_count', '?')
                print(f"✅ {name:6s} - CPU ({cpu_count} cores, BLAS: {blas_lib})")
            
            elif name == 'cupy':
                gpu_count = info.get('device_count', '?')
                total_memory = info.get('total_memory_gb', '?')
                devices = info.get('devices', [])
                if devices and len(devices) > 0:
                    gpu_name = devices[0]['name']
                    memory_gb = devices[0]['memory_gb']
                    print(f"✅ {name:6s} - NVIDIA GPU ({gpu_count} devices, {memory_gb}GB)")
                else:
                    print(f"✅ {name:6s} - NVIDIA GPU ({gpu_count} devices, {total_memory}GB)")
            
            elif name == 'metal':
                print(f"✅ {name:6s} - Apple Silicon GPU (unified memory)")
            
            elif name == 'jax':
                device_count = info.get('device_count', '?')
                print(f"✅ {name:6s} - {device_type.upper()} ({device_count} devices)")
        
        else:
            if name == 'numpy':
                print(f"❌ {name:6s} - Not available (this should never happen!)")
            elif name == 'cupy':
                if cupy_info['available']:
                    print(f"⚠️  {name:6s} - NVIDIA GPU ({cupy_info['reason']})")
                else:
                    print(f"⚪ {name:6s} - NVIDIA GPU (install: pip install cupy)")
            elif name == 'metal':
                print(f"⚪ {name:6s} - Apple Silicon GPU (install: pip install torch)")
            elif name == 'jax':
                print(f"⚪ {name:6s} - TPU/GPU (install: pip install jax)")
    
    print()
    
    # Show what would be auto-selected for different problem sizes
    small_backend = select_optimal_backend((50, 5), available)
    medium_backend = select_optimal_backend((1000, 50), available) 
    large_backend = select_optimal_backend((5000, 100), available)
    
    print("Auto-selection for different problem sizes:")
    print(f"  Small  (50×5):     {small_backend}")
    print(f"  Medium (1000×50):  {medium_backend}")  
    print(f"  Large  (5000×100): {large_backend}")
    
    # Show GPU preference reasoning if relevant
    if cupy_info['available'] and cupy_info['compute_caps']:
        max_cc = max(cupy_info['compute_caps'])
        print(f"\nGPU Detection: Compute capability {max_cc}")
        if max_cc >= 12.0 and not cupy_info['gpu_compatible']:
            print("  → Preferring JAX over CuPy for this GPU generation")


# Initialize the module by trying to import optional backends
_import_optional_backends()