"""
Backend module initialization.

Provides a unified interface for backend selection and management.
"""

from typing import Optional, Union
import warnings
import numpy as np


# Import base classes
from .base import (
    BackendBase,
    CPUBackend,
    GPUBackendFP32,
    GPUBackendFP64,
    BackendFactory
)

# Import concrete implementations
from .cpu_fp64_backend import NumpyBackendFP64

# Try to import GPU backends (optional)
try:
    from .gpu_fp32_backend import PyTorchBackendFP32
    from .gpu_fp64_backend import PyTorchBackendFP64
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchBackendFP32 = None
    PyTorchBackendFP64 = None


# Maintain backward compatibility with old names
BackendInterface = BackendBase  # Alias for compatibility
GPUBackendBase = GPUBackendFP32  # Alias for compatibility


class BackendNotAvailableError(ImportError):
    """Raised when a requested backend is not available."""
    pass


def get_backend(backend: str = 'auto', 
                use_fp64: Optional[bool] = None,
                **kwargs) -> BackendBase:
    """
    Get computational backend.
    
    Parameters
    ----------
    backend : str
        Backend name: 'auto', 'cpu', 'gpu', 'numpy', 'pytorch', 'cuda', 'metal'
    use_fp64 : bool or None
        If None, auto-select based on hardware
    **kwargs
        Additional backend-specific options
        
    Returns
    -------
    BackendBase
        Initialized backend instance
        
    Raises
    ------
    BackendNotAvailableError
        If requested backend is not available
    """
    backend = backend.lower()
    
    # Handle backend aliases
    if backend in ['numpy', 'cpu']:
        return NumpyBackendFP64()
    
    elif backend == 'auto':
        return BackendFactory.get_optimal_backend(use_fp64)
    
    elif backend in ['gpu', 'pytorch', 'cuda', 'metal']:
        if not PYTORCH_AVAILABLE:
            raise BackendNotAvailableError(
                "GPU backend requires PyTorch. "
                "Install with: pip install torch"
            )
        
        # Determine device type
        from .precision_detector import detect_gpu_capabilities
        caps = detect_gpu_capabilities()
        
        if not caps.has_gpu:
            warnings.warn(
                "No GPU detected, falling back to CPU backend. "
                "This will be slower than using the native CPU backend."
            )
            device_type = 'cpu'
        else:
            device_type = caps.gpu_type
        
        # Create backend
        return BackendFactory.create_backend(
            use_fp64=use_fp64 if use_fp64 is not None else caps.recommended_fp64,
            device_type=device_type
        )
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: 'auto', 'cpu', 'gpu', 'numpy', 'pytorch'"
        )


def get_backend_with_fallback(backend: str = 'auto',
                              use_fp64: Optional[bool] = None,
                              fallback: str = 'cpu',
                              **kwargs) -> BackendBase:
    """
    Get backend with automatic fallback.
    
    Parameters
    ----------
    backend : str
        Primary backend choice
    use_fp64 : bool or None
        Precision preference
    fallback : str
        Fallback backend if primary fails
    **kwargs
        Backend-specific options
        
    Returns
    -------
    BackendBase
        Initialized backend (primary or fallback)
    """
    try:
        return get_backend(backend, use_fp64, **kwargs)
    except (BackendNotAvailableError, RuntimeError, ImportError) as e:
        warnings.warn(
            f"Backend '{backend}' not available: {e}. "
            f"Falling back to '{fallback}' backend."
        )
        return get_backend(fallback, use_fp64, **kwargs)


def get_available_backends() -> dict:
    """
    Get all available backends and their capabilities.
    
    Returns
    -------
    dict
        Backend information including availability and capabilities
    """
    backends = {
        'numpy': {
            'available': True,
            'backend_class': 'NumpyBackendFP64',
            'precision': 'fp64',
            'autodiff': False,
            'description': 'NumPy CPU backend (always available)'
        },
        'cpu': {
            'available': True,
            'backend_class': 'NumpyBackendFP64',
            'precision': 'fp64',
            'autodiff': False,
            'description': 'NumPy CPU backend (always available)'
        }
    }
    
    if PYTORCH_AVAILABLE:
        import torch
        
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        backends['pytorch'] = {
            'available': True,
            'backend_class': 'PyTorchBackend',
            'precision': 'fp32/fp64',
            'autodiff': True,
            'description': 'PyTorch backend'
        }
        
        if has_cuda:
            backends['cuda'] = {
                'available': True,
                'backend_class': 'PyTorchBackendFP32/FP64',
                'precision': 'fp32/fp64',
                'autodiff': True,
                'device_name': torch.cuda.get_device_name(0),
                'description': 'PyTorch CUDA backend'
            }
        
        if has_mps:
            backends['metal'] = {
                'available': True,
                'backend_class': 'PyTorchBackendFP32',
                'precision': 'fp32',
                'autodiff': True,
                'description': 'PyTorch Metal Performance Shaders backend'
            }
    
    return backends


# Alias for compatibility
list_available_backends = get_available_backends


def list_available_backends() -> dict:
    """
    List all available backends and their capabilities.
    
    Returns
    -------
    dict
        Backend information including availability and capabilities
    """
    backends = {
        'cpu': {
            'available': True,
            'backend_class': 'NumpyBackendFP64',
            'precision': 'fp64',
            'autodiff': False,
            'description': 'NumPy CPU backend (always available)'
        }
    }
    
    if PYTORCH_AVAILABLE:
        import torch
        
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if has_cuda:
            backends['cuda'] = {
                'available': True,
                'backend_class': 'PyTorchBackendFP32/FP64',
                'precision': 'fp32/fp64',
                'autodiff': True,
                'device_name': torch.cuda.get_device_name(0),
                'description': 'PyTorch CUDA backend'
            }
        
        if has_mps:
            backends['metal'] = {
                'available': True,
                'backend_class': 'PyTorchBackendFP32',
                'precision': 'fp32',
                'autodiff': True,
                'description': 'PyTorch Metal Performance Shaders backend'
            }
    
    return backends


def benchmark_backends(test_size: int = 100) -> dict:
    """
    Benchmark available backends.
    
    Parameters
    ----------
    test_size : int
        Size of test matrices
        
    Returns
    -------
    dict
        Benchmark results for each backend
    """
    import time
    
    results = {}
    
    # Test data
    np.random.seed(42)
    A = np.random.randn(test_size, test_size)
    pos_def = A @ A.T + np.eye(test_size)
    
    # Benchmark each available backend
    for backend_name in ['cpu', 'gpu']:
        try:
            backend = get_backend(backend_name)
            
            # Transfer to device
            matrix = backend.to_device(pos_def)
            
            # Benchmark Cholesky
            start = time.perf_counter()
            for _ in range(10):
                L = backend.cholesky(matrix, upper=False)
            elapsed = time.perf_counter() - start
            
            results[backend_name] = {
                'available': True,
                'cholesky_time': elapsed / 10,
                'backend_info': backend.get_device_info() if hasattr(backend, 'get_device_info') else {}
            }
            
        except Exception as e:
            results[backend_name] = {
                'available': False,
                'error': str(e)
            }
    
    return results


# Convenience function for testing
def get_test_backend() -> BackendBase:
    """Get a backend suitable for testing (CPU preferred for determinism)."""
    return NumpyBackendFP64()


__all__ = [
    # Base classes
    'BackendBase',
    'CPUBackend', 
    'GPUBackendFP32',
    'GPUBackendFP64',
    'BackendFactory',
    
    # Concrete implementations
    'NumpyBackendFP64',
    'PyTorchBackendFP32',
    'PyTorchBackendFP64',
    
    # Functions
    'get_backend',
    'get_backend_with_fallback',
    'get_available_backends',  # Added this
    'list_available_backends',
    'benchmark_backends',
    'get_test_backend',
    
    # Exceptions
    'BackendNotAvailableError',
    
    # Compatibility aliases
    'BackendInterface',
    'GPUBackendBase'
]