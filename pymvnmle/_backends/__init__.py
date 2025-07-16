"""
Backend registry for PyMVNMLE v2.0
Simple backend registration and selection - no bloat

DESIGN PRINCIPLE: Minimal functionality to get core system working
- Register available backends
- Simple selection logic
- Basic error handling
- Nothing else

Advanced features (diagnostics, benchmarking, complex selection) come later.
"""

from typing import List, Optional, Tuple
from .base import BackendInterface, BackendNotAvailableError


# Global registry of available backend classes
_BACKEND_CLASSES = {}

# Global cache of initialized backends
_BACKEND_INSTANCES = {}


def _register_numpy_backend():
    """Register the always-available NumPy backend."""
    try:
        from .numpy_backend import NumPyBackend
        _BACKEND_CLASSES['numpy'] = NumPyBackend
    except ImportError:
        # This should never happen since NumPy is a hard dependency
        raise RuntimeError("Critical error: NumPy backend unavailable")


def _register_optional_backends():
    """Register optional backends that may not be available."""
    # PyTorch backend (revolutionary autodiff)
    try:
        from .pytorch_backend import PyTorchBackend
        _BACKEND_CLASSES['pytorch'] = PyTorchBackend
    except ImportError:
        pass  # PyTorch not available
    
    # JAX backend (for completeness)
    try:
        from .jax_backend import JAXBackend
        _BACKEND_CLASSES['jax'] = JAXBackend
    except ImportError:
        pass  # JAX not available


def get_available_backends() -> List[str]:
    """
    Get list of backends available on this system.
    
    Returns
    -------
    List[str]
        Names of available backends, with 'numpy' always first
    """
    # Ensure backends are registered
    _register_numpy_backend()
    _register_optional_backends()
    
    available = []
    
    # Test each backend for actual availability
    for name, backend_class in _BACKEND_CLASSES.items():
        try:
            backend = backend_class()
            if backend.is_available:
                available.append(name)
        except Exception:
            # Backend failed initialization - skip it
            continue
    
    # Ensure numpy is always first (most reliable)
    if 'numpy' in available:
        available.remove('numpy')
        available.insert(0, 'numpy')
    
    return available


def select_backend(backend_name: str, data_shape: Optional[Tuple[int, int]] = None) -> str:
    """
    Simple backend selection logic.
    
    Parameters
    ----------
    backend_name : str
        Requested backend name or 'auto'
    data_shape : tuple of int, optional
        Shape of data matrix for auto-selection
        
    Returns
    -------
    str
        Name of selected backend
        
    Notes
    -----
    Auto-selection logic is conservative:
    - Small problems: Always use 'numpy' (reliable)
    - Large problems: Use GPU if available, otherwise 'numpy'
    - Always default to 'numpy' for safety
    """
    if backend_name != 'auto':
        return backend_name
    
    # Auto-selection: conservative approach
    available = get_available_backends()
    
    if data_shape is None:
        return available[0]  # Default to first available (should be 'numpy')
    
    n_obs, n_vars = data_shape
    
    # Small problems: CPU is optimal (avoid GPU overhead)
    if n_vars <= 20 or n_obs <= 500:
        return 'numpy'
    
    # Large problems: Use GPU if available
    if 'pytorch' in available:
        return 'pytorch'
    elif 'jax' in available:
        return 'jax'
    
    # Fallback to CPU
    return 'numpy'


def get_backend(backend_name: str) -> BackendInterface:
    """
    Get a backend instance by name.
    
    Parameters
    ----------
    backend_name : str
        Name of backend to retrieve
        
    Returns
    -------
    BackendInterface
        Initialized backend instance
        
    Raises
    ------
    BackendNotAvailableError
        If requested backend is not available
    """
    # Handle aliases
    aliases = {
        'cpu': 'numpy',
        'gpu': 'pytorch',
    }
    actual_name = aliases.get(backend_name, backend_name)
    
    # Check if backend is already cached
    if actual_name in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[actual_name]
    
    # Ensure backends are registered
    _register_numpy_backend()
    _register_optional_backends()
    
    # Check if backend class is available
    if actual_name not in _BACKEND_CLASSES:
        available = get_available_backends()
        raise BackendNotAvailableError(
            f"Backend '{actual_name}' is not available. "
            f"Available backends: {available}"
        )
    
    # Instantiate and validate backend
    try:
        backend_class = _BACKEND_CLASSES[actual_name]
        backend = backend_class()
        
        if not backend.is_available:
            raise BackendNotAvailableError(
                f"Backend '{actual_name}' failed availability check"
            )
        
        # Cache the instance
        _BACKEND_INSTANCES[actual_name] = backend
        return backend
        
    except Exception as e:
        if isinstance(e, BackendNotAvailableError):
            raise
        else:
            raise BackendNotAvailableError(
                f"Failed to initialize backend '{actual_name}': {e}"
            )


def get_backend_with_fallback(backend_name: str, 
                             data_shape: Optional[Tuple[int, int]] = None,
                             **kwargs) -> BackendInterface:
    """
    Get backend with automatic fallback to CPU if requested backend fails.
    
    Parameters
    ----------
    backend_name : str
        Requested backend name ('auto', 'numpy', 'pytorch', 'jax')
    data_shape : tuple of int, optional
        Shape of data matrix for auto-selection
    **kwargs
        Additional arguments (ignored for now - extensibility)
        
    Returns
    -------
    BackendInterface
        Functional backend instance
        
    Raises
    ------
    BackendNotAvailableError
        If no functional backends are available (critical system error)
    """
    # Handle auto-selection
    if backend_name == 'auto':
        backend_name = select_backend('auto', data_shape)
    
    # Try to get the requested backend
    try:
        return get_backend(backend_name)
        
    except BackendNotAvailableError:
        # If requested backend failed and it's not numpy, try numpy fallback
        if backend_name != 'numpy':
            try:
                return get_backend('numpy')
            except BackendNotAvailableError:
                # Even numpy failed - critical system error
                raise BackendNotAvailableError(
                    f"All backends failed. Requested '{backend_name}' failed, "
                    f"and NumPy fallback also failed. Check your environment."
                )
        else:
            # NumPy itself failed - re-raise original error
            raise


# Export the essential functions only
__all__ = [
    'BackendInterface', 'BackendNotAvailableError',
    'get_backend', 'get_backend_with_fallback', 'get_available_backends'
]