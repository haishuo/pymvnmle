"""
Objective functions for MLE computation.

Provides different objective implementations optimized for various
backends and precision levels.
"""

from typing import Optional, Union
import numpy as np
import warnings

# Import base class
from .base import MLEObjectiveBase, PatternData

# Import parameterizations
from .parameterizations import (
    CovarianceParameterization,
    InverseCholeskyParameterization,
    CholeskyParameterization,
    MatrixLogParameterization,
    get_parameterization,
    convert_parameters
)

# Import CPU objective (always available)
from .cpu_fp64_objective import CPUObjectiveFP64

# MODIFIED: Lazy import GPU objectives to avoid PyTorch initialization
# Try to import GPU objectives (optional)
GPU_AVAILABLE = False
GPUObjectiveFP32 = None
GPUObjectiveFP64 = None

def _lazy_import_gpu():
    """Lazily import GPU objectives only when needed."""
    global GPU_AVAILABLE, GPUObjectiveFP32, GPUObjectiveFP64
    
    if GPU_AVAILABLE:
        return True  # Already imported
    
    try:
        # Only import when actually needed
        from .gpu_fp32_objective import GPUObjectiveFP32 as _FP32
        from .gpu_fp64_objective import GPUObjectiveFP64 as _FP64
        GPUObjectiveFP32 = _FP32
        GPUObjectiveFP64 = _FP64
        GPU_AVAILABLE = True
        return True
    except ImportError:
        return False


def get_objective(data: np.ndarray,
                 backend: str = 'cpu',
                 precision: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs) -> MLEObjectiveBase:
    """
    Factory function to create appropriate objective.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_vars)
        Input data with missing values as np.nan
    backend : str
        Backend type: 'cpu', 'gpu', 'numpy', 'pytorch'
    precision : str or None
        Precision: 'fp32', 'fp64', or None for auto
    device : str or None
        Device specification for GPU backends
    **kwargs
        Additional backend-specific options
        
    Returns
    -------
    MLEObjectiveBase
        Objective function instance
        
    Raises
    ------
    ValueError
        If backend/precision combination is invalid
    ImportError
        If GPU backend requested but PyTorch not available
    """
    backend = backend.lower()
    
    # Handle backend aliases
    if backend in ['numpy', 'cpu']:
        return CPUObjectiveFP64(data, **kwargs)
    
    elif backend in ['gpu', 'pytorch', 'cuda', 'metal']:
        # MODIFIED: Lazy import GPU objectives
        if not _lazy_import_gpu():
            raise ImportError(
                "GPU objectives require PyTorch. "
                "Install with: pip install torch"
            )
        
        # Determine precision if not specified
        if precision is None:
            # Auto-select based on hardware
            precision = _auto_select_precision(device)
        
        precision = precision.lower()
        
        if precision == 'fp32':
            return GPUObjectiveFP32(data, device=device, **kwargs)
        elif precision == 'fp64':
            return GPUObjectiveFP64(data, device=device, **kwargs)
        else:
            raise ValueError(f"Invalid precision: {precision}")
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose from: 'cpu', 'gpu', 'numpy', 'pytorch'"
        )


def _auto_select_precision(device: Optional[str] = None) -> str:
    """
    Auto-select precision based on device capabilities.
    """
    # MODIFIED: Import torch only when needed
    import torch
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if device == 'cuda':
        # Check compute capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Datacenter GPUs (V100, A100, etc) have good FP64
            if capability[0] >= 7:
                return 'fp64'
    
    # Default to FP32 for consumer GPUs and Apple Silicon
    return 'fp32'


def create_objective(data: np.ndarray,
                    use_gpu: bool = False,
                    use_fp64: bool = True,
                    device: Optional[str] = None,
                    compile_enabled: bool = True,
                    **kwargs) -> MLEObjectiveBase:
    """
    Simplified interface for creating objectives.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with missing values
    use_gpu : bool
        Whether to use GPU acceleration
    use_fp64 : bool
        Whether to use 64-bit precision (if False, uses 32-bit)
    device : str or None
        Specific device to use ('cuda', 'mps', etc)
    compile_enabled : bool
        Whether to compile GPU objectives with torch.compile
    **kwargs
        Additional backend-specific options
        
    Returns
    -------
    MLEObjectiveBase
        Objective function instance
        
    Examples
    --------
    >>> # CPU objective
    >>> obj = create_objective(data, use_gpu=False)
    >>> 
    >>> # GPU with FP32
    >>> obj = create_objective(data, use_gpu=True, use_fp64=False)
    """
    if use_gpu:
        # MODIFIED: Check GPU availability lazily
        if not _lazy_import_gpu():
            warnings.warn(
                "GPU requested but PyTorch not available. Falling back to CPU.",
                RuntimeWarning
            )
            return CPUObjectiveFP64(data, **kwargs)
        
        precision = 'fp64' if use_fp64 else 'fp32'
        return get_objective(
            data, 
            backend='gpu', 
            precision=precision,
            device=device,
            compile_enabled=compile_enabled,
            **kwargs
        )
    else:
        return CPUObjectiveFP64(data, **kwargs)


def compare_objectives(data: np.ndarray,
                      theta: np.ndarray,
                      backends: Optional[list] = None) -> dict:
    """
    Compare different objective implementations.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    theta : np.ndarray
        Parameter vector
    backends : list
        List of backends to compare
        
    Returns
    -------
    dict
        Comparison results
        
    Notes
    -----
    Useful for debugging and verifying consistency across implementations.
    """
    results = {}
    
    for backend in backends:
        try:
            if backend == 'cpu':
                obj = CPUObjectiveFP64(data)
            elif backend == 'gpu_fp32':
                # MODIFIED: Lazy import check
                if _lazy_import_gpu():
                    obj = GPUObjectiveFP32(data)
                else:
                    continue
            elif backend == 'gpu_fp64':
                # MODIFIED: Lazy import check
                if _lazy_import_gpu():
                    obj = GPUObjectiveFP64(data)
                else:
                    continue
            else:
                continue
            
            # Compute objective value
            obj_value = obj.compute_objective(theta)
            
            # Extract parameters
            mu, sigma, loglik = obj.extract_parameters(theta)
            
            results[backend] = {
                'objective': obj_value,
                'loglik': loglik,
                'mu': mu,
                'sigma': sigma,
                'device_info': obj.get_device_info() if hasattr(obj, 'get_device_info') else {}
            }
            
        except Exception as e:
            results[backend] = {'error': str(e)}
    
    return results


def benchmark_objectives(data: np.ndarray,
                        n_iterations: int = 10) -> dict:
    """
    Benchmark different objective implementations.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    n_iterations : int
        Number of iterations for timing
        
    Returns
    -------
    dict
        Benchmark results
    """
    import time
    
    results = {}
    
    # Test each backend
    backends = [
        ('cpu', CPUObjectiveFP64),
    ]
    
    # MODIFIED: Only add GPU if successfully imported
    if _lazy_import_gpu():
        backends.extend([
            ('gpu_fp32', GPUObjectiveFP32),
            ('gpu_fp64', GPUObjectiveFP64),
        ])
    
    for name, obj_class in backends:
        try:
            # Create objective
            obj = obj_class(data)
            theta = obj.get_initial_parameters()
            
            # Warm-up
            _ = obj.compute_objective(theta)
            _ = obj.compute_gradient(theta)
            
            # Time objective evaluation
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = obj.compute_objective(theta)
            obj_time = (time.perf_counter() - start) / n_iterations
            
            # Time gradient evaluation
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = obj.compute_gradient(theta)
            grad_time = (time.perf_counter() - start) / n_iterations
            
            results[name] = {
                'objective_time': obj_time,
                'gradient_time': grad_time,
                'total_time': obj_time + grad_time,
                'speedup': 1.0  # Will calculate relative to CPU
            }
            
            # Add device info for GPU backends
            if hasattr(obj, 'get_device_info'):
                results[name]['device_info'] = obj.get_device_info()
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Calculate speedups relative to CPU
    if 'cpu' in results and 'objective_time' in results['cpu']:
        cpu_total = results['cpu']['total_time']
        for name in results:
            if 'total_time' in results[name]:
                results[name]['speedup'] = cpu_total / results[name]['total_time']
    
    return results


# Convenience function for testing
def get_test_objective(data: np.ndarray) -> MLEObjectiveBase:
    """
    Get objective suitable for testing.
    
    Always returns CPU objective for deterministic testing.
    """
    return CPUObjectiveFP64(data)


__all__ = [
    # Base classes
    'MLEObjectiveBase',
    'PatternData',
    
    # Parameterizations
    'CovarianceParameterization',
    'InverseCholeskyParameterization',
    'CholeskyParameterization',
    'MatrixLogParameterization',
    'get_parameterization',
    'convert_parameters',
    
    # Objectives
    'CPUObjectiveFP64',
    # MODIFIED: Don't export GPU objectives directly to avoid import
    # 'GPUObjectiveFP32',
    # 'GPUObjectiveFP64',
    
    # Factory functions
    'get_objective',
    'create_objective',
    'compare_objectives',
    'benchmark_objectives',
    'get_test_objective',
    
    # Constants
    'GPU_AVAILABLE'
]