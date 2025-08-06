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

# Try to import GPU objectives (optional)
try:
    from .gpu_fp32_objective import GPUObjectiveFP32
    from .gpu_fp64_objective import GPUObjectiveFP64
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUObjectiveFP32 = None
    GPUObjectiveFP64 = None


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
        if not GPU_AVAILABLE:
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
            f"Available: 'cpu', 'gpu', 'numpy', 'pytorch'"
        )


def _auto_select_precision(device: Optional[str]) -> str:
    """
    Auto-select precision based on hardware capabilities.
    
    Parameters
    ----------
    device : str or None
        Device specification
        
    Returns
    -------
    str
        'fp32' or 'fp64'
    """
    # Import here to avoid circular dependency
    from pymvnmle._backends.precision_detector import detect_gpu_capabilities
    
    caps = detect_gpu_capabilities()
    
    # If Metal or gimped FP64, use FP32
    if caps.gpu_type == 'metal' or not caps.recommended_fp64:
        return 'fp32'
    
    # If full FP64 support, use it
    return 'fp64'


def create_objective(data: np.ndarray,
                    use_fp64: bool = True,
                    use_gpu: bool = False,
                    **kwargs) -> MLEObjectiveBase:
    """
    Create objective with simple boolean flags.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    use_fp64 : bool
        Whether to use FP64 precision
    use_gpu : bool
        Whether to use GPU acceleration
    **kwargs
        Additional options
        
    Returns
    -------
    MLEObjectiveBase
        Objective instance
    """
    if not use_gpu:
        return CPUObjectiveFP64(data, **kwargs)
    
    if not GPU_AVAILABLE:
        warnings.warn(
            "GPU requested but PyTorch not available, falling back to CPU"
        )
        return CPUObjectiveFP64(data, **kwargs)
    
    if use_fp64:
        try:
            return GPUObjectiveFP64(data, **kwargs)
        except RuntimeError as e:
            warnings.warn(f"FP64 GPU failed: {e}. Falling back to FP32.")
            return GPUObjectiveFP32(data, **kwargs)
    else:
        return GPUObjectiveFP32(data, **kwargs)


def compare_objectives(data: np.ndarray,
                      theta: np.ndarray,
                      backends: list = ['cpu', 'gpu']) -> dict:
    """
    Compare objective values across different backends.
    
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
            elif backend == 'gpu_fp32' and GPU_AVAILABLE:
                obj = GPUObjectiveFP32(data)
            elif backend == 'gpu_fp64' and GPU_AVAILABLE:
                obj = GPUObjectiveFP64(data)
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
    
    if GPU_AVAILABLE:
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
    'GPUObjectiveFP32',
    'GPUObjectiveFP64',
    
    # Factory functions
    'get_objective',
    'create_objective',
    'compare_objectives',
    'benchmark_objectives',
    'get_test_objective',
    
    # Constants
    'GPU_AVAILABLE'
]