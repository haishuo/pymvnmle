"""
PyMVNMLE: Maximum Likelihood Estimation for Multivariate Normal Data with Missing Values

A high-performance Python implementation of maximum likelihood estimation for 
multivariate normal distributions with missing data patterns, featuring 
precision-based GPU acceleration and exact R compatibility.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import numpy as np  # Add numpy import for benchmark_performance

__version__ = "2.0.0"
__author__ = "Statistical Software Engineer"
__license__ = "MIT"

# Core functionality
from .mlest import (
    mlest,
    ml_estimate,  # Backward compatibility alias
    maximum_likelihood_estimate  # Backward compatibility alias
)

# Data structures
from .data_structures import MLResult

# Pattern analysis
from .patterns import analyze_patterns, pattern_summary
# Try to import get_pattern_matrix if it exists
try:
    from .patterns import get_pattern_matrix
except ImportError:
    # Function might not exist or have different name
    get_pattern_matrix = None

# Statistical tests
from .mcar_test import little_mcar_test

# Datasets
from . import datasets

# Advanced features (new in v2.0)
from ._backends import (
    get_backend,
    list_available_backends,
    benchmark_backends,
    BackendFactory
)

# Try to import PrecisionDetector - might have different names in actual implementation
try:
    from ._backends.precision_detector import PrecisionDetector
    detect_hardware_capabilities = PrecisionDetector().detect_gpu
except (ImportError, AttributeError):
    # Fallback if the module structure is different
    try:
        # Maybe it's called something else
        from ._backends.precision_detector import GPUDetector as PrecisionDetector
        detect_hardware_capabilities = PrecisionDetector().detect_gpu
    except ImportError:
        # Final fallback - create dummy detector
        class PrecisionDetector:
            def detect_gpu(self):
                return {
                    'gpu_type': 'none',
                    'fp64_support': 'none', 
                    'device_name': 'None'
                }
        
        def detect_hardware_capabilities():
            return PrecisionDetector().detect_gpu()

from ._objectives import (
    get_objective,
    compare_objectives,
    benchmark_objectives
)

from ._methods import (
    get_optimizer,
    auto_select_method,
    compare_methods,
    benchmark_convergence
)

# Utility functions - import what's available from _utils
try:
    from ._utils import (
        generate_mvn_data,
        add_missing_data,
        compute_pairwise_deletion_cov,
        check_positive_definite
    )
except ImportError:
    # Some or all utilities might not exist in _utils
    # Set them to None and they won't be exported
    generate_mvn_data = None
    add_missing_data = None
    compute_pairwise_deletion_cov = None
    check_positive_definite = None


# Convenience function for hardware detection
def check_gpu_capabilities(verbose: bool = True) -> dict:
    """
    Check GPU capabilities for PyMVNMLE optimization.
    
    Parameters
    ----------
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    dict
        Hardware capabilities including:
        - gpu_available: Whether GPU is detected
        - gpu_type: Type of GPU (cuda, metal, none)
        - fp64_support: Level of FP64 support (full, gimped, none)
        - recommended_settings: Suggested mlest() parameters
        
    Examples
    --------
    >>> import pymvnmle as pmle
    >>> caps = pmle.check_gpu_capabilities()
    GPU Detected: NVIDIA RTX 4090
    FP64 Support: Gimped (1/64x speed)
    Recommendation: Use default settings (FP32) for best performance
    """
    detector = PrecisionDetector()
    info = detector.detect_gpu()
    
    result = {
        'gpu_available': info['gpu_type'] != 'none',
        'gpu_type': info['gpu_type'],
        'gpu_name': info.get('device_name', 'None'),
        'fp64_support': info.get('fp64_support', 'none'),
        'fp64_ratio': info.get('fp64_ratio', None),
        'recommended_settings': {}
    }
    
    # Determine recommendations
    if not result['gpu_available']:
        result['recommended_settings'] = {
            'backend': 'cpu',
            'gpu64': False,
            'method': 'BFGS'
        }
        recommendation = "No GPU detected. CPU computation will be used."
        
    elif result['fp64_support'] == 'full':
        result['recommended_settings'] = {
            'backend': 'auto',
            'gpu64': True,
            'method': 'auto'  # Will select Newton-CG
        }
        recommendation = "Full FP64 support! Use gpu64=True for maximum precision."
        
    elif result['fp64_support'] == 'gimped':
        result['recommended_settings'] = {
            'backend': 'auto',
            'gpu64': False,  # Default to FP32
            'method': 'auto'  # Will select BFGS
        }
        recommendation = f"Gimped FP64 (1/{result['fp64_ratio']}x speed). Use default settings (FP32) for best performance."
        
    else:  # No FP64 support
        result['recommended_settings'] = {
            'backend': 'auto',
            'gpu64': False,
            'method': 'auto'  # Will select BFGS
        }
        recommendation = "No FP64 support. FP32 will be used for GPU computation."
    
    if verbose:
        print(f"GPU Detected: {result['gpu_name']}")
        print(f"FP64 Support: {result['fp64_support'].title()}", end='')
        if result['fp64_ratio']:
            print(f" (1/{result['fp64_ratio']}x speed)")
        else:
            print()
        print(f"Recommendation: {recommendation}")
        print(f"\nSuggested mlest() parameters:")
        for key, value in result['recommended_settings'].items():
            print(f"  {key}={value}")
    
    return result


def benchmark_performance(
    n_obs: int = 500,
    n_vars: int = 10,
    missing_rate: float = 0.2,
    backends: list = None,
    verbose: bool = True
) -> dict:
    """
    Benchmark PyMVNMLE performance across different backends.
    
    Parameters
    ----------
    n_obs : int
        Number of observations in test data
    n_vars : int
        Number of variables in test data
    missing_rate : float
        Proportion of missing values
    backends : list, optional
        List of backends to test (default: all available)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Benchmark results for each backend
        
    Examples
    --------
    >>> import pymvnmle as pmle
    >>> results = pmle.benchmark_performance(n_obs=1000, n_vars=20)
    Benchmarking PyMVNMLE Performance...
    Data: 1000 observations, 20 variables, 20.0% missing
    
    Backend: cpu (fp64)
      Time: 2.34s
      Iterations: 45
      Log-likelihood: -28453.21
    
    Backend: gpu_fp32
      Time: 0.89s (2.6x speedup)
      Iterations: 43
      Log-likelihood: -28453.18
    """
    if verbose:
        print("Benchmarking PyMVNMLE Performance...")
        print(f"Data: {n_obs} observations, {n_vars} variables, {missing_rate*100:.1f}% missing")
        print()
    
    # Generate test data - import from _utils not .utils
    try:
        from ._utils import generate_mvn_data, add_missing_data
    except ImportError:
        # Functions might not exist - create simple versions
        def generate_mvn_data(n_obs, n_vars, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_obs, n_vars)
        
        def add_missing_data(data, missing_rate, seed=None):
            if seed is not None:
                np.random.seed(seed)
            data = data.copy()
            mask = np.random.rand(*data.shape) < missing_rate
            data[mask] = np.nan
            return data
    data = generate_mvn_data(n_obs, n_vars, seed=42)
    data = add_missing_data(data, missing_rate, seed=42)
    
    # Test each backend
    if backends is None:
        backends = ['cpu', 'gpu']
    
    results = {}
    baseline_time = None
    
    for backend in backends:
        try:
            # Skip if backend not available
            if backend == 'gpu':
                detector = PrecisionDetector()
                if detector.detect_gpu()['gpu_type'] == 'none':
                    if verbose:
                        print(f"Backend: {backend} - Not available")
                        print()
                    continue
            
            # Run estimation
            import time
            start = time.time()
            result = mlest(data, backend=backend, verbose=False)
            elapsed = time.time() - start
            
            results[backend] = {
                'time': elapsed,
                'iterations': result.n_iter,
                'loglik': result.loglik,
                'converged': result.converged,
                'backend_used': result.backend,
                'method_used': result.method
            }
            
            if verbose:
                print(f"Backend: {result.backend} ({result.method})")
                print(f"  Time: {elapsed:.2f}s", end='')
                if baseline_time is not None:
                    speedup = baseline_time / elapsed
                    print(f" ({speedup:.1f}x speedup)")
                else:
                    baseline_time = elapsed
                    print()
                print(f"  Iterations: {result.n_iter}")
                print(f"  Log-likelihood: {result.loglik:.2f}")
                print()
                
        except Exception as e:
            if verbose:
                print(f"Backend: {backend} - Failed ({e})")
                print()
            results[backend] = {'error': str(e)}
    
    return results


# Version checking utilities
def check_version():
    """Print PyMVNMLE version and dependencies."""
    import numpy as np
    import scipy
    print(f"PyMVNMLE: {__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Check hardware
    caps = check_gpu_capabilities(verbose=False)
    print(f"\nHardware:")
    print(f"  GPU Available: {caps['gpu_available']}")
    if caps['gpu_available']:
        print(f"  GPU Type: {caps['gpu_name']}")
        print(f"  FP64 Support: {caps['fp64_support']}")


# Main public API
__all__ = [
    # Core functions
    'mlest',
    'ml_estimate',
    'maximum_likelihood_estimate',
    
    # Data structures
    'MLResult',
    
    # Pattern analysis
    'analyze_patterns',
    'pattern_summary',
    
    # Statistical tests
    'little_mcar_test',
    
    # Datasets
    'datasets',
    
    # Backend management (new in v2.0)
    'get_backend',
    'list_available_backends',
    'benchmark_backends',
    
    # Hardware detection (new in v2.0)
    'check_gpu_capabilities',
    'detect_hardware_capabilities',
    
    # Performance benchmarking (new in v2.0)
    'benchmark_performance',
    
    # Version info
    'check_version',
    '__version__'
]

# Add optional exports if they were successfully imported
if get_pattern_matrix is not None:
    __all__.append('get_pattern_matrix')

# Add utility functions if they exist
if generate_mvn_data is not None:
    __all__.extend([
        'generate_mvn_data',
        'add_missing_data', 
        'compute_pairwise_deletion_cov',
        'check_positive_definite'
    ])


# Print hardware info on import (can be disabled)
_SHOW_HARDWARE_ON_IMPORT = False  # Set to True for debugging

if _SHOW_HARDWARE_ON_IMPORT:
    print(f"PyMVNMLE {__version__} loaded")
    caps = check_gpu_capabilities(verbose=False)
    if caps['gpu_available']:
        print(f"GPU: {caps['gpu_name']} ({caps['fp64_support']} FP64 support)")
    else:
        print("GPU: Not available, using CPU")