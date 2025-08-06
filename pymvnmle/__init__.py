"""
PyMVNMLE: Maximum Likelihood Estimation for Multivariate Normal Data with Missing Values

A high-performance Python implementation of maximum likelihood estimation for 
multivariate normal distributions with missing data patterns, featuring 
precision-based GPU acceleration and exact R compatibility.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import numpy as np

__version__ = "2.0.0"
__author__ = "Statistical Software Engineer"
__license__ = "MIT"

# Core functionality - ONLY import what exists
from .mlest import mlest

# Data structures
from .data_structures import MLResult

# Pattern analysis
from .patterns import analyze_patterns, pattern_summary

# Try to import get_pattern_matrix if it exists
try:
    from .patterns import get_pattern_matrix
except ImportError:
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

# Try to import PrecisionDetector
try:
    from ._backends.precision_detector import detect_gpu_capabilities
    HAS_GPU_DETECTION = True
except ImportError:
    HAS_GPU_DETECTION = False
    # Create dummy function
    def detect_gpu_capabilities():
        return {
            'gpu_type': 'none',
            'fp64_support': 'none', 
            'device_name': 'None',
            'has_gpu': False
        }

# Import objectives if available
try:
    from ._objectives import (
        get_objective,
        compare_objectives,
        benchmark_objectives
    )
    HAS_OBJECTIVES = True
except ImportError:
    HAS_OBJECTIVES = False
    get_objective = None
    compare_objectives = None
    benchmark_objectives = None

# Import optimizer methods if available
try:
    from ._scipy_optimizers import (
        optimize_with_scipy,
        validate_method,
        auto_select_method
    )
    HAS_OPTIMIZERS = True
except ImportError:
    HAS_OPTIMIZERS = False
    optimize_with_scipy = None
    validate_method = None
    auto_select_method = None

# Utility functions - import what's available from _utils
try:
    from ._utils import (
        generate_mvn_data,
        add_missing_data,
        compute_pairwise_deletion_cov,
        check_positive_definite
    )
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
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
        Print detailed information
        
    Returns
    -------
    dict
        GPU capability information
    """
    caps = detect_gpu_capabilities()
    
    result = {
        'gpu_available': caps.get('has_gpu', False),
        'gpu_name': caps.get('device_name', 'None'),
        'fp64_support': caps.get('fp64_support', 'none')
    }
    
    if verbose:
        print("GPU Capabilities:")
        print(f"  Available: {result['gpu_available']}")
        if result['gpu_available']:
            print(f"  Device: {result['gpu_name']}")
            print(f"  FP64 Support: {result['fp64_support']}")
    
    return result


# Performance benchmarking function
def benchmark_performance(data: np.ndarray, backends: list = None) -> dict:
    """
    Benchmark PyMVNMLE performance across backends.
    
    Parameters
    ----------
    data : np.ndarray
        Test data matrix
    backends : list, optional
        List of backends to test. If None, tests all available
        
    Returns
    -------
    dict
        Performance results for each backend
    """
    import time
    
    if backends is None:
        backends = ['cpu']
        if HAS_GPU_DETECTION and detect_gpu_capabilities().get('has_gpu', False):
            backends.append('gpu')
    
    results = {}
    
    for backend in backends:
        try:
            start = time.time()
            result = mlest(data, backend=backend, verbose=False)
            elapsed = time.time() - start
            
            results[backend] = {
                'time': elapsed,
                'converged': result.converged,
                'iterations': result.n_iter,
                'loglik': result.loglik
            }
        except Exception as e:
            results[backend] = {
                'error': str(e)
            }
    
    return results


# Version checking
def check_version():
    """Print version information for PyMVNMLE and dependencies."""
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


# Main public API - ONLY export what exists
__all__ = [
    # Core functions
    'mlest',
    
    # Data structures
    'MLResult',
    
    # Pattern analysis
    'analyze_patterns',
    'pattern_summary',
    
    # Statistical tests
    'little_mcar_test',
    
    # Datasets
    'datasets',
    
    # Backend management
    'get_backend',
    'list_available_backends',
    'benchmark_backends',
    
    # Hardware detection
    'check_gpu_capabilities',
    
    # Performance benchmarking
    'benchmark_performance',
    
    # Version info
    'check_version',
    '__version__'
]

# Add optional exports if they were successfully imported
if get_pattern_matrix is not None:
    __all__.append('get_pattern_matrix')

# Add utility functions if they exist
if HAS_UTILS:
    utility_exports = []
    if generate_mvn_data is not None:
        utility_exports.append('generate_mvn_data')
    if add_missing_data is not None:
        utility_exports.append('add_missing_data')
    if compute_pairwise_deletion_cov is not None:
        utility_exports.append('compute_pairwise_deletion_cov')
    if check_positive_definite is not None:
        utility_exports.append('check_positive_definite')
    __all__.extend(utility_exports)

# Add objectives if available
if HAS_OBJECTIVES:
    if get_objective is not None:
        __all__.append('get_objective')
    if compare_objectives is not None:
        __all__.append('compare_objectives')
    if benchmark_objectives is not None:
        __all__.append('benchmark_objectives')

# Add optimizer methods if available
if HAS_OPTIMIZERS:
    if optimize_with_scipy is not None:
        __all__.append('optimize_with_scipy')
    if validate_method is not None:
        __all__.append('validate_method')
    if auto_select_method is not None:
        __all__.append('auto_select_method')