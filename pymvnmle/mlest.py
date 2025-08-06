"""
Maximum likelihood estimation for multivariate normal data with missing values.

This module provides the main entry point for PyMVNMLE, implementing the
mlest() function that performs maximum likelihood estimation using the
precision-based architecture.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import warnings
import time
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np

# Import from new architecture
from ._backends import get_backend
from ._objectives import get_objective
from ._methods import get_optimizer

# Import PrecisionDetector
try:
    from ._backends.precision_detector import detect_gpu_capabilities
except ImportError:
    # Fallback if function has different name
    def detect_gpu_capabilities():
        return {
            'has_gpu': False,
            'gpu_type': 'none',
            'fp64_support': 'none',
            'device_name': 'None'
        }

# Import data structures
from .data_structures import MLResult
from .patterns import analyze_patterns


def mlest(
    data: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str = 'auto',
    backend: str = 'auto',
    gpu64: bool = False,
    verbose: bool = False
) -> MLResult:
    """
    Maximum likelihood estimation for multivariate normal data with missing values.
    
    This function computes maximum likelihood estimates of the mean vector and
    covariance matrix for multivariate normal data with missing values using
    the expectation-maximization (EM) algorithm or direct optimization.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_observations, n_variables).
        Missing values must be represented as np.nan.
        
    max_iter : int, default=1000
        Maximum number of optimization iterations.
        
    tol : float, default=1e-6
        Convergence tolerance for optimization.
        
    method : str, default='auto'
        Optimization algorithm:
        - 'auto': Automatically select based on backend
        - 'BFGS': Broyden-Fletcher-Goldfarb-Shanno (good for FP32)
        - 'Newton-CG': Newton conjugate gradient (requires FP64)
        - 'L-BFGS-B': Limited memory BFGS with bounds
        - 'Nelder-Mead': Gradient-free simplex method
        
    backend : str, default='auto'
        Computational backend:
        - 'auto': Automatically select based on hardware
        - 'cpu': Force CPU computation (exact R compatibility)
        - 'gpu': Force GPU computation (if available)
        
    gpu64 : bool, default=False
        If True and GPU is available, force FP64 precision on GPU.
        Will fail with appropriate message if:
        - No GPU available (falls back to CPU)
        - GPU doesn't support FP64 (falls back to FP32)
        - GPU has gimped FP64 (proceeds with warning)
        
    verbose : bool, default=False
        Whether to print optimization progress and debugging info.
        
    Returns
    -------
    MLResult
        Result object containing:
        - muhat: Estimated mean vector
        - sigmahat: Estimated covariance matrix
        - loglik: Log-likelihood at convergence
        - n_iter: Number of iterations
        - converged: Whether optimization converged
        - computation_time: Total computation time
        - backend: Backend used
        - method: Optimization method used
        - patterns: Missing data patterns
        
    Raises
    ------
    ValueError
        If input data is invalid or optimization fails
        
    Examples
    --------
    >>> import numpy as np
    >>> from pymvnmle import mlest
    >>> 
    >>> # Generate data with missing values
    >>> np.random.seed(42)
    >>> data = np.random.randn(100, 3)
    >>> data[np.random.rand(100, 3) < 0.2] = np.nan
    >>> 
    >>> # Estimate parameters
    >>> result = mlest(data)
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Mean: {result.muhat}")
    >>> print(f"Log-likelihood: {result.loglik:.2f}")
    """
    # Start timing
    start_time = time.time()
    
    # Input validation
    data = _validate_input(data)
    n_obs, n_vars = data.shape
    
    if verbose:
        print(f"PyMVNMLE Maximum Likelihood Estimation")
        print(f"Data: {n_obs} observations, {n_vars} variables")
        print(f"Missing: {np.isnan(data).sum()} values ({np.isnan(data).mean()*100:.1f}%)")
        print("-" * 60)
    
    # Analyze missing data patterns
    patterns = analyze_patterns(data)
    
    # Check if patterns is a dict or list and handle accordingly
    if isinstance(patterns, dict):
        n_patterns = len(patterns.get('pattern_indices', []))
        pattern_indices = patterns.get('pattern_indices', [])
        observed_vars = patterns.get('observed_variables', [])
    else:
        # patterns might be a list of pattern info
        n_patterns = len(patterns) if patterns else 0
        pattern_indices = None
        observed_vars = None
        # Convert to dict format for consistency
        patterns = {'patterns': patterns}
    
    if verbose:
        print(f"Missing data patterns: {n_patterns}")
        if pattern_indices and observed_vars:
            for i, (idx, obs_vars) in enumerate(zip(pattern_indices, observed_vars)):
                n_obs_pattern = len(idx)
                n_vars_pattern = len(obs_vars)
                print(f"  Pattern {i+1}: {n_obs_pattern} obs, {n_vars_pattern}/{n_vars} vars observed")
    
    # Step 1: Determine backend based on hardware and user preferences
    backend_type, precision = _determine_backend(
        backend=backend,
        gpu64=gpu64,
        data_size=(n_obs, n_vars),
        verbose=verbose
    )
    
    # Step 2: Create backend using the get_backend function
    # The get_backend function handles the complexity of backend selection
    use_fp64 = (precision == 'fp64')
    backend_obj = get_backend(backend=backend_type, use_fp64=use_fp64)
    
    if verbose:
        print(f"\nBackend Configuration:")
        print(f"  Type: {backend_type}")
        print(f"  Precision: {precision}")
        print(f"  Device: {backend_obj.device}")
        print("-" * 60)
    
    # Step 3: Create objective function
    # get_objective expects 'backend' as a parameter name, not 'backend_type'
    if backend_type == 'cpu':
        objective_backend = 'cpu'
    else:
        objective_backend = 'gpu'
    
    objective = get_objective(
        data=data,
        backend=objective_backend,
        precision=precision
    )
    
    # Check if objective provides Hessian (for method selection)
    has_hessian = hasattr(objective, 'hessian') and callable(objective.hessian)
    
    # Step 4: Select and create optimizer
    from ._methods import auto_select_method
    method_name, optimizer, opt_config = auto_select_method(
        backend_type=backend_type,
        precision=precision,
        problem_size=(n_obs, n_vars),
        has_hessian=has_hessian,
        user_preference=method if method != 'auto' else None,
        max_iter=max_iter,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nOptimization Configuration:")
        print(f"  Method: {method_name}")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {tol:.2e}")
        print("-" * 60)
        print("\nStarting optimization...")
    
    # Step 5: Run optimization
    try:
        # Get initial parameters
        x0 = objective.get_initial_params()
        
        # Define objective and gradient functions for optimizer
        def obj_fn(x):
            return objective.compute_objective(x)
        
        def grad_fn(x):
            return objective.compute_gradient(x)
        
        # Add Hessian if available and using Newton-CG
        if method_name == 'Newton-CG' and has_hessian:
            def hess_fn(x):
                return objective.compute_hessian(x)
            
            result = optimizer.optimize(
                objective_fn=obj_fn,
                gradient_fn=grad_fn,
                hessian_fn=hess_fn,
                x0=x0
            )
        else:
            result = optimizer.optimize(
                objective_fn=obj_fn,
                gradient_fn=grad_fn,
                x0=x0
            )
        
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}")
    
    # Step 6: Extract final parameters
    mu_final, sigma_final = objective.extract_parameters(result['x'])
    
    # Compute final log-likelihood (negative of objective for minimization)
    final_loglik = -result['fun']
    
    # Total computation time
    computation_time = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"Optimization Complete:")
        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['n_iter']}")
        print(f"  Final gradient norm: {result['grad_norm']:.2e}")
        print(f"  Log-likelihood: {final_loglik:.6f}")
        print(f"  Computation time: {computation_time:.3f}s")
        print("-" * 60)
    
    # Step 7: Create and return result object
    return MLResult(
        muhat=mu_final,
        sigmahat=sigma_final,
        loglik=final_loglik,
        n_iter=result['n_iter'],
        converged=result['converged'],
        computation_time=computation_time,
        backend=backend_type,
        method=method_name,
        patterns=patterns,
        n_obs=n_obs,
        n_vars=n_vars,
        n_missing=np.isnan(data).sum(),
        grad_norm=result['grad_norm'],
        message=result.get('message', '')
    )


def _validate_input(data: np.ndarray) -> np.ndarray:
    """
    Validate and prepare input data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix
        
    Returns
    -------
    np.ndarray
        Validated data as float64 array
        
    Raises
    ------
    ValueError
        If data is invalid
    """
    # Convert to numpy array if needed
    data = np.asarray(data, dtype=np.float64)
    
    # Check dimensions
    if data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got shape {data.shape}")
    
    n_obs, n_vars = data.shape
    
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")
    
    if n_vars < 1:
        raise ValueError(f"Need at least 1 variable, got {n_vars}")
    
    # Check for all missing
    if np.isnan(data).all():
        raise ValueError("All data values are missing")
    
    # Check for variables with all missing
    all_missing_vars = np.isnan(data).all(axis=0)
    if all_missing_vars.any():
        bad_vars = np.where(all_missing_vars)[0]
        raise ValueError(f"Variables {bad_vars} have all missing values")
    
    # Check for observations with all missing
    all_missing_obs = np.isnan(data).all(axis=1)
    if all_missing_obs.any():
        bad_obs = np.where(all_missing_obs)[0]
        raise ValueError(f"Observations {bad_obs} have all missing values")
    
    return data


def _determine_backend(
    backend: str,
    gpu64: bool,
    data_size: Tuple[int, int],
    verbose: bool
) -> Tuple[str, str]:
    """
    Determine the backend and precision to use.
    
    Parameters
    ----------
    backend : str
        User-requested backend ('auto', 'cpu', 'gpu')
    gpu64 : bool
        Whether to force FP64 on GPU
    data_size : tuple
        (n_observations, n_variables)
    verbose : bool
        Print decision process
        
    Returns
    -------
    backend_type : str
        Selected backend ('cpu', 'gpu', 'auto')
    precision : str
        Selected precision ('fp32' or 'fp64')
    """
    # Get GPU capabilities
    gpu_info = detect_gpu_capabilities()
    
    # Handle both dict and object returns
    if hasattr(gpu_info, '__dict__'):
        # It's an object (GPUCapabilities), convert to dict-like access
        has_gpu = getattr(gpu_info, 'has_gpu', False)
        gpu_type = getattr(gpu_info, 'gpu_type', 'none')
        fp64_support = getattr(gpu_info, 'fp64_support', 'none')
        device_name = getattr(gpu_info, 'device_name', 'None')
        fp64_ratio = getattr(gpu_info, 'fp64_ratio', None)
    else:
        # It's a dict
        has_gpu = gpu_info.get('has_gpu', False)
        gpu_type = gpu_info.get('gpu_type', 'none')
        fp64_support = gpu_info.get('fp64_support', 'none')
        device_name = gpu_info.get('device_name', 'None')
        fp64_ratio = gpu_info.get('fp64_ratio', None)
    
    # If user explicitly wants CPU
    if backend == 'cpu':
        if verbose and gpu64:
            print("Note: gpu64=True ignored with backend='cpu'")
        return 'cpu', 'fp64'
    
    # If no GPU available
    if not has_gpu or gpu_type == 'none':
        if backend == 'gpu':
            warnings.warn(
                "GPU requested but no GPU detected. Falling back to CPU.",
                RuntimeWarning
            )
        if gpu64:
            warnings.warn(
                "gpu64=True but no GPU detected. Using CPU with FP64.",
                RuntimeWarning
            )
        return 'cpu', 'fp64'
    
    # GPU is available - determine precision
    if gpu64:
        # User explicitly wants FP64 on GPU
        if fp64_support == 'none':
            # GPU doesn't support FP64 at all (e.g., Metal)
            warnings.warn(
                f"gpu64=True but GPU doesn't support FP64. "
                f"Falling back to FP32 with BFGS.",
                RuntimeWarning
            )
            return 'gpu', 'fp32'
        
        elif fp64_support == 'gimped':
            # GPU has severely limited FP64 (consumer cards)
            warnings.warn(
                f"gpu64=True on {device_name} with gimped FP64 "
                f"(1/{fp64_ratio or 32}x speed). "
                f"This will be MUCH slower than FP32. Consider gpu64=False.",
                RuntimeWarning
            )
            return 'gpu', 'fp64'
        
        else:  # fp64_support == 'full'
            # Data center GPU with good FP64
            if verbose:
                print(f"Using {device_name} with full FP64 support")
            return 'gpu', 'fp64'
    
    else:
        # Default: Use FP32 on GPU (fastest for consumer hardware)
        if backend == 'auto':
            # For auto mode, check problem size
            n_obs, n_vars = data_size
            
            # Small problems might be faster on CPU
            if n_obs < 100 and n_vars < 10:
                if verbose:
                    print(f"Small problem (n={n_obs}, p={n_vars}), using CPU")
                return 'cpu', 'fp64'
        
        # Use GPU FP32 for larger problems
        if verbose:
            print(f"Using {device_name} with FP32 (fastest for consumer GPUs)")
        return 'gpu', 'fp32'


# Backward compatibility aliases
def ml_estimate(data: np.ndarray, **kwargs) -> MLResult:
    """Alias for mlest() for backward compatibility."""
    return mlest(data, **kwargs)


def maximum_likelihood_estimate(data: np.ndarray, **kwargs) -> MLResult:
    """Alias for mlest() for backward compatibility."""
    return mlest(data, **kwargs)