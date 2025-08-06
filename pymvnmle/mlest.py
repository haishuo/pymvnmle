"""
Maximum likelihood estimation for multivariate normal distributions with missing data.

This module uses scipy's battle-tested optimization algorithms rather than
custom implementations. This approach prioritizes correctness and reproducibility
for regulatory submissions.

Author: Senior Biostatistician
Date: January 2025
"""

import numpy as np
import warnings
import time
from typing import Optional, Tuple, Union

# Import precision detector
from pymvnmle._backends.precision_detector import detect_gpu_capabilities

# Import data structures
from pymvnmle.data_structures import MLResult

# Import pattern analysis
from pymvnmle.patterns import analyze_patterns

# Import scipy optimizer wrappers  
from pymvnmle._scipy_optimizers import (
    optimize_with_scipy,
    validate_method,
    auto_select_method
)


def _backend_method_selection(
    backend: str,
    method: str,
    gpu64: bool,
    verbose: bool
) -> tuple:
    """
    Select backend and method based on hardware and user preferences.
    
    CRITICAL: backend='auto' defaults to 'cpu' for R compatibility.
    GPU must be explicitly requested.
    
    Parameters
    ----------
    backend : str
        Requested backend ('auto', 'cpu', 'gpu')
    method : str
        Requested method ('auto', 'BFGS', etc.)
    gpu64 : bool
        Force FP64 on GPU
    verbose : bool
        Print selection info
        
    Returns
    -------
    tuple
        (selected_backend, selected_method)
    """
    backend_lower = backend.lower()
    
    # CRITICAL: Auto defaults to CPU for R compatibility
    if backend_lower == 'auto':
        if verbose:
            print("Backend 'auto': defaulting to CPU for R compatibility")
        selected_backend = 'cpu'
        
    elif backend_lower in ['cpu', 'numpy', 'r']:
        selected_backend = 'cpu'
        
    elif backend_lower in ['gpu', 'cuda', 'metal', 'pytorch']:
        # Check if GPU is available
        try:
            gpu_caps = detect_gpu_capabilities()
            if gpu_caps.has_gpu:
                selected_backend = 'gpu_fp64' if gpu64 else 'gpu_fp32'
                if verbose:
                    print(f"GPU backend selected: {selected_backend}")
                    print(f"Device: {gpu_caps.device_name}")
            else:
                warnings.warn("GPU requested but not available, using CPU")
                selected_backend = 'cpu'
        except:
            warnings.warn("Could not detect GPU, using CPU")
            selected_backend = 'cpu'
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Method selection based on backend
    method_upper = method.upper() if method else 'AUTO'
    
    if method_upper == 'AUTO':
        if selected_backend == 'cpu':
            selected_method = 'BFGS'
        elif 'gpu' in selected_backend:
            # For GPU, could use Newton-CG if Hessian available
            selected_method = 'BFGS'  # Safe default
        else:
            selected_method = 'BFGS'
    else:
        # Use requested method
        selected_method = method
    
    if verbose:
        print(f"Selected backend: {selected_backend}")
        print(f"Selected method: {selected_method}")
    
    return selected_backend, selected_method


def mlest(
    data: np.ndarray,
    backend: str = 'auto',
    gpu64: bool = False,
    method: str = 'auto',
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False
) -> MLResult:
    """
    Maximum likelihood estimation for multivariate normal with missing data.
    
    Uses scipy.optimize for robust, validated optimization algorithms.
    Prioritizes numerical correctness and regulatory compliance over 
    custom implementations.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_vars)
        Data matrix with missing values as np.nan
    backend : {'auto', 'cpu', 'gpu'}
        Computational backend. 'auto' selects optimal based on hardware
    gpu64 : bool
        Force FP64 precision on GPU. May be very slow on consumer GPUs
    method : {'auto', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'Nelder-Mead', 'Powell'}
        Optimization method. 'auto' selects based on backend
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    verbose : bool
        Print progress information
        
    Returns
    -------
    MLResult
        Result object containing estimates and diagnostics
    """
    # Start timing
    start_time = time.time()
    
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float64)
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D array, got shape {data.shape}")
    
    n_obs, n_vars = data.shape
    
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")
    
    if n_vars < 1:
        raise ValueError(f"Need at least 1 variable, got {n_vars}")
    
    # Check for missing data
    n_missing = np.sum(np.isnan(data))
    missing_rate = n_missing / data.size
    
    # Analyze patterns
    patterns = analyze_patterns(data)
    n_patterns = len(patterns['pattern_counts'])
    n_complete = patterns['n_complete_cases']
    
    if verbose:
        print("=" * 60)
        print("Maximum Likelihood Estimation for MVN with Missing Data")
        print("-" * 60)
        print(f"Data shape: {n_obs} observations, {n_vars} variables")
        print(f"Missing values: {n_missing} ({missing_rate*100:.1f}%)")
        print(f"Missingness patterns: {n_patterns} unique patterns")
        print(f"Complete cases: {n_complete}/{n_obs}")
    
    # Select backend and method
    selected_backend, selected_method = _backend_method_selection(
        backend, method, gpu64, verbose
    )
    
    # FIXED: Create objective using factory function
    try:
        from pymvnmle._objectives import get_objective
        
        if 'gpu' in selected_backend:
            # Determine precision from backend name
            if 'fp64' in selected_backend:
                objective = get_objective(data, backend='gpu', precision='fp64')
                if verbose:
                    print("Using GPU objective with fp64 precision")
            else:
                objective = get_objective(data, backend='gpu', precision='fp32')
                if verbose:
                    print("Using GPU objective with fp32 precision")
                    
            # Verify GPU is actually being used
            if hasattr(objective, 'get_device_info'):
                device_info = objective.get_device_info()
                if verbose:
                    print(f"GPU device: {device_info.get('gpu_name', 'Unknown')}")
                    mem_allocated = device_info.get('memory_allocated', 0)
                    if mem_allocated > 0:
                        print(f"GPU memory allocated: {mem_allocated / 1024**2:.1f} MB")
        else:
            # CPU backend
            objective = get_objective(data, backend='cpu')
            if verbose:
                print("Using CPU objective with fp64 precision")
                
    except ImportError as e:
        # Fallback to CPU if GPU objective not available
        if verbose:
            print(f"GPU objective not available: {e}, falling back to CPU")
        from pymvnmle._objectives import get_objective
        objective = get_objective(data, backend='cpu')
        selected_backend = 'cpu'  # Update for result
    except Exception as e:
        raise RuntimeError(f"Failed to create objective: {e}")
    
    if verbose:
        print("-" * 60)
        print(f"Backend: {selected_backend}")
        print(f"Method: {selected_method}")
        print(f"Tolerance: {tol:.2e}")
        print(f"Max iterations: {max_iter}")
        print("-" * 60)
    
    # Get initial parameters
    theta_init = objective.get_initial_parameters()
    
    # Run optimization
    if verbose:
        print("Starting optimization...")
    
    result = optimize_with_scipy(
        objective=objective,
        theta_init=theta_init,
        method=selected_method,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # Extract final parameters
    mu_final, sigma_final, loglik = objective.extract_parameters(result.x)
    
    # Compute final gradient for diagnostics
    try:
        grad_final = objective.compute_gradient(result.x)
        grad_norm = np.linalg.norm(grad_final)
    except:
        grad_final = result.jac if hasattr(result, 'jac') else None
        grad_norm = np.linalg.norm(grad_final) if grad_final is not None else np.nan
    
    # Total computation time
    computation_time = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print("Optimization Complete:")
        print(f"  Converged: {result.success}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")
        if hasattr(result, 'njev'):
            print(f"  Gradient evaluations: {result.njev}")
        print(f"  Final gradient norm: {grad_norm:.2e}")
        print(f"  Log-likelihood: {loglik:.6f}")
        print(f"  Computation time: {computation_time:.3f}s")
        print("-" * 60)
    
    # Create result object
    ml_result = MLResult(
        muhat=mu_final,
        sigmahat=sigma_final,
        loglik=loglik,
        n_iter=result.nit,
        converged=result.success,
        computation_time=computation_time,
        backend=selected_backend,
        method=selected_method,
        patterns=patterns,
        n_obs=n_obs,
        n_vars=n_vars,
        n_missing=n_missing,
        grad_norm=grad_norm,
        message=result.message if hasattr(result, 'message') else ""
    )
    
    return ml_result


# Backward compatibility aliases
ml_estimate = mlest
maximum_likelihood_estimate = mlest


def run_validation():
    """Quick validation against reference datasets."""
    from pymvnmle import datasets
    
    print("\n" + "="*70)
    print("PYMVNMLE VALIDATION")
    print("="*70)
    
    # Test Apple dataset
    print("\n📊 Apple Dataset:")
    result = mlest(datasets.apple, verbose=False)
    print(f"  Log-likelihood: {result.loglik:.6f}")
    print(f"  Converged: {result.converged}")
    print(f"  Backend: {result.backend}")
    
    # Expected R values
    r_loglik = -74.217476
    diff = abs(result.loglik - r_loglik)
    if diff < 1e-5:
        print(f"  ✅ Matches R reference (diff: {diff:.2e})")
    else:
        print(f"  ⚠️ Differs from R reference (diff: {diff:.2e})")
    
    return result


if __name__ == "__main__":
    # Run validation if executed directly
    run_validation()