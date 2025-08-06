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
    
    CRITICAL: 'auto' and default always select CPU.
    GPU must be explicitly requested with backend='gpu'.
    
    Parameters
    ----------
    backend : str
        Requested backend ('cpu', 'gpu', or legacy 'auto')
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
    
    # Import precision detector only if GPU is explicitly requested
    gpu_caps = None
    has_gpu = False
    
    # Backend selection logic - SIMPLE AND EXPLICIT
    if backend_lower in ['auto', 'cpu', 'numpy', 'r']:
        # Auto, CPU, and all CPU aliases always use CPU
        selected_backend = 'cpu'
        if verbose and backend_lower == 'auto':
            print("Backend 'auto': Using CPU (default)")
        elif verbose:
            print(f"Backend '{backend}': Using CPU")
            
    elif backend_lower in ['gpu', 'cuda', 'metal', 'pytorch']:
        # GPU explicitly requested - check availability
        try:
            from pymvnmle._backends.precision_detector import detect_gpu_capabilities
            gpu_caps = detect_gpu_capabilities()
            has_gpu = gpu_caps.has_gpu
        except:
            has_gpu = False
        
        if has_gpu:
            selected_backend = 'gpu_fp64' if gpu64 else 'gpu_fp32'
            if verbose:
                print(f"GPU backend selected: {selected_backend}")
                if gpu_caps:
                    if hasattr(gpu_caps, 'device_name'):
                        print(f"Device: {gpu_caps.device_name}")
                    else:
                        print(f"Device: CUDA GPU detected")
                    if gpu64:
                        if hasattr(gpu_caps, 'fp64_ratio') and gpu_caps.fp64_ratio and gpu_caps.fp64_ratio < 1.0:
                            print(f"WARNING: FP64 performance ratio: {gpu_caps.fp64_ratio:.1%} of FP32")
        else:
            warnings.warn(
                f"GPU backend '{backend}' requested but no GPU detected, falling back to CPU",
                RuntimeWarning
            )
            selected_backend = 'cpu'
            
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cpu' or 'gpu'")
    
    # Method selection based on backend
    method_upper = method.upper() if method else 'AUTO'
    
    if method_upper == 'AUTO':
        # Always use BFGS for now (most robust)
        selected_method = 'BFGS'
    else:
        selected_method = method
    
    if verbose:
        print(f"Selected backend: {selected_backend}")
        print(f"Selected method: {selected_method}")
    
    return selected_backend, selected_method


def mlest(
    data: np.ndarray,
    backend: str = 'cpu',  # Changed default from 'auto' to 'cpu'
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
    backend : {'cpu', 'gpu'}
        Computational backend. Default is 'cpu'. GPU must be explicitly requested.
        Legacy value 'auto' is supported and maps to 'cpu'.
    gpu64 : bool
        Force FP64 precision on GPU. May be very slow on consumer GPUs
    method : {'auto', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'Nelder-Mead', 'Powell'}
        Optimization method. Uses scipy.optimize.minimize
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    verbose : bool
        Print optimization progress
        
    Returns
    -------
    MLResult
        Result object containing parameter estimates and diagnostics
        
    Raises
    ------
    ValueError
        If input data is invalid or optimization configuration is incompatible
    RuntimeError
        If optimization fails
    """
    start_time = time.time()
    
    # Step 1: Validate and prepare input data
    data = _validate_input(data)
    n_obs, n_vars = data.shape
    
    if verbose:
        print("=" * 60)
        print("Maximum Likelihood Estimation for MVN with Missing Data")
        print("-" * 60)
        print(f"Data shape: {n_obs} observations, {n_vars} variables")
        print(f"Missing values: {np.isnan(data).sum()} ({np.isnan(data).mean():.1%})")
    
    # Step 2: Analyze missingness patterns
    patterns = analyze_patterns(data)
    
    if verbose:
        print(f"Missingness patterns: {len(patterns)} unique patterns")
        complete_cases = sum(1 for p in patterns if len(p.missing_indices) == 0)
        if patterns and len(patterns[0].missing_indices) == 0:
            print(f"Complete cases: {patterns[0].n_cases}/{n_obs}")
        else:
            print(f"Complete cases: 0/{n_obs}")
    
    # Step 3: Select backend and method
    selected_backend, selected_method = _backend_method_selection(
        backend, method, gpu64, verbose
    )
    
    # Step 4: Create objective function based on backend
    from pymvnmle._objectives import get_objective
    
    try:
        # Map backend selection to objective parameters
        if selected_backend == 'cpu':
            objective = get_objective(data, backend='cpu')
        elif selected_backend == 'gpu_fp32':
            objective = get_objective(data, backend='gpu', precision='fp32')
        elif selected_backend == 'gpu_fp64':
            objective = get_objective(data, backend='gpu', precision='fp64')
        else:
            raise ValueError(f"Unknown backend: {selected_backend}")
            
        if verbose:
            device_info = objective.get_device_info() if hasattr(objective, 'get_device_info') else {}
            if device_info:
                print(f"Objective created on: {device_info.get('device', 'unknown')}")
                if 'gpu_name' in device_info:
                    print(f"GPU: {device_info['gpu_name']}")
                    
    except ImportError as e:
        if 'gpu' in selected_backend:
            warnings.warn(
                f"Failed to create GPU objective: {e}. Falling back to CPU.",
                RuntimeWarning
            )
            selected_backend = 'cpu'
            objective = get_objective(data, backend='cpu')
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"Failed to create objective function: {e}")
    
    if verbose:
        print("-" * 60)
        print(f"Backend: {selected_backend}")
        print(f"Method: {selected_method}")
        print(f"Tolerance: {tol:.2e}")
        print(f"Max iterations: {max_iter}")
        print("-" * 60)
    
    # Step 5: Get initial parameters
    x0 = objective.get_initial_parameters()
    
    # Step 6: Run scipy optimization
    if verbose:
        print("Starting optimization...")
    
    # Check for Hessian availability
    has_hessian = hasattr(objective, 'compute_hessian')
    
    # Validate method
    try:
        validated_method = validate_method(selected_method, selected_backend, has_hessian)
    except ValueError as e:
        # Auto-select if validation fails
        if verbose:
            print(f"Method validation failed: {e}")
            print("Auto-selecting appropriate method...")
        validated_method = auto_select_method(
            backend=selected_backend,
            has_hessian=has_hessian,
            problem_size=(n_obs, n_vars),
            precision='fp64' if selected_backend == 'cpu' else 'fp32'
        )
        if verbose:
            print(f"Selected method: {validated_method}")
    
    # Run optimization
    try:
        result = optimize_with_scipy(
            objective_fn=objective.compute_objective,
            gradient_fn=objective.compute_gradient,
            hessian_fn=objective.compute_hessian if has_hessian else None,
            x0=x0,
            method=validated_method,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}")
    
    # Step 7: Extract final parameters
    mu_final, sigma_final, _ = objective.extract_parameters(result['x'])
    
    # Compute final log-likelihood (negative of objective / 2 for R convention)
    final_loglik = -result['fun'] / 2.0
    
    # Total computation time
    computation_time = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"Optimization Complete:")
        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['n_iter']}")
        print(f"  Function evaluations: {result['n_fev']}")
        if result.get('n_jev'):
            print(f"  Gradient evaluations: {result['n_jev']}")
        print(f"  Final gradient norm: {result['grad_norm']:.2e}")
        print(f"  Log-likelihood: {final_loglik:.6f}")
        print(f"  Computation time: {computation_time:.3f}s")
        print("-" * 60)
    
    # Step 8: Create and return result object
    return MLResult(
        muhat=mu_final,
        sigmahat=sigma_final,
        loglik=final_loglik,
        n_iter=result['n_iter'],
        converged=result['converged'],
        computation_time=computation_time,
        backend=selected_backend,
        method=validated_method,
        patterns=patterns,
        n_obs=n_obs,
        n_vars=n_vars,
        n_missing=np.isnan(data).sum(),
        grad_norm=result['grad_norm'],
        message=result['message']
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
    
    # Check for all missing in any variable
    for j in range(n_vars):
        if np.isnan(data[:, j]).all():
            raise ValueError(f"Variable {j} has all missing values")
    
    # Check for at least one complete case (for initial values)
    complete_cases = ~np.isnan(data).any(axis=1)
    if not complete_cases.any():
        warnings.warn(
            "No complete cases found. Initial values may be poor.",
            RuntimeWarning
        )
    
    return data