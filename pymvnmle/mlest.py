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
from pymvnmle._objectives import get_objective

# Import precision detector
from pymvnmle._backends.precision_detector import detect_gpu_capabilities

# Import backend selector
from pymvnmle._backends import get_backend

# Import data structures
from pymvnmle.data_structures import MLResult

# Import pattern analysis
from pymvnmle.patterns import analyze_patterns


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
        (selected_backend, selected_method, backend_obj)
    """
    backend_lower = backend.lower()
    
    # CRITICAL FIX: Auto should default to CPU for R compatibility
    if backend_lower == 'auto':
        if verbose:
            print("Backend 'auto': defaulting to CPU for R compatibility")
        selected_backend = 'cpu'
        backend_obj = None  # Not needed for scipy
        
    elif backend_lower in ['cpu', 'numpy', 'r']:
        selected_backend = 'cpu'
        backend_obj = None
        
    elif backend_lower in ['gpu', 'cuda', 'metal', 'pytorch']:
        # Check if GPU is available
        try:
            gpu_caps = detect_gpu_capabilities()
            if gpu_caps.has_gpu:
                selected_backend = 'gpu_fp64' if gpu64 else 'gpu_fp32'
                backend_obj = None
                if verbose:
                    print(f"GPU backend selected: {selected_backend}")
                    print(f"Device: {gpu_caps.device_name}")
                    print("Note: GPU uses different parameterization (standard Cholesky)")
            else:
                warnings.warn("GPU requested but not available, using CPU")
                selected_backend = 'cpu'
                backend_obj = None
        except:
            warnings.warn("Could not detect GPU, using CPU")
            selected_backend = 'cpu'
            backend_obj = None
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
    
    return selected_backend, selected_method, backend_obj

# Import scipy optimizer wrappers  
from pymvnmle._scipy_optimizers import (
    optimize_with_scipy,
    validate_method,
    auto_select_method
)


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
        print(f"Complete cases: {patterns[0].n_cases if patterns and len(patterns[0].missing_indices) == 0 else 0}/{n_obs}")
    
    # Step 3: Select backend and method
    selected_backend, selected_method, backend_obj = _backend_method_selection(
        backend, method, gpu64, verbose
    )
    
    # Step 4: Create objective function based on backend
    try:
        # Determine which objective to use based on backend
        backend_name = backend_obj.name if hasattr(backend_obj, 'name') else 'numpy_fp64'
        
        if 'pytorch' in backend_name or 'gpu' in backend_name:
            # GPU backend - determine precision
            precision = backend_obj.precision if hasattr(backend_obj, 'precision') else 'fp32'
            
            if 'gpu' in selected_backend:
                precision = 'fp64' if 'fp64' in selected_backend else 'fp32'
                objective = get_objective(data, backend='gpu', precision=precision)
            else:
                objective = get_objective(data, backend='cpu')

            if verbose:
                print(f"Using GPU objective with {precision} precision")
                
        else:
            # CPU backend
            from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
            objective = CPUObjectiveFP64(data)
            
            if verbose:
                print("Using CPU objective with fp64 precision")
                
    except ImportError as e:
        # Fallback to CPU if GPU objective not available
        if verbose:
            print(f"GPU objective not available: {e}, falling back to CPU")
        from pymvnmle._objectives.cpu_fp64_objective import CPUObjectiveFP64
        objective = CPUObjectiveFP64(data)
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
        if result['n_jev']:
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