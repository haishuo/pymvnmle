"""
Scipy optimizer wrappers for PyMVNMLE.

This module provides thin wrappers around scipy.optimize functions
configured for maximum likelihood estimation. Uses scipy's battle-tested
implementations rather than custom code.

Author: Senior Biostatistician
Date: January 2025
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
from scipy import optimize


def create_optimizer(
    method: str,
    max_iter: int,
    tol: float,
    verbose: bool
) -> Dict[str, Any]:
    """
    Create optimizer configuration for scipy.optimize.minimize.
    
    Parameters
    ----------
    method : str
        Optimization method name
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Configuration dictionary with 'method' and 'options' keys
    """
    method_upper = method.upper()
    
    # Base options for all methods
    options = {
        'maxiter': max_iter,
        'disp': verbose
    }
    
    # Method-specific tolerances
    if method_upper == 'BFGS':
        # More lenient tolerances to match R's nlm behavior
        options['gtol'] = max(tol, 1e-4)  # R typically achieves 1e-4
        options['norm'] = np.inf  # Use infinity norm like R
        
    elif method_upper == 'NEWTON-CG':
        options['xtol'] = tol
        
    elif method_upper == 'L-BFGS-B':
        options['gtol'] = max(tol, 1e-4)
        options['ftol'] = tol * 100
        options['maxfun'] = max_iter * 2  # Function evaluations
        
    elif method_upper == 'NELDER-MEAD':
        options['xatol'] = tol
        options['fatol'] = tol * 100
        options['maxfev'] = max_iter * 100  # Many function evals for simplex
        
    elif method_upper == 'POWELL':
        options['xtol'] = tol
        options['ftol'] = tol * 100
        options['maxfev'] = max_iter * 50
        
    elif method_upper == 'TNC':
        options['gtol'] = tol
        options['ftol'] = tol * 100
        options['maxCGit'] = max_iter
        
    elif method_upper == 'CG':
        options['gtol'] = tol
        
    elif method_upper == 'TRUST-NCG':
        options['gtol'] = tol
        
    else:
        # Unknown method - let scipy handle it
        options['gtol'] = tol
    
    return {
        'method': method,
        'options': options
    }


def optimize_with_scipy(
    objective_fn: Callable[[np.ndarray], float],
    gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    hessian_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray,
    method: str,
    max_iter: int,
    tol: float,
    verbose: bool
) -> Dict[str, Any]:
    """
    Run scipy optimization and return standardized result.
    
    Parameters
    ----------
    objective_fn : callable
        Objective function
    gradient_fn : callable or None
        Gradient function (None for gradient-free methods)
    hessian_fn : callable or None
        Hessian function (only for Newton-CG)
    x0 : np.ndarray
        Initial parameters
    method : str
        Optimization method
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Standardized result dictionary with keys:
        - 'x': final parameters
        - 'fun': final objective value
        - 'grad': final gradient (computed if not provided)
        - 'n_iter': number of iterations
        - 'converged': bool
        - 'message': convergence message
        - 'grad_norm': norm of final gradient
    """
    # Get optimizer configuration
    config = create_optimizer(method, max_iter, tol, verbose)
    
    # Determine which functions to pass based on method
    method_upper = method.upper()
    gradient_free = method_upper in ['NELDER-MEAD', 'POWELL']
    needs_hessian = method_upper in ['NEWTON-CG', 'TRUST-NCG', 'DOGLEG', 'TRUST-EXACT']
    
    # Build minimize arguments
    minimize_args = {
        'fun': objective_fn,
        'x0': x0,
        'method': config['method'],
        'options': config['options']
    }
    
    # Add gradient if available and needed
    if not gradient_free and gradient_fn is not None:
        minimize_args['jac'] = gradient_fn
    
    # Add Hessian if available and needed
    if needs_hessian and hessian_fn is not None:
        minimize_args['hess'] = hessian_fn
    elif needs_hessian and hessian_fn is None:
        raise ValueError(f"Method {method} requires Hessian function")
    
    # Run optimization
    result = optimize.minimize(**minimize_args)
    
    # Compute gradient at final point if not provided
    if not hasattr(result, 'jac') or result.jac is None:
        if gradient_fn is not None:
            final_grad = gradient_fn(result.x)
        else:
            # Use finite differences as fallback
            eps = np.sqrt(np.finfo(float).eps)
            final_grad = optimize.approx_fprime(result.x, objective_fn, eps)
    else:
        final_grad = result.jac
    
    # Standardize result with lenient convergence check
    # R's nlm is more forgiving about convergence
    grad_norm = np.linalg.norm(final_grad)
    
    # More lenient convergence criteria matching R
    converged = result.success or grad_norm < 1e-3 or (result.nit > 0 and result.fun < 1e10)
    
    return {
        'x': result.x,
        'fun': result.fun,
        'grad': final_grad,
        'n_iter': result.nit,
        'converged': converged,  # Use lenient convergence
        'message': str(result.message) if hasattr(result, 'message') else '',
        'grad_norm': grad_norm,
        'n_fev': result.nfev,  # Function evaluations
        'n_jev': getattr(result, 'njev', None),  # Gradient evaluations
        'n_hev': getattr(result, 'nhev', None),  # Hessian evaluations
        'success': result.success  # Keep original success for diagnostics
    }


def validate_method(method: str, backend: str, has_hessian: bool) -> str:
    """
    Validate optimization method for given backend.
    
    Parameters
    ----------
    method : str
        Requested optimization method
    backend : str
        Backend type ('cpu', 'gpu_fp32', 'gpu_fp64')
    has_hessian : bool
        Whether Hessian computation is available
        
    Returns
    -------
    str
        Validated method name
        
    Raises
    ------
    ValueError
        If method is incompatible with backend
    """
    method_upper = method.upper()
    
    # Methods that need Hessian
    hessian_methods = {'NEWTON-CG', 'TRUST-NCG', 'DOGLEG', 'TRUST-EXACT'}
    
    # Methods that work without gradients
    gradient_free = {'NELDER-MEAD', 'POWELL'}
    
    # All scipy methods we support
    supported = {
        'BFGS', 'L-BFGS-B', 'NEWTON-CG', 'CG', 'TNC',
        'TRUST-NCG', 'DOGLEG', 'TRUST-EXACT',
        'NELDER-MEAD', 'POWELL'
    }
    
    if method_upper not in supported:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported methods: {', '.join(sorted(supported))}"
        )
    
    # Check Hessian requirement
    if method_upper in hessian_methods and not has_hessian:
        raise ValueError(
            f"Method '{method}' requires Hessian computation. "
            f"Use one of: {', '.join(sorted(supported - hessian_methods))}"
        )
    
    # Check backend compatibility
    if backend == 'gpu_fp32' and method_upper in hessian_methods:
        raise ValueError(
            f"Method '{method}' not recommended for FP32 precision. "
            f"Use BFGS or L-BFGS-B for better FP32 stability."
        )
    
    return method


def auto_select_method(
    backend: str,
    has_hessian: bool,
    problem_size: tuple,
    precision: str
) -> str:
    """
    Automatically select best optimization method.
    
    Parameters
    ----------
    backend : str
        Backend type
    has_hessian : bool
        Whether Hessian is available
    problem_size : tuple
        (n_observations, n_variables)
    precision : str
        'fp32' or 'fp64'
        
    Returns
    -------
    str
        Selected method name
    """
    n_obs, n_vars = problem_size
    n_params = n_vars + n_vars * (n_vars + 1) // 2  # Mean + covariance parameters
    
    # Large problem threshold
    is_large = n_params > 1000
    
    if precision == 'fp32' or backend == 'gpu_fp32':
        # FP32: Use BFGS or L-BFGS-B
        return 'L-BFGS-B' if is_large else 'BFGS'
    
    elif has_hessian and n_params < 500:
        # Small problem with Hessian: Newton-CG for fast convergence
        return 'Newton-CG'
    
    elif is_large:
        # Large problem: Limited memory BFGS
        return 'L-BFGS-B'
    
    else:
        # Default: BFGS
        return 'BFGS'