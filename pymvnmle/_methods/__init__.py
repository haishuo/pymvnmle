"""
Optimization methods module for PyMVNMLE.

This module provides optimization algorithms tailored for different
precision levels and hardware backends. The key principle is:
- BFGS for FP32 (consumer GPUs, Apple Metal)
- Newton-CG for FP64 (data center GPUs with full FP64 support)

The module automatically selects the optimal method based on backend
capabilities and problem characteristics.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

from typing import Dict, Tuple, Optional, Any, Callable
import numpy as np

# Import optimizers
from .bfgs import BFGSOptimizer, create_bfgs_optimizer
from .newton_cg import NewtonCGOptimizer, create_newton_cg_optimizer
from .method_selector import (
    MethodSelector,
    auto_select_method
)


def get_optimizer(
    method: str,
    backend_type: str,
    precision: str,
    problem_size: Tuple[int, int],
    max_iter: int,
    tol: float,
    verbose: bool
) -> Any:
    """
    Get optimizer instance for specified method and backend.
    
    Parameters
    ----------
    method : str
        Optimization method ('BFGS', 'Newton-CG', or 'auto')
    backend_type : str
        Backend type ('cpu', 'gpu_fp32', 'gpu_fp64')
    precision : str
        Precision level ('fp32' or 'fp64')
    problem_size : tuple
        (n_observations, n_variables)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
        
    Returns
    -------
    optimizer
        Configured optimizer instance
        
    Raises
    ------
    ValueError
        If method is incompatible with backend
    """
    # Validate method - use uppercase for comparison
    valid_methods = {'BFGS', 'NEWTON-CG', 'AUTO'}
    method_upper = method.upper() if method else 'AUTO'
    
    if method_upper not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Must be one of ['BFGS', 'Newton-CG', 'auto']"
        )
    
    # Check for Hessian support (Newton-CG requirement)
    has_hessian = (backend_type in {'gpu_fp32', 'gpu_fp64'})
    
    # Auto-select if requested
    if method_upper == 'AUTO':
        method_name, optimizer, config = auto_select_method(
            backend_type=backend_type,
            precision=precision,
            problem_size=problem_size,
            has_hessian=has_hessian,
            user_preference=None,
            max_iter=max_iter,
            verbose=verbose
        )
        return optimizer
    
    # Manual method selection
    if method_upper == 'BFGS':
        # BFGS works with any backend
        if precision == 'fp32':
            gtol = max(tol, 1e-5)  # FP32 limits
            ftol = max(tol, 1e-6)
        else:
            gtol = tol
            ftol = tol * 100  # Function tolerance looser than gradient
        
        return BFGSOptimizer(
            max_iter=max_iter,
            gtol=gtol,
            ftol=ftol,
            step_size_init=1.0,
            armijo_c1=1e-4,
            wolfe_c2=0.9,
            max_line_search=20,
            verbose=verbose
        )
    
    elif method_upper == 'NEWTON-CG':
        # Newton-CG requires FP64 and Hessian support
        if precision != 'fp64':
            raise ValueError(
                "Newton-CG requires FP64 precision for convergence. "
                f"Current precision: {precision}. Use BFGS instead."
            )
        
        if not has_hessian:
            raise ValueError(
                "Newton-CG requires analytical Hessian computation. "
                f"Backend '{backend_type}' doesn't support this. "
                "Use BFGS instead."
            )
        
        # Compute appropriate CG iterations
        n_params = problem_size[1] + problem_size[1] * (problem_size[1] + 1) // 2
        max_cg_iter = min(50, n_params // 2)
        
        return NewtonCGOptimizer(
            max_iter=max_iter,
            max_cg_iter=max_cg_iter,
            gtol=tol,
            ftol=tol * 0.01,  # Very tight function tolerance
            xtol=tol * 0.1,    # Parameter change tolerance
            cg_tol=1e-5,       # CG solver tolerance
            line_search_maxiter=10,
            trust_radius_init=1.0,
            verbose=verbose
        )


def compare_methods(
    objective_fn: Callable,
    gradient_fn: Callable,
    x0: np.ndarray,
    methods: Optional[list] = None,
    max_iter: int = 100,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different optimization methods on the same problem.
    
    Parameters
    ----------
    objective_fn : callable
        Objective function
    gradient_fn : callable
        Gradient function
    x0 : np.ndarray
        Initial point
    methods : list, optional
        Methods to compare (default: ['BFGS'])
    max_iter : int
        Maximum iterations
    verbose : bool
        Print comparison results
        
    Returns
    -------
    dict
        Results for each method
    """
    if methods is None:
        methods = ['BFGS']  # Default to BFGS only (Newton-CG needs Hessian)
    
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\nTesting {method}...")
            print("-" * 40)
        
        try:
            if method == 'BFGS':
                opt = create_bfgs_optimizer(
                    max_iter=max_iter,
                    precision='fp64',
                    verbose=False
                )
                result = opt.optimize(objective_fn, gradient_fn, x0)
                
            else:
                if verbose:
                    print(f"Skipping {method}: requires additional setup")
                continue
            
            results[method] = result
            
            if verbose:
                print(f"Converged: {result['converged']}")
                print(f"Iterations: {result['n_iter']}")
                print(f"Final objective: {result['fun']:.6e}")
                print(f"Gradient norm: {result['grad_norm']:.6e}")
                
        except Exception as e:
            if verbose:
                print(f"Method {method} failed: {e}")
            results[method] = {'error': str(e)}
    
    return results


def benchmark_convergence(
    objective_fn: Callable,
    gradient_fn: Callable,
    x0: np.ndarray,
    true_optimum: Optional[np.ndarray] = None,
    method: str = 'BFGS',
    max_iter: int = 100,
    record_every: int = 1
) -> Dict[str, Any]:
    """
    Benchmark convergence behavior of an optimization method.
    
    Parameters
    ----------
    objective_fn : callable
        Objective function
    gradient_fn : callable
        Gradient function
    x0 : np.ndarray
        Initial point
    true_optimum : np.ndarray, optional
        True optimal point for error calculation
    method : str
        Optimization method
    max_iter : int
        Maximum iterations
    record_every : int
        Record metrics every N iterations
        
    Returns
    -------
    dict
        Convergence history and metrics
    """
    history = {
        'iter': [],
        'obj_val': [],
        'grad_norm': [],
        'x': []
    }
    
    if true_optimum is not None:
        history['error'] = []
    
    # Callback to record history
    iteration_count = [0]
    
    def callback(x):
        iteration_count[0] += 1
        if iteration_count[0] % record_every == 0:
            obj_val = objective_fn(x)
            grad = gradient_fn(x)
            grad_norm = np.linalg.norm(grad)
            
            history['iter'].append(iteration_count[0])
            history['obj_val'].append(obj_val)
            history['grad_norm'].append(grad_norm)
            history['x'].append(x.copy())
            
            if true_optimum is not None:
                error = np.linalg.norm(x - true_optimum)
                history['error'].append(error)
    
    # Run optimization
    if method.upper() == 'BFGS':
        opt = create_bfgs_optimizer(
            max_iter=max_iter,
            precision='fp64',
            verbose=False
        )
        result = opt.optimize(objective_fn, gradient_fn, x0, callback)
    else:
        raise ValueError(f"Benchmarking not implemented for {method}")
    
    # Add final result
    history['final_result'] = result
    
    return history


# Public API
__all__ = [
    # Optimizers
    'BFGSOptimizer',
    'NewtonCGOptimizer',
    
    # Factory functions
    'create_bfgs_optimizer',
    'create_newton_cg_optimizer',
    'get_optimizer',
    
    # Method selection
    'MethodSelector',
    'auto_select_method',
    
    # Utilities
    'compare_methods',
    'benchmark_convergence'
]