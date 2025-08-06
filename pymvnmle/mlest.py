"""
Maximum Likelihood Estimation for multivariate normal data with missing values.

This module provides the main API for PyMVNMLE, implementing ML estimation
using various backends and optimization methods.
"""

import numpy as np
import warnings
from typing import Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass
import time

from ._objectives import get_objective, MLEObjectiveBase
from ._objectives.parameterizations import get_parameterization


@dataclass
class MLResult:
    """Result from ML estimation."""
    muhat: np.ndarray
    sigmahat: np.ndarray
    converged: bool
    n_iter: int
    loglik: float
    computation_time: float
    backend: str
    method: str
    n_patterns: int
    n_obs: int
    n_vars: int
    gradient_norm: Optional[float] = None
    

def _detect_gpu_precision() -> str:
    """
    Detect GPU precision capability.
    
    Returns
    -------
    str
        'fp64' if GPU supports fast FP64, 'fp32' otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            
            # Data center GPUs with full FP64
            if any(gpu in name for gpu in ['A100', 'A6000', 'H100', 'V100']):
                return 'fp64'
            
            # Consumer GPUs have gimped FP64
            if any(gpu in name for gpu in ['RTX', 'GTX', 'GeForce']):
                return 'fp32'
                
            # Default to FP32 for unknown GPUs
            return 'fp32'
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Metal only supports FP32
            return 'fp32'
    except ImportError:
        pass
    
    return 'fp32'  # Default to FP32


def _select_optimizer_config(backend: str, precision: str, method: Optional[str] = None) -> Dict[str, Any]:
    """
    Select optimizer configuration based on backend and precision.
    
    Parameters
    ----------
    backend : str
        'cpu' or 'gpu'
    precision : str
        'fp32' or 'fp64'
    method : str or None
        Explicit method override
        
    Returns
    -------
    dict
        Optimizer configuration with method, options, and use_bounds flag
    """
    if method is not None:
        # User override
        return {
            'method': method,
            'options': {'maxiter': 100},
            'use_bounds': method == 'L-BFGS-B'
        }
    
    if backend == 'gpu' and precision == 'fp32':
        # FP32 GPU needs bounded optimization for stability
        return {
            'method': 'L-BFGS-B',
            'options': {
                'maxiter': 100,
                'gtol': 1e-5,
                'ftol': 1e-9
            },
            'use_bounds': True
        }
    elif backend == 'gpu' and precision == 'fp64':
        # FP64 GPU can use Newton-CG
        return {
            'method': 'Newton-CG',
            'options': {
                'maxiter': 100,
                'xtol': 1e-8
            },
            'use_bounds': False
        }
    else:
        # CPU uses BFGS (R-compatible)
        return {
            'method': 'BFGS',
            'options': {
                'maxiter': 100,
                'gtol': 1e-5
            },
            'use_bounds': False
        }


def mlest(data: Union[np.ndarray, 'pd.DataFrame'],
         backend: str = 'cpu',
         method: Optional[str] = None,
         tol: float = 1e-5,
         max_iter: int = 100,
         verbose: bool = False) -> MLResult:
    """
    Maximum likelihood estimation for multivariate normal with missing data.
    
    Parameters
    ----------
    data : array-like, shape (n_obs, n_vars)
        Input data with missing values as np.nan
    backend : str, default='cpu'
        Backend to use: 'cpu' or 'gpu'
    method : str or None
        Optimization method. If None, automatically selected based on backend
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    verbose : bool
        Print optimization progress
        
    Returns
    -------
    MLResult
        Estimation results
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 3)
    >>> data[np.random.random(data.shape) < 0.1] = np.nan
    >>> result = mlest(data)
    >>> print(result.muhat)  # Estimated mean
    >>> print(result.sigmahat)  # Estimated covariance
    """
    # Convert to numpy array
    if hasattr(data, 'values'):  # pandas DataFrame
        data = data.values
    else:
        data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    n_obs, n_vars = data.shape
    
    # Handle backend selection
    backend = backend.lower()
    precision = 'fp64'  # Default
    
    if backend == 'gpu':
        precision = _detect_gpu_precision()
        if verbose:
            print(f"GPU detected with {precision.upper()} precision")
    
    # Create objective function
    try:
        if backend == 'gpu' and precision == 'fp32':
            # Use bounded parameterization for FP32 stability
            objective = get_objective(data, backend='gpu', precision='fp32')
            
            # Replace with bounded parameterization
            from ._objectives.parameterizations import BoundedCholeskyParameterization
            objective.parameterization = BoundedCholeskyParameterization(n_vars)
            objective.n_params = objective.parameterization.n_params
            
        elif backend == 'gpu' and precision == 'fp64':
            objective = get_objective(data, backend='gpu', precision='fp64')
        else:
            objective = get_objective(data, backend='cpu')
            
    except ImportError as e:
        if 'torch' in str(e).lower():
            warnings.warn(f"GPU backend requires PyTorch. Falling back to CPU.")
            backend = 'cpu'
            objective = get_objective(data, backend='cpu')
        else:
            raise
    
    # Get optimizer configuration
    opt_config = _select_optimizer_config(backend, precision, method)
    
    # Initial parameters
    theta0 = objective.get_initial_parameters()
    
    if verbose:
        print(f"Backend: {backend}")
        print(f"Method: {opt_config['method']}")
        print(f"Parameters: {objective.n_params}")
        print(f"Patterns: {objective.n_patterns}")
    
    # Run optimization
    start_time = time.time()
    
    from scipy.optimize import minimize
    
    # Prepare optimization arguments
    opt_args = {
        'fun': objective.compute_objective,
        'x0': theta0,
        'method': opt_config['method'],
        'jac': objective.compute_gradient,
        'options': opt_config['options']
    }
    
    # Add method-specific arguments
    if opt_config['method'] == 'Newton-CG':
        opt_args['hess'] = objective.compute_hessian
    elif opt_config['method'] == 'L-BFGS-B' and opt_config['use_bounds']:
        if hasattr(objective, 'get_optimization_bounds'):
            opt_args['bounds'] = objective.get_optimization_bounds()
        else:
            # Fallback bounds for FP32 stability
            bounds = []
            for i in range(n_vars):
                bounds.append((None, None))  # Mean: unbounded
            for i in range(n_vars):
                bounds.append((-5.0, 5.0))  # Log diagonal: bounded
            n_off_diag = (n_vars * (n_vars - 1)) // 2
            for i in range(n_off_diag):
                bounds.append((-2.0, 2.0))  # Off-diagonal: bounded
            opt_args['bounds'] = bounds
    
    # Run optimization
    result = minimize(**opt_args)
    
    computation_time = time.time() - start_time
    
    # Extract results
    mu_final, sigma_final, loglik = objective.extract_parameters(result.x)
    
    # Final gradient norm
    grad_norm = None
    if hasattr(result, 'jac'):
        grad_norm = np.linalg.norm(result.jac)
    
    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Converged: {result.success}")
        print(f"  Iterations: {result.nit}")
        print(f"  Time: {computation_time:.3f}s")
        if grad_norm is not None:
            print(f"  Final gradient norm: {grad_norm:.2e}")
    
    return MLResult(
        muhat=mu_final,
        sigmahat=sigma_final,
        converged=result.success,
        n_iter=result.nit,
        loglik=loglik,
        computation_time=computation_time,
        backend=backend,
        method=opt_config['method'],
        n_patterns=objective.n_patterns,
        n_obs=n_obs,
        n_vars=n_vars,
        gradient_norm=grad_norm
    )