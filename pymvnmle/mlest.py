"""
Main maximum likelihood estimation function for PyMVNMLE
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass

try:
    from scipy.optimize import minimize
except ImportError:
    raise ImportError("SciPy is required for optimization. Install with: pip install scipy")

from ._utils import (
    validate_input_data, 
    mysort_data, 
    get_starting_values,
    format_result
)
from ._likelihood import create_likelihood_function
from ._backends import get_backend_with_fallback, select_optimal_backend


@dataclass
class MLResult:
    """
    Result object for maximum likelihood estimation.
    
    Attributes
    ----------
    muhat : np.ndarray
        Maximum likelihood estimate of mean vector
    sigmahat : np.ndarray  
        Maximum likelihood estimate of covariance matrix
    loglik : float
        Log-likelihood value at the maximum
    converged : bool
        Whether optimization converged successfully
    convergence_message : str
        Human-readable convergence information
    n_iter : int
        Number of optimization iterations used
    method : str
        Optimization method that was used
    backend : str
        Computational backend that was used
    gpu_accelerated : bool
        Whether GPU acceleration was used
    computation_time : float
        Wall-clock time for estimation (seconds)
    gradient : np.ndarray, optional
        Final gradient vector (for diagnostics)
    hessian : np.ndarray, optional
        Final Hessian matrix (if computed by optimizer)
    """
    muhat: np.ndarray
    sigmahat: np.ndarray
    loglik: float
    converged: bool
    convergence_message: str
    n_iter: int
    method: str
    backend: str
    gpu_accelerated: bool
    computation_time: float
    gradient: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        """String representation focusing on key results."""
        conv_status = "✓" if self.converged else "✗"
        gpu_status = "🔥" if self.gpu_accelerated else "💻"
        
        return (f"MLResult({conv_status} converged in {self.n_iter} iter, "
                f"loglik={self.loglik:.3f}, {gpu_status} {self.backend}, "
                f"{self.computation_time:.3f}s)")


def mlest(data: Union[np.ndarray, pd.DataFrame], 
          backend: str = 'auto',
          method: str = 'bfgs',  # Change default to BFGS - more reliable
          max_iter: int = 1000, 
          tol: float = 1e-6, 
          verbose: bool = False,
          **optimizer_kwargs) -> MLResult:
    """
    Maximum likelihood estimation for multivariate normal data with missing values.
    
    Finds the ML estimates of the mean vector and covariance matrix for multivariate 
    normal data with arbitrary missing data patterns. Uses GPU acceleration when 
    beneficial and available.
    
    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Multivariate data matrix. Missing values should be np.nan.
        Accepts NumPy arrays or pandas DataFrames.
        
    backend : str, default='auto'
        Computational backend for linear algebra. Options:
        - 'auto': Intelligent selection based on problem size and hardware
        - 'numpy': CPU-only NumPy/SciPy (always available)
        - 'cupy': NVIDIA GPU acceleration (requires cupy)
        - 'metal': Apple Silicon GPU acceleration (requires torch with MPS)
        - 'jax': JAX/XLA compilation for GPU/TPU (requires jax)
        
            method : str, default='bfgs'
        Optimization algorithm. Options:
        - 'bfgs': Broyden-Fletcher-Goldfarb-Shanno (reliable, recommended)
        - 'newton-cg': Newton-Conjugate Gradient (may need analytical gradients)
        - 'l-bfgs-b': Limited memory BFGS with bounds
        - 'nelder-mead': Nelder-Mead simplex (gradient-free)
        - 'powell': Powell's method (gradient-free)
        
        Note: BFGS is more robust than Newton-CG with numerical gradients
        and produces identical results to R's implementation.
        
    max_iter : int, default=1000
        Maximum number of optimization iterations.
        
    tol : float, default=1e-6
        Convergence tolerance for optimization.
        
    verbose : bool, default=False
        Whether to print optimization progress and backend selection.
        
    **optimizer_kwargs
        Additional arguments passed to scipy.optimize.minimize
        
    Returns
    -------
    MLResult
        Result object with ML estimates and computation details
        
    Examples
    --------
    >>> import numpy as np
    >>> from pymvnmle import mlest
    >>> 
    >>> # Basic usage
    >>> data = np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 4.0]])
    >>> result = mlest(data)
    >>> print(f"Mean: {result.muhat}")
    >>> print(f"Covariance: {result.sigmahat}")
    
    >>> # With GPU acceleration
    >>> result = mlest(data, backend='auto', verbose=True)
    >>> print(f"Used {result.backend} backend")
    >>> print(f"GPU accelerated: {result.gpu_accelerated}")
    
    >>> # Custom optimization
    >>> result = mlest(data, method='l-bfgs-b', max_iter=500, tol=1e-8)
    
    Notes
    -----
    This function implements maximum likelihood estimation for the multivariate
    normal distribution with missing data under the Missing At Random (MAR) 
    assumption. The algorithm uses an inverse Cholesky parameterization to 
    ensure positive definite covariance estimates and groups observations by 
    missingness patterns for computational efficiency.
    
    The implementation is mathematically equivalent to R's mvnmle package but
    offers significant performance improvements through GPU acceleration and
    modern optimization algorithms.
    
    References
    ----------
    Little, R.J.A. and Rubin, D.B. (2019). Statistical Analysis with Missing 
    Data, 3rd ed. Hoboken, NJ: Wiley.
    
    Pinheiro, J.C. and Bates, D.M. (2000). Mixed-Effects Models in S and S-PLUS. 
    New York: Springer-Verlag.
    """
    start_time = time.time()
    
    # Input validation and preprocessing
    if verbose:
        print("Validating input data...")
    
    data_array = validate_input_data(data)
    n_obs, n_vars = data_array.shape
    
    if verbose:
        print(f"Data shape: {n_obs} observations × {n_vars} variables")
        missing_rate = np.sum(np.isnan(data_array)) / (n_obs * n_vars)
        print(f"Missing data rate: {missing_rate:.1%}")
    
    # Backend selection
    if verbose:
        print("Selecting computational backend...")
    
    try:
        backend_obj = get_backend_with_fallback(
            backend, 
            data_shape=(n_obs, n_vars),
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize backend '{backend}': {e}")
    
    # Data preprocessing: sort by missingness patterns
    if verbose:
        print("Preprocessing data (sorting by missingness patterns)...")
    
    sorted_data, freq, presence_absence = mysort_data(data_array)
    n_patterns = len(freq)
    
    if verbose:
        print(f"Found {n_patterns} unique missingness patterns")
        print(f"Pattern frequencies: {freq}")
    
    # Compute starting values
    if verbose:
        print("Computing starting values...")
    
    try:
        start_vals = get_starting_values(data_array)
    except Exception as e:
        raise RuntimeError(f"Failed to compute starting values: {e}")
    
    if verbose:
        print(f"Starting values computed ({len(start_vals)} parameters)")
    
    # Create likelihood function
    if verbose:
        print(f"Creating likelihood function (using {backend_obj.name} backend)...")
    
    # Use gradient-enabled version for Newton methods, regular version for others
    gradient_methods = ['newton-cg', 'trust-exact', 'trust-krylov', 'trust-constr', 'bfgs']
    if method in gradient_methods:
        from ._likelihood import create_likelihood_function_with_gradient
        likelihood_func = create_likelihood_function_with_gradient(
            sorted_data, freq, presence_absence, backend_obj
        )
        use_gradients = True
    else:
        likelihood_func = create_likelihood_function(
            sorted_data, freq, presence_absence, backend_obj
        )
        use_gradients = False
    
    # Set up optimization
    if verbose:
        print(f"Starting optimization (method: {method})...")
    
    # Prepare optimizer arguments
    opt_args = {
        'method': method,
        'options': {
            'maxiter': max_iter,
            'disp': verbose
        }
    }
    
    # Add method-specific options
    if method == 'newton-cg':
        opt_args['jac'] = use_gradients  # Use analytical gradients if available
        opt_args['options']['xtol'] = tol
    elif method == 'bfgs':
        opt_args['jac'] = use_gradients  # BFGS can use gradients too
        opt_args['options']['gtol'] = tol
    elif method == 'l-bfgs-b':
        opt_args['jac'] = use_gradients if use_gradients else None
        opt_args['options']['ftol'] = tol
        opt_args['options']['gtol'] = tol
    else:
        opt_args['options']['xtol'] = tol
    
    # Merge user-provided options
    opt_args.update(optimizer_kwargs)
    
    # Run optimization
    try:
        opt_result = minimize(likelihood_func, start_vals, **opt_args)
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}")
    
    # Compute final timing
    computation_time = time.time() - start_time
    
    if verbose:
        print(f"Optimization completed in {computation_time:.3f}s")
        print(f"Converged: {opt_result.success}")
        print(f"Iterations: {opt_result.get('nit', 'unknown')}")
    
    # Format results
    try:
        # Handle different scipy result formats
        if hasattr(opt_result, 'fun'):
            final_value = opt_result.fun
        elif hasattr(opt_result, 'fval'):
            final_value = opt_result.fval
        else:
            final_value = float('inf')  # Fallback
            
        result_dict = format_result(
            {
                'fun': final_value,
                'x': opt_result.x if hasattr(opt_result, 'x') else start_vals,
                'success': getattr(opt_result, 'success', False),
                'nit': getattr(opt_result, 'nit', 0),
                'message': getattr(opt_result, 'message', 'Unknown'),
                'jac': getattr(opt_result, 'jac', None),
                'hess': getattr(opt_result, 'hess', None)
            },
            opt_result.x if hasattr(opt_result, 'x') else start_vals,
            n_vars,
            backend_obj.name,
            backend_obj.name != 'numpy',
            computation_time
        )
    except Exception as e:
        raise RuntimeError(f"Failed to format results: {e}")
    
    # Create result object
    result = MLResult(
        muhat=result_dict['muhat'],
        sigmahat=result_dict['sigmahat'],
        loglik=result_dict['loglik'],
        converged=result_dict['converged'],
        convergence_message=result_dict['convergence_message'],
        n_iter=result_dict['n_iter'],
        method=result_dict['method'],
        backend=result_dict['backend'],
        gpu_accelerated=result_dict['gpu_accelerated'],
        computation_time=result_dict['computation_time'],
        gradient=result_dict.get('gradient'),
        hessian=result_dict.get('hessian')
    )
    
    if verbose:
        print(f"Final result: {result}")
    
    return result


# For backwards compatibility and convenience
def ml_estimate(data, **kwargs):
    """Alias for mlest() function."""
    return mlest(data, **kwargs)