"""
Automatic method selection based on precision and hardware.

This module provides intelligent selection of optimization methods
based on the backend precision, hardware capabilities, and problem
characteristics. Ensures optimal algorithm choice for each scenario.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import warnings

from .bfgs import BFGSOptimizer, create_bfgs_optimizer
from .newton_cg import NewtonCGOptimizer, create_newton_cg_optimizer


class MethodSelector:
    """
    Intelligent optimization method selection based on backend capabilities.
    
    This class analyzes the backend precision, problem size, and hardware
    characteristics to select the optimal optimization method. Follows
    the principle: BFGS for FP32, Newton-CG for FP64.
    
    Attributes
    ----------
    backend_type : str
        Type of backend (cpu, gpu_fp32, gpu_fp64)
    precision : str
        Floating point precision (fp32 or fp64)
    problem_size : tuple
        Dimensions of the problem (n_obs, n_vars)
    has_hessian : bool
        Whether analytical Hessian is available
    verbose : bool
        Print selection reasoning
    """
    
    def __init__(
        self,
        backend_type: str,
        precision: str,
        problem_size: Tuple[int, int],
        has_hessian: bool,
        verbose: bool
    ):
        """
        Initialize method selector with problem characteristics.
        
        Parameters
        ----------
        backend_type : str
            One of 'cpu', 'gpu_fp32', 'gpu_fp64'
        precision : str
            One of 'fp32', 'fp64'
        problem_size : tuple
            (n_observations, n_variables)
        has_hessian : bool
            Whether the objective provides analytical Hessian
        verbose : bool
            Print selection reasoning
        """
        # Validate inputs
        valid_backends = {'cpu', 'gpu_fp32', 'gpu_fp64'}
        if backend_type not in valid_backends:
            raise ValueError(
                f"Invalid backend_type '{backend_type}'. "
                f"Must be one of {valid_backends}"
            )
        
        valid_precisions = {'fp32', 'fp64'}
        if precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{precision}'. "
                f"Must be one of {valid_precisions}"
            )
        
        if len(problem_size) != 2:
            raise ValueError(
                f"problem_size must be (n_obs, n_vars), got {problem_size}"
            )
        
        n_obs, n_vars = problem_size
        if n_obs <= 0 or n_vars <= 0:
            raise ValueError(
                f"Problem dimensions must be positive, got n_obs={n_obs}, n_vars={n_vars}"
            )
        
        self.backend_type = backend_type
        self.precision = precision
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.has_hessian = has_hessian
        self.verbose = verbose
        
        # Compute problem characteristics
        self.n_params = self._compute_n_params(n_vars)
        self.is_large_problem = self._is_large_problem(n_obs, n_vars)
        
    def select_method(
        self,
        user_preference: Optional[str] = None,
        max_iter: int = 1000
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Select optimal optimization method.
        
        Parameters
        ----------
        user_preference : str, optional
            User's preferred method (may be overridden if incompatible)
        max_iter : int
            Maximum iterations for optimization
            
        Returns
        -------
        method_name : str
            Selected method name ('BFGS' or 'Newton-CG')
        optimizer : BFGSOptimizer or NewtonCGOptimizer
            Configured optimizer instance
        config : dict
            Configuration parameters used
        """
        # Determine method based on rules
        method_name, reason = self._determine_method(user_preference)
        
        # Create optimizer with appropriate settings
        if method_name == 'BFGS':
            optimizer, config = self._create_bfgs(max_iter)
        elif method_name == 'Newton-CG':
            optimizer, config = self._create_newton_cg(max_iter)
        else:
            raise RuntimeError(f"Unknown method selected: {method_name}")
        
        if self.verbose:
            self._print_selection_summary(method_name, reason, config)
        
        return method_name, optimizer, config
    
    def _determine_method(
        self,
        user_preference: Optional[str]
    ) -> Tuple[str, str]:
        """
        Determine which method to use based on backend and user preference.
        
        Parameters
        ----------
        user_preference : str or None
            User's requested method
            
        Returns
        -------
        method : str
            Selected method name
        reason : str
            Explanation for selection
        """
        # Rule 1: CPU backend always uses BFGS (R compatibility)
        if self.backend_type == 'cpu':
            if user_preference == 'Newton-CG':
                warnings.warn(
                    "Newton-CG requires GPU with FP64 support. "
                    "Falling back to BFGS for CPU backend."
                )
                return 'BFGS', 'CPU backend (R compatibility)'
            return 'BFGS', 'CPU backend (R compatibility)'
        
        # Rule 2: GPU FP32 always uses BFGS
        if self.backend_type == 'gpu_fp32':
            if user_preference == 'Newton-CG':
                warnings.warn(
                    "Newton-CG requires FP64 precision for convergence. "
                    "Using BFGS for FP32 backend."
                )
                return 'BFGS', 'FP32 precision (insufficient for Newton-CG)'
            return 'BFGS', 'FP32 optimal method'
        
        # Rule 3: GPU FP64 - decide based on problem characteristics
        if self.backend_type == 'gpu_fp64':
            # Check if we have Hessian support
            if not self.has_hessian:
                if user_preference == 'Newton-CG':
                    warnings.warn(
                        "Newton-CG requires analytical Hessian. "
                        "Falling back to BFGS."
                    )
                return 'BFGS', 'No analytical Hessian available'
            
            # Check problem size suitability for Newton-CG
            if self.n_params > 10000:
                # Very large problems - Newton-CG might be too expensive
                if user_preference == 'Newton-CG':
                    warnings.warn(
                        f"Newton-CG may be slow for {self.n_params} parameters. "
                        f"Consider using BFGS."
                    )
                    # Respect user preference but warn
                    return 'Newton-CG', 'User preference (large problem warning issued)'
                return 'BFGS', f'Large problem ({self.n_params} parameters)'
            
            # Small to medium problems - Newton-CG is ideal
            if user_preference == 'BFGS':
                # Respect user preference
                return 'BFGS', 'User preference'
            
            return 'Newton-CG', 'FP64 with Hessian support (optimal)'
        
        # Shouldn't reach here
        raise RuntimeError(f"Unexpected backend type: {self.backend_type}")
    
    def _create_bfgs(
        self,
        max_iter: int
    ) -> Tuple[BFGSOptimizer, Dict[str, Any]]:
        """
        Create BFGS optimizer with appropriate settings.
        
        Parameters
        ----------
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        optimizer : BFGSOptimizer
            Configured optimizer
        config : dict
            Configuration used
        """
        # Adjust tolerances based on precision and problem size
        if self.precision == 'fp32':
            gtol = 1e-5 if not self.is_large_problem else 5e-5
            ftol = 1e-6 if not self.is_large_problem else 5e-6
        else:  # fp64
            gtol = 1e-7 if not self.is_large_problem else 1e-6
            ftol = 1e-9 if not self.is_large_problem else 1e-8
        
        config = {
            'max_iter': max_iter,
            'gtol': gtol,
            'ftol': ftol,
            'step_size_init': 1.0,
            'armijo_c1': 1e-4,
            'wolfe_c2': 0.9,
            'max_line_search': 20,
            'verbose': self.verbose
        }
        
        optimizer = BFGSOptimizer(**config)
        
        return optimizer, config
    
    def _create_newton_cg(
        self,
        max_iter: int
    ) -> Tuple[NewtonCGOptimizer, Dict[str, Any]]:
        """
        Create Newton-CG optimizer with appropriate settings.
        
        Parameters
        ----------
        max_iter : int
            Maximum iterations
            
        Returns
        -------
        optimizer : NewtonCGOptimizer
            Configured optimizer
        config : dict
            Configuration used
        """
        # Newton-CG requires FP64
        if self.precision != 'fp64':
            raise ValueError(
                "Newton-CG requires FP64 precision. "
                "This should have been caught earlier."
            )
        
        # Adjust CG iterations based on problem size
        max_cg_iter = min(50, self.n_params // 2)
        
        # Tighter tolerances for FP64 Newton-CG
        config = {
            'max_iter': max_iter,
            'max_cg_iter': max_cg_iter,
            'gtol': 1e-8,
            'ftol': 1e-10,
            'xtol': 1e-9,
            'cg_tol': 1e-5,
            'line_search_maxiter': 10,
            'trust_radius_init': 1.0,
            'verbose': self.verbose
        }
        
        optimizer = NewtonCGOptimizer(**config)
        
        return optimizer, config
    
    def _compute_n_params(self, n_vars: int) -> int:
        """
        Compute number of parameters in the optimization.
        
        For multivariate normal: n_vars means + n_vars*(n_vars+1)/2 covariance params
        
        Parameters
        ----------
        n_vars : int
            Number of variables
            
        Returns
        -------
        int
            Total number of parameters
        """
        # Mean parameters + covariance parameters (triangular)
        return n_vars + n_vars * (n_vars + 1) // 2
    
    def _is_large_problem(self, n_obs: int, n_vars: int) -> bool:
        """
        Determine if this is a large-scale problem.
        
        Parameters
        ----------
        n_obs : int
            Number of observations
        n_vars : int
            Number of variables
            
        Returns
        -------
        bool
            True if problem is considered large
        """
        # Problem is large if:
        # - Many observations (>10000)
        # - Many variables (>50)
        # - Many parameters (>1000)
        n_params = self._compute_n_params(n_vars)
        
        return (n_obs > 10000) or (n_vars > 50) or (n_params > 1000)
    
    def _print_selection_summary(
        self,
        method: str,
        reason: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Print summary of method selection.
        
        Parameters
        ----------
        method : str
            Selected method
        reason : str
            Reason for selection
        config : dict
            Configuration parameters
        """
        print("=" * 60)
        print("Optimization Method Selection")
        print("=" * 60)
        print(f"Backend: {self.backend_type}")
        print(f"Precision: {self.precision}")
        print(f"Problem size: n_obs={self.n_obs}, n_vars={self.n_vars}")
        print(f"Parameters: {self.n_params}")
        print(f"Has Hessian: {self.has_hessian}")
        print("-" * 60)
        print(f"Selected method: {method}")
        print(f"Reason: {reason}")
        print("-" * 60)
        print("Configuration:")
        for key, value in config.items():
            if key != 'verbose':  # Don't print verbose itself
                if isinstance(value, float):
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value}")
        print("=" * 60)


def auto_select_method(
    backend_type: str,
    precision: str,
    problem_size: Tuple[int, int],
    has_hessian: bool = False,
    user_preference: Optional[str] = None,
    max_iter: int = 1000,
    verbose: bool = False
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Automatically select optimization method based on problem characteristics.
    
    This is the main entry point for method selection. It analyzes the
    backend, precision, and problem size to choose the optimal method.
    
    Parameters
    ----------
    backend_type : str
        One of 'cpu', 'gpu_fp32', 'gpu_fp64'
    precision : str
        One of 'fp32', 'fp64'
    problem_size : tuple
        (n_observations, n_variables)
    has_hessian : bool
        Whether analytical Hessian is available
    user_preference : str, optional
        User's preferred method (may be overridden)
    max_iter : int
        Maximum optimization iterations
    verbose : bool
        Print selection details
        
    Returns
    -------
    method_name : str
        Selected method ('BFGS' or 'Newton-CG')
    optimizer : object
        Configured optimizer instance
    config : dict
        Configuration parameters
        
    Examples
    --------
    >>> # CPU backend - will select BFGS
    >>> method, opt, cfg = auto_select_method('cpu', 'fp64', (100, 5))
    
    >>> # GPU FP32 - will select BFGS
    >>> method, opt, cfg = auto_select_method('gpu_fp32', 'fp32', (1000, 10))
    
    >>> # GPU FP64 with Hessian - will select Newton-CG
    >>> method, opt, cfg = auto_select_method('gpu_fp64', 'fp64', (500, 20), 
    ...                                       has_hessian=True)
    """
    selector = MethodSelector(
        backend_type=backend_type,
        precision=precision,
        problem_size=problem_size,
        has_hessian=has_hessian,
        verbose=verbose
    )
    
    return selector.select_method(
        user_preference=user_preference,
        max_iter=max_iter
    )