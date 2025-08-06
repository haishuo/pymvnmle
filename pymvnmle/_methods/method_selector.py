"""
Method selection logic for PyMVNMLE optimization.

This module implements intelligent method selection based on backend
capabilities, precision levels, and problem characteristics.

The key principle: Choose the method that will converge most reliably
given the hardware and numerical precision constraints.

FIXED: Now properly handles tolerance parameter throughout the chain.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import warnings
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .bfgs import BFGSOptimizer
from .newton_cg import NewtonCGOptimizer


class MethodSelector:
    """
    Intelligent optimization method selector.
    
    Selects between BFGS and Newton-CG based on:
    - Backend type (CPU vs GPU)
    - Precision (FP32 vs FP64)
    - Problem size
    - Availability of analytical Hessian
    
    Attributes
    ----------
    backend_type : str
        Type of backend ('cpu', 'gpu_fp32', 'gpu_fp64')
    precision : str
        Precision level ('fp32' or 'fp64')
    problem_size : tuple
        (n_observations, n_variables)
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
            Backend type identifier
        precision : str
            Floating point precision
        problem_size : tuple
            (n_observations, n_variables)
        has_hessian : bool
            Hessian availability
        verbose : bool
            Verbosity flag
        """
        self.backend_type = backend_type
        self.precision = precision
        self.problem_size = problem_size
        self.n_obs, self.n_vars = problem_size
        self.has_hessian = has_hessian
        self.verbose = verbose
        
        # Compute problem characteristics
        self.n_params = self._compute_n_params(self.n_vars)
        self.is_large_problem = self._is_large_problem(self.n_obs, self.n_vars)
    
    def select_method(
        self,
        user_preference: Optional[str] = None,
        max_iter: int = 100,
        tol: float = 1e-6  # FIXED: Added tolerance parameter with R-compatible default
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Select optimization method and create configured optimizer.
        
        Parameters
        ----------
        user_preference : str or None
            User's requested method ('BFGS', 'Newton-CG', or None for auto)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance (gradient norm)
            
        Returns
        -------
        method_name : str
            Selected method name
        optimizer : object
            Configured optimizer instance
        config : dict
            Configuration parameters used
        """
        # Determine method
        method_name, reason = self._determine_method(user_preference)
        
        if self.verbose:
            print(f"Method Selection: {method_name}")
            print(f"  Reason: {reason}")
            print(f"  Backend: {self.backend_type}")
            print(f"  Precision: {self.precision}")
            print(f"  Problem size: {self.n_obs} × {self.n_vars}")
            print(f"  Parameters: {self.n_params}")
        
        # Create optimizer with proper tolerance
        if method_name == 'BFGS':
            optimizer, config = self._create_bfgs(max_iter, tol)
        elif method_name == 'Newton-CG':
            optimizer, config = self._create_newton_cg(max_iter, tol)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        return method_name, optimizer, config
    
    def _determine_method(
        self,
        user_preference: Optional[str]
    ) -> Tuple[str, str]:
        """
        Determine which optimization method to use.
        
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
        max_iter: int,
        tol: float  # FIXED: Now accepts tolerance parameter
    ) -> Tuple[BFGSOptimizer, Dict[str, Any]]:
        """
        Create BFGS optimizer with appropriate settings.
        
        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
        Returns
        -------
        optimizer : BFGSOptimizer
            Configured optimizer
        config : dict
            Configuration used
        """
        # FIXED: Use the provided tolerance with R-compatible bounds
        if self.precision == 'fp32':
            # FP32 cannot achieve very tight tolerances
            gtol = max(tol, 1e-5)
            ftol = max(tol * 100, 1e-6)
        else:  # fp64
            # For R compatibility, use the provided tolerance but not tighter than 1e-8
            # R's nlm typically achieves ~1e-4, so 1e-6 is a good default
            gtol = max(tol, 1e-8)  # Allow down to 1e-8 but no tighter
            ftol = tol * 100  # Function tolerance looser than gradient
        
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
        max_iter: int,
        tol: float  # FIXED: Now accepts tolerance parameter
    ) -> Tuple[NewtonCGOptimizer, Dict[str, Any]]:
        """
        Create Newton-CG optimizer with appropriate settings.
        
        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
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
        
        # FIXED: Use provided tolerance for Newton-CG
        # Newton-CG can achieve tighter tolerances than BFGS
        gtol = max(tol, 1e-10)  # Can go very tight with Newton-CG
        ftol = tol * 0.01  # Very tight function tolerance
        xtol = tol * 0.1   # Parameter change tolerance
        
        config = {
            'max_iter': max_iter,
            'max_cg_iter': max_cg_iter,
            'gtol': gtol,
            'ftol': ftol,
            'xtol': xtol,
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
            True if this is considered a large problem
        """
        # Large if many parameters or many observations
        n_params = self._compute_n_params(n_vars)
        return n_params > 1000 or n_obs > 10000


def auto_select_method(
    backend_type: str,
    precision: str,
    problem_size: Tuple[int, int],
    has_hessian: bool,
    user_preference: Optional[str],
    max_iter: int,
    tol: float,  # FIXED: Added tolerance parameter
    verbose: bool
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Automatically select and configure optimization method.
    
    This is a convenience function that creates a MethodSelector
    and immediately selects a method.
    
    Parameters
    ----------
    backend_type : str
        Backend type ('cpu', 'gpu_fp32', 'gpu_fp64')
    precision : str
        Precision level ('fp32' or 'fp64')
    problem_size : tuple
        (n_observations, n_variables)
    has_hessian : bool
        Whether analytical Hessian is available
    user_preference : str or None
        User's method preference
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print selection process
        
    Returns
    -------
    method_name : str
        Selected method
    optimizer : object
        Configured optimizer
    config : dict
        Configuration used
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
        max_iter=max_iter,
        tol=tol  # FIXED: Pass through tolerance
    )