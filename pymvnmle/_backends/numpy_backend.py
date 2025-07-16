"""
NumPy backend for PyMVNMLE v2.0
CPU-based implementation using NumPy and SciPy - core functionality only

DESIGN PRINCIPLE: Minimal, focused implementation
- Core mathematical operations only
- No system diagnostics bloat
- No benchmarking utilities  
- No complex device information
- Just reliable linear algebra and finite differences

This serves as the reference implementation for regulatory compliance.
"""

import numpy as np
import scipy.linalg as linalg
from typing import Tuple, Callable
from .base import BackendInterface, NumericalError, GradientComputationError


class NumPyBackend(BackendInterface):
    """
    CPU backend using NumPy and SciPy - lean and focused.
    
    Responsibilities:
    1. Core linear algebra operations (cholesky, solve, slogdet)
    2. Finite difference gradient computation (R-compatible)
    3. Reliable availability (always works)
    4. Nothing else
    
    This backend maintains exact R compatibility through finite differences
    and serves as the gold standard that GPU backends must match numerically.
    """
    
    def __init__(self):
        """Initialize NumPy backend - simple and reliable."""
        # No complex initialization - NumPy is always available
        pass
    
    @property
    def is_available(self) -> bool:
        """NumPy backend is always available."""
        return True
    
    @property
    def name(self) -> str:
        """Backend name."""
        return "numpy"
    
    def gradient_method(self) -> str:
        """NumPy backend uses finite differences for R compatibility."""
        return 'finite_differences'
    
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition using scipy.linalg.cholesky.
        
        Uses scipy's implementation for better error handling than np.linalg.cholesky.
        Critical for inverse Cholesky parameterization in missing data MLE.
        """
        try:
            # scipy.linalg.cholesky with lower=False gives upper triangular
            return linalg.cholesky(matrix, lower=not upper)
        except linalg.LinAlgError as e:
            if "not positive definite" in str(e).lower():
                raise NumericalError(f"Matrix is not positive definite: {e}")
            else:
                raise NumericalError(f"Cholesky decomposition failed: {e}")
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        lower: bool = False) -> np.ndarray:
        """
        Solve triangular system using scipy.linalg.solve_triangular.
        
        Optimized for triangular systems - more stable than general linear solve.
        Used extensively in likelihood computation for pattern-specific calculations.
        """
        try:
            return linalg.solve_triangular(a, b, lower=lower)
        except linalg.LinAlgError as e:
            raise NumericalError(f"Triangular solve failed: {e}")
    
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant using scipy.linalg.slogdet.
        
        Numerically stable for computing log-determinants without overflow.
        Essential for log-likelihood computation in missing data MLE.
        """
        try:
            sign, logdet = linalg.slogdet(matrix)
            return float(sign), float(logdet)
        except Exception as e:
            raise NumericalError(f"Log-determinant computation failed: {e}")
    
    def compute_gradient(self, objective_func: Callable, theta: np.ndarray, 
                        step_size: float = 1e-8) -> np.ndarray:
        """
        Compute gradient using finite differences - R-compatible method.
        
        This implements the exact finite difference scheme used by R's nlm() function,
        producing gradient norms ~1e-4 at convergence (not machine precision).
        
        Parameters
        ----------
        objective_func : callable
            Function that takes parameter vector and returns scalar
        theta : np.ndarray
            Parameter vector at which to evaluate gradient  
        step_size : float, default=1e-8
            Step size for finite differences (matches R's nlm default)
            
        Returns
        -------
        np.ndarray
            Gradient vector computed via finite differences
            
        Notes
        -----
        This maintains exact R compatibility for regulatory compliance.
        The "approximate" gradient norms are intentional - this is how R works.
        """
        if not callable(objective_func):
            raise TypeError("objective_func must be callable")
        
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError("theta must be 1-dimensional")
        
        try:
            # Evaluate at current point
            f0 = objective_func(theta)
            if not np.isfinite(f0):
                raise GradientComputationError(f"Objective function returned {f0}")
            
            # Compute finite difference gradient
            gradient = np.zeros_like(theta)
            
            for i in range(len(theta)):
                # Forward step
                theta_plus = theta.copy()
                theta_plus[i] += step_size
                f_plus = objective_func(theta_plus)
                
                if np.isfinite(f_plus):
                    # Forward difference (preferred)
                    gradient[i] = (f_plus - f0) / step_size
                else:
                    # Try backward step if forward fails
                    theta_minus = theta.copy()
                    theta_minus[i] -= step_size
                    f_minus = objective_func(theta_minus)
                    
                    if not np.isfinite(f_minus):
                        raise GradientComputationError(
                            f"Both forward and backward steps failed at parameter {i}"
                        )
                    
                    # Backward difference
                    gradient[i] = (f0 - f_minus) / step_size
            
            return gradient
            
        except Exception as e:
            if isinstance(e, GradientComputationError):
                raise
            else:
                raise GradientComputationError(f"Finite difference computation failed: {e}")
    
    def __repr__(self) -> str:
        """Simple string representation."""
        return f"NumPyBackend(available=True, method=finite_differences)"