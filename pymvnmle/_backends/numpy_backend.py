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
        Compute sign and log determinant using numpy.linalg.slogdet.
        
        More numerically stable than computing det() and taking log.
        Critical for likelihood computation in missing data MLE.
        """
        try:
            sign, logdet = np.linalg.slogdet(matrix)
            return float(sign), float(logdet)
        except Exception as e:
            raise NumericalError(f"Log determinant computation failed: {e}")
    
    def compute_gradient(self, objective_func: Callable, theta: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """
        Compute gradient using R's nlm finite difference approach.
        
        CRITICAL: Uses R's exact finite difference parameters to ensure
        identical behavior. This is why gradient norms are ~1e-4, not ~1e-15.
        
        This is the exact same logic as your working _objective.py gradient method,
        just adapted to work as a backend method.
        """
        if not callable(objective_func):
            raise TypeError("objective_func must be callable")
        
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError("theta must be 1-dimensional")
        
        try:
            n_params = len(theta)
            gradient = np.zeros(n_params)
            
            # R's nlm uses this specific epsilon - EXACT SAME AS WORKING VERSION
            eps = 1.49011612e-08  # R's .Machine$double.eps^(1/3)
            
            # Base function value
            f0 = objective_func(theta)
            
            for i in range(n_params):
                # R's step size calculation - EXACT SAME AS WORKING VERSION
                h = eps * max(abs(theta[i]), 1.0)
                
                # Ensure step is not too small - EXACT SAME AS WORKING VERSION
                if h < 1e-12:
                    h = 1e-12
                
                # Forward difference (R's default for nlm) - EXACT SAME AS WORKING VERSION
                theta_plus = theta.copy()
                theta_plus[i] = theta[i] + h
                
                try:
                    f_plus = objective_func(theta_plus)
                    gradient[i] = (f_plus - f0) / h
                except:
                    # If forward fails, try backward - EXACT SAME AS WORKING VERSION
                    theta_minus = theta.copy()
                    theta_minus[i] = theta[i] - h
                    try:
                        f_minus = objective_func(theta_minus)
                        gradient[i] = (f0 - f_minus) / h
                    except:
                        gradient[i] = 0.0
            
            return gradient
            
        except Exception as e:
            raise GradientComputationError(f"Finite difference computation failed: {e}")
    
    def __repr__(self) -> str:
        """Simple string representation."""
        return f"NumPyBackend(available=True, method=finite_differences)"