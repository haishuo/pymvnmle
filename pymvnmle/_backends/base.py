"""
Abstract backend interfaces for PyMVNMLE v2.0
Pure abstract base classes defining the computational contracts

DESIGN PRINCIPLE: Unix Philosophy - "Do one thing and do it well"
This module contains ONLY abstract interface definitions. No implementation,
no utilities, no diagnostics, no validation - just pure contracts.

All concrete functionality belongs in separate, focused modules:
- Validation → _backends/validation.py
- Benchmarking → _backends/benchmarking.py  
- Diagnostics → _utils/system_info.py
- Error handling → Custom exception classes only
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Callable


class BackendInterface(ABC):
    """
    Pure abstract interface for all computational backends.
    
    Defines the minimal contract that all backends (CPU and GPU) must implement
    for multivariate normal maximum likelihood estimation with missing data.
    
    Design Principles:
    1. Interface only - no implementation details
    2. Minimal surface area - only essential operations
    3. NumPy arrays at boundaries - consistent input/output types
    4. Mathematical focus - operations needed for MLE computation
    
    The interface covers exactly four core mathematical operations:
    - Cholesky decomposition (for inverse Cholesky parameterization)
    - Triangular system solving (for likelihood computation)  
    - Log-determinant computation (for log-likelihood evaluation)
    - Gradient computation (finite differences or autodiff)
    """
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @property
    @abstractmethod  
    def name(self) -> str:
        """Get the backend name."""
        pass
    
    @abstractmethod
    def gradient_method(self) -> str:
        """
        Return the gradient computation method.
        
        Returns
        -------
        str
            Either 'finite_differences' or 'autodiff'
        """
        pass
    
    @abstractmethod
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition of positive definite matrix.
        
        Parameters
        ----------
        matrix : np.ndarray, shape (n, n)
            Positive definite matrix to decompose
        upper : bool, default=True
            If True, return upper triangular factor U where A = U.T @ U
            If False, return lower triangular factor L where A = L @ L.T
            
        Returns
        -------
        np.ndarray, shape (n, n)
            Cholesky factor as NumPy array
            
        Notes
        -----
        Critical for inverse Cholesky parameterization: Σ = (Δ⁻¹)ᵀ Δ⁻¹
        Must handle near-singular matrices gracefully.
        """
        pass
    
    @abstractmethod
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        lower: bool = False) -> np.ndarray:
        """
        Solve triangular linear system ax = b.
        
        Parameters
        ----------
        a : np.ndarray, shape (n, n)
            Triangular matrix (upper or lower)
        b : np.ndarray, shape (n,) or (n, k)
            Right-hand side vector(s)
        lower : bool, default=False
            True if 'a' is lower triangular, False if upper triangular
            
        Returns
        -------
        np.ndarray, shape (n,) or (n, k)
            Solution to the triangular system
            
        Notes
        -----
        Used extensively in likelihood computation for pattern-specific calculations.
        """
        pass
    
    @abstractmethod
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant of matrix.
        
        Parameters
        ----------
        matrix : np.ndarray, shape (n, n)
            Square matrix
            
        Returns
        -------
        sign : float
            Sign of the determinant (-1, 0, or 1)
        logdet : float
            Natural logarithm of absolute determinant
            
        Notes
        -----
        Essential for log-likelihood computation.
        For positive definite matrices, sign should always be +1.
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, objective_func: Callable, theta: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """
        Compute gradient of objective function at given parameter vector.
        
        Parameters
        ----------
        objective_func : callable
            Function that takes parameter vector and returns scalar objective value
        theta : np.ndarray, shape (n_params,)
            Parameter vector at which to evaluate gradient
        **kwargs : dict
            Backend-specific arguments
            
        Returns
        -------
        np.ndarray, shape (n_params,)
            Gradient vector
            
        Notes
        -----
        Implementation varies by backend:
        - CPU backends: Use finite differences 
        - GPU backends: Use autodiff for analytical gradients
        
        This is the key method that enables the two-track system.
        """
        pass


class GPUBackendBase(BackendInterface):
    """
    Abstract base class for GPU backends.
    
    Provides the contract for GPU-specific functionality while maintaining
    the core BackendInterface. GPU backends inherit common behavior patterns
    through this base class.
    
    Key Responsibilities:
    1. Define GPU-specific abstract methods
    2. Establish inheritance hierarchy for code reuse
    3. Declare autodiff gradient computation capability
    
    Inheritance Hierarchy:
    BackendInterface (pure interface)
    └── GPUBackendBase (GPU-specific interface)
        ├── PyTorchBackend (CUDA, Metal, Intel GPU)
        ├── JAXBackend (XLA compilation, TPU support)
        └── [Future GPU backends]
    """
    
    def gradient_method(self) -> str:
        """GPU backends use analytical gradients."""
        return 'autodiff'
    
    @abstractmethod
    def _create_tensor(self, array: np.ndarray, requires_grad: bool = False):
        """
        Create tensor appropriate for this backend.
        
        Parameters
        ----------
        array : np.ndarray
            NumPy array to convert to tensor
        requires_grad : bool, default=False
            Whether tensor should track gradients for autodiff
            
        Returns
        -------
        tensor
            Backend-specific tensor type
        """
        pass
    
    @abstractmethod
    def _tensor_to_numpy(self, tensor) -> np.ndarray:
        """
        Convert tensor back to NumPy array.
        
        Parameters
        ----------
        tensor
            Backend-specific tensor
            
        Returns
        -------
        np.ndarray
            NumPy array on CPU
        """
        pass


# Custom exception classes for backend operations
class BackendError(Exception):
    """Base exception for backend-related errors."""
    pass


class BackendNotAvailableError(BackendError):
    """Raised when requested backend is not available on this system."""
    pass


class NumericalError(BackendError):
    """Raised when numerical computation fails (e.g., non-positive definite matrix)."""
    pass


class GradientComputationError(BackendError):
    """Raised when gradient computation fails (finite differences or autodiff)."""
    pass