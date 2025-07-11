"""
Abstract backend interface for PyMVNMLE
Defines the contract that all computational backends must implement
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union


class BackendInterface(ABC):
    """
    Abstract interface for computational backends.
    
    All backends (NumPy, CuPy, Metal, JAX) must implement these methods
    to provide consistent linear algebra operations for ML estimation.
    
    The interface is designed around the core operations needed for
    multivariate normal maximum likelihood estimation:
    - Cholesky decomposition (for parameterization)
    - Triangular system solving (for likelihood computation)
    - Log-determinant computation (for likelihood)
    - Matrix operations (inversion, multiplication)
    
    Design Principles:
    - All methods accept NumPy arrays as input
    - All methods return NumPy arrays as output (via to_cpu if needed)
    - Backends handle their own memory management internally
    - Error handling should be consistent across backends
    """
    
    def __init__(self):
        """Initialize the backend and check availability."""
        self.available = False  # Set to True in subclass if backend works
        self.name = "base"      # Override in subclasses
        self.device_info = {}   # Backend-specific device information
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition of positive definite matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Positive definite matrix to decompose
        upper : bool, default=True
            If True, return upper triangular factor (U where A = U.T @ U)
            If False, return lower triangular factor (L where A = L @ L.T)
            
        Returns
        -------
        np.ndarray
            Cholesky factor as NumPy array
            
        Notes
        -----
        This is critical for our inverse Cholesky parameterization.
        Must handle near-singular matrices gracefully.
        """
        pass
    
    @abstractmethod
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool = True, trans: str = 'N') -> np.ndarray:
        """
        Solve triangular system ax = b.
        
        Parameters
        ----------
        a : np.ndarray
            Triangular matrix (2D)
        b : np.ndarray  
            Right-hand side (1D or 2D)
        upper : bool, default=True
            Whether a is upper or lower triangular
        trans : str, default='N'
            'N': solve ax = b
            'T': solve a.T x = b
            'C': solve a.H x = b (conjugate transpose)
            
        Returns
        -------
        np.ndarray
            Solution x as NumPy array
            
        Notes
        -----
        Used extensively in likelihood computation for solving
        with Cholesky factors.
        """
        pass
    
    @abstractmethod
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant of matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Square matrix
            
        Returns
        -------
        sign : float
            Sign of the determinant (+1, -1, or 0)
        logdet : float
            Natural logarithm of absolute determinant
            
        Notes
        -----
        Critical for likelihood computation. Should handle
        near-singular matrices by returning appropriate values.
        For positive definite matrices, sign should always be +1.
        """
        pass
    
    @abstractmethod
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse.
        
        Parameters
        ----------
        matrix : np.ndarray
            Invertible square matrix
            
        Returns
        -------
        np.ndarray
            Matrix inverse as NumPy array
            
        Notes
        -----
        Used for converting between parameterizations.
        Should use stable algorithms (e.g., LU decomposition).
        """
        pass
    
    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication.
        
        Parameters
        ----------
        a, b : np.ndarray
            Matrices to multiply
            
        Returns
        -------
        np.ndarray
            Product a @ b as NumPy array
        """
        pass
    
    @abstractmethod
    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer array to CPU memory as NumPy array.
        
        Parameters
        ----------
        array : backend-specific array type
            Array in backend's native format (CuPy array, JAX array, etc.)
            
        Returns
        -------
        np.ndarray
            Array converted to NumPy format on CPU
            
        Notes
        -----
        For NumPy backend, this is a no-op.
        For GPU backends, this transfers from GPU to CPU memory.
        Critical for returning results to user in consistent format.
        """
        pass
    
    @abstractmethod
    def asarray(self, array: Union[np.ndarray, list], dtype=None) -> object:
        """
        Convert input to backend's native array format.
        
        Parameters
        ----------
        array : array-like
            Input array or list
        dtype : data type, optional
            Desired data type
            
        Returns
        -------
        backend-specific array type
            Array in backend's native format
            
        Notes
        -----
        Converts NumPy arrays to backend format (CuPy, JAX, etc.)
        Used at start of each operation.
        """
        pass
    
    def get_device_info(self) -> dict:
        """
        Get information about computational device(s).
        
        Returns
        -------
        dict
            Device information including:
            - device_type: 'cpu', 'cuda', 'metal', 'tpu'
            - device_count: number of devices
            - memory_info: available memory (if applicable)
            - backend_version: version of backend library
        """
        return self.device_info.copy()
    
    def benchmark_operation(self, operation_name: str, 
                          matrix_size: int = 1000) -> float:
        """
        Benchmark a basic operation for performance testing.
        
        Parameters
        ----------
        operation_name : str
            Name of operation to benchmark ('cholesky', 'matmul', etc.)
        matrix_size : int, default=1000
            Size of test matrices
            
        Returns
        -------
        float
            Time in seconds for the operation
            
        Notes
        -----
        Used for backend auto-selection based on performance.
        Should run a standardized test for fair comparison.
        """
        import time
        
        # Generate test data
        np.random.seed(42)  # Reproducible benchmarks
        test_matrix = np.random.randn(matrix_size, matrix_size)
        test_matrix = test_matrix @ test_matrix.T  # Make positive definite
        
        # For Metal backend, use float32 to avoid MPS dtype limitations
        if self.name == "metal":
            test_matrix = test_matrix.astype(np.float32)
        
        # Convert to backend format
        backend_matrix = self.asarray(test_matrix)
        
        # Benchmark the requested operation
        start_time = time.time()
        
        if operation_name == 'cholesky':
            result = self.cholesky(test_matrix)
        elif operation_name == 'matmul':
            result = self.matmul(backend_matrix, backend_matrix)
        elif operation_name == 'inv':
            result = self.inv(test_matrix)
        elif operation_name == 'slogdet':
            result = self.slogdet(test_matrix)
        else:
            raise ValueError(f"Unknown operation: {operation_name}")
            
        end_time = time.time()
        
        # Ensure computation completed (for async backends)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
            
        return end_time - start_time
    
    def validate_positive_definite(self, matrix: np.ndarray, 
                                 name: str = "matrix") -> None:
        """
        Validate that matrix is positive definite.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix to validate
        name : str
            Name for error messages
            
        Raises
        ------
        ValueError
            If matrix is not positive definite
        """
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2-dimensional")
            
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
            
        # Check eigenvalues are positive
        eigenvals = np.linalg.eigvals(matrix)
        if np.any(eigenvals <= 0):
            raise ValueError(f"{name} is not positive definite")
    
    def __repr__(self) -> str:
        """String representation of backend."""
        status = "available" if self.is_available else "unavailable"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


class BackendError(Exception):
    """Base exception for backend-related errors."""
    pass


class BackendNotAvailableError(BackendError):
    """Raised when requested backend is not available on this system."""
    pass


class NumericalError(BackendError):
    """Raised when numerical computation fails (e.g., non-positive definite matrix)."""
    pass