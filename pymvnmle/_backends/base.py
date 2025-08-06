"""
Base interfaces for computational backends in PyMVNMLE.

Defines abstract base classes that all backends must implement.
Separates concerns by precision level (FP32 vs FP64).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class BackendBase(ABC):
    """
    Abstract base class for all computational backends.
    
    Every backend must implement these core operations.
    No defaults - explicit implementation required.
    """
    
    def __init__(self, precision: str):
        """
        Initialize backend with precision specification.
        
        Parameters
        ----------
        precision : str
            Either 'fp32' or 'fp64'
        """
        if precision not in ['fp32', 'fp64']:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp32' or 'fp64'")
        
        self.precision = precision
        self.dtype = np.float32 if precision == 'fp32' else np.float64
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on current hardware.
        
        Returns
        -------
        bool
            True if backend can be used
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the compute device.
        
        Returns
        -------
        dict
            Device information including name, memory, capabilities
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer array to backend's compute device.
        
        Parameters
        ----------
        array : np.ndarray
            NumPy array to transfer
            
        Returns
        -------
        device_array
            Array on target device (format depends on backend)
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_numpy(self, device_array: Any) -> np.ndarray:
        """
        Transfer array from device back to NumPy.
        
        Parameters
        ----------
        device_array
            Array on device
            
        Returns
        -------
        np.ndarray
            NumPy array on CPU
        """
        raise NotImplementedError
    
    @abstractmethod
    def cholesky(self, matrix: Any, upper: bool) -> Any:
        """
        Compute Cholesky decomposition.
        
        Parameters
        ----------
        matrix
            Positive definite matrix (on device)
        upper : bool
            If True, compute upper triangular. If False, lower.
            
        Returns
        -------
        cholesky_factor
            Triangular Cholesky factor (on device)
            
        Raises
        ------
        LinAlgError
            If matrix is not positive definite
        """
        raise NotImplementedError
    
    @abstractmethod
    def solve_triangular(self, a: Any, b: Any, upper: bool, trans: str) -> Any:
        """
        Solve triangular system a @ x = b or a.T @ x = b.
        
        Parameters
        ----------
        a
            Triangular matrix (on device)
        b
            Right-hand side (on device)
        upper : bool
            Whether a is upper triangular
        trans : str
            'N' for a @ x = b, 'T' for a.T @ x = b
            
        Returns
        -------
        x
            Solution (on device)
        """
        raise NotImplementedError
    
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Matrix multiplication.
        
        Parameters
        ----------
        a, b
            Matrices to multiply (on device)
            
        Returns
        -------
        product
            Matrix product a @ b (on device)
        """
        raise NotImplementedError
    
    @abstractmethod
    def inv(self, matrix: Any) -> Any:
        """
        Matrix inversion.
        
        Parameters
        ----------
        matrix
            Square matrix to invert (on device)
            
        Returns
        -------
        inverse
            Matrix inverse (on device)
            
        Raises
        ------
        LinAlgError
            If matrix is singular
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_det(self, matrix: Any) -> float:
        """
        Compute log determinant of positive definite matrix.
        
        Parameters
        ----------
        matrix
            Positive definite matrix (on device)
            
        Returns
        -------
        float
            Natural logarithm of determinant
            
        Notes
        -----
        Should use Cholesky decomposition for numerical stability:
        log_det(A) = 2 * sum(log(diag(cholesky(A))))
        """
        raise NotImplementedError
    
    @abstractmethod
    def quadratic_form(self, x: Any, A: Any) -> float:
        """
        Compute quadratic form x.T @ A @ x.
        
        Parameters
        ----------
        x
            Vector (on device)
        A
            Positive definite matrix (on device)
            
        Returns
        -------
        float
            Scalar result of quadratic form
        """
        raise NotImplementedError


class CPUBackend(BackendBase):
    """
    Base class for CPU backends.
    
    Always uses FP64 for R compatibility.
    """
    
    def __init__(self):
        """Initialize CPU backend with FP64 precision."""
        super().__init__(precision='fp64')
        self.device_type = 'cpu'
    
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        import platform
        import psutil
        
        return {
            'device_type': 'cpu',
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'precision': self.precision
        }


class GPUBackendFP32(BackendBase):
    """
    Base class for FP32 GPU backends.
    
    Optimized for consumer GPUs (RTX, Apple Metal).
    Uses BFGS optimization method.
    """
    
    def __init__(self):
        """Initialize GPU backend with FP32 precision."""
        super().__init__(precision='fp32')
        self.device_type = 'gpu'
        self.optimization_method = 'BFGS'
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current GPU memory usage.
        
        Returns
        -------
        dict
            Memory usage statistics (allocated, cached, total)
        """
        raise NotImplementedError
    
    def supports_autodiff(self) -> bool:
        """
        Check if backend supports automatic differentiation.
        
        Returns
        -------
        bool
            True for PyTorch/JAX backends
        """
        return False  # Override in implementations that support it


class GPUBackendFP64(BackendBase):
    """
    Base class for FP64 GPU backends.
    
    For data center GPUs (A100, H100) with full FP64 support.
    Uses Newton-CG optimization method.
    """
    
    def __init__(self):
        """Initialize GPU backend with FP64 precision."""
        super().__init__(precision='fp64')
        self.device_type = 'gpu'
        self.optimization_method = 'Newton-CG'
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current GPU memory usage.
        
        Returns
        -------
        dict
            Memory usage statistics (allocated, cached, total)
        """
        raise NotImplementedError
    
    @abstractmethod
    def check_fp64_performance(self) -> float:
        """
        Benchmark FP64 performance ratio.
        
        Returns
        -------
        float
            Ratio of FP64 to FP32 throughput
            
        Notes
        -----
        Should run a small benchmark to verify FP64 is not gimped.
        Expected: ~0.5 for full FP64, ~0.015 for gimped.
        """
        raise NotImplementedError
    
    def supports_autodiff(self) -> bool:
        """
        Check if backend supports automatic differentiation.
        
        Returns
        -------
        bool
            True for PyTorch/JAX backends
        """
        return False  # Override in implementations that support it


class BackendFactory:
    """
    Factory for creating appropriate backend based on precision and hardware.
    
    This is the main entry point for backend selection.
    """
    
    @staticmethod
    def create_backend(use_fp64: bool, device_type: str) -> BackendBase:
        """
        Create appropriate backend based on precision and device.
        
        Parameters
        ----------
        use_fp64 : bool
            Whether to use FP64 precision
        device_type : str
            'cpu', 'cuda', or 'metal'
            
        Returns
        -------
        BackendBase
            Concrete backend implementation
            
        Raises
        ------
        ImportError
            If required backend library not available
        RuntimeError
            If requested configuration not supported
        """
        if device_type == 'cpu':
            from .cpu_fp64_backend import NumpyBackendFP64
            return NumpyBackendFP64()
        
        elif device_type == 'cuda':
            if use_fp64:
                from .gpu_fp64_backend import PyTorchBackendFP64
                return PyTorchBackendFP64()
            else:
                from .gpu_fp32_backend import PyTorchBackendFP32
                return PyTorchBackendFP32()
        
        elif device_type == 'metal':
            if use_fp64:
                raise RuntimeError(
                    "FP64 not supported on Apple Metal. "
                    "Please use FP32 (use_fp64=False) or CPU backend."
                )
            from .gpu_fp32_backend import PyTorchBackendFP32
            return PyTorchBackendFP32()
        
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    @staticmethod
    def get_optimal_backend(use_fp64: Optional[bool] = None) -> BackendBase:
        """
        Automatically select optimal backend based on hardware.
        
        Parameters
        ----------
        use_fp64 : bool or None
            If None, auto-select based on hardware capabilities
            
        Returns
        -------
        BackendBase
            Optimal backend for current hardware
        """
        from .precision_detector import detect_gpu_capabilities, recommend_precision
        
        # Detect hardware
        caps = detect_gpu_capabilities()
        
        # Determine precision
        if use_fp64 is None:
            use_fp64 = recommend_precision(caps, None)
        else:
            # Validate user request
            from .precision_detector import validate_fp64_request
            validate_fp64_request(caps, use_fp64)
        
        # Select backend
        if not caps.has_gpu:
            device_type = 'cpu'
        else:
            device_type = caps.gpu_type
        
        return BackendFactory.create_backend(use_fp64, device_type)