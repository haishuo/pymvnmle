"""
CuPy backend for PyMVNMLE
NVIDIA GPU acceleration using CuPy
"""

import numpy as np
from typing import Tuple, Union
from .base import BackendInterface


class CuPyBackend(BackendInterface):
    """
    NVIDIA GPU backend using CuPy.
    
    Provides GPU acceleration for NVIDIA graphics cards using CUDA.
    CuPy offers a NumPy-compatible API with automatic GPU memory management
    and optimized CUDA kernels for linear algebra operations.
    
    Features:
    - High-performance GPU acceleration via CUDA
    - Automatic memory management
    - Optimized cuBLAS/cuSOLVER integration
    - Seamless NumPy compatibility
    - Support for multiple GPU devices
    """
    
    def __init__(self):
        """Initialize CuPy backend."""
        super().__init__()
        self.name = "cupy"
        
        try:
            import cupy as cp
            import cupyx.scipy.linalg as cu_linalg
            
            self.cp = cp
            self.cu_linalg = cu_linalg
            
            # Check if CUDA is available and working
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                self.available = False
                return
            
            # Test basic GPU operation
            test_array = cp.array([1.0, 2.0, 3.0])
            _ = cp.sum(test_array)  # This will fail if GPU is not working
            
            self.available = True
            self.device_count = device_count
            
            # Get device information
            self.device_info = self._get_device_info()
            
        except ImportError:
            self.available = False
            self.cp = None
            self.cu_linalg = None
        except Exception:
            # GPU operations failed
            self.available = False
            self.cp = None
            self.cu_linalg = None
    
    @property
    def is_available(self) -> bool:
        """Check if CuPy backend is available and working."""
        return self.available
    
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition on GPU using CuPy.
        
        Parameters
        ----------
        matrix : np.ndarray
            Positive definite matrix to decompose
        upper : bool, default=True
            If True, return upper triangular factor
            If False, return lower triangular factor
            
        Returns
        -------
        np.ndarray
            Cholesky factor as NumPy array (transferred from GPU)
        """
        try:
            # Transfer to GPU
            gpu_matrix = self.cp.asarray(matrix)
            
            # Use cupy.linalg.cholesky (not cupyx.scipy.linalg)
            # CuPy's cholesky returns upper triangular by default
            if upper:
                gpu_result = self.cp.linalg.cholesky(gpu_matrix)
            else:
                # For lower triangular, transpose the result
                gpu_upper = self.cp.linalg.cholesky(gpu_matrix)
                gpu_result = gpu_upper.T
            
            # Transfer back to CPU
            return self.cp.asnumpy(gpu_result)
            
        except self.cp.linalg.LinAlgError as e:
            if "not positive definite" in str(e).lower():
                from .base import NumericalError
                raise NumericalError(f"Matrix is not positive definite: {e}")
            else:
                raise
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"CuPy Cholesky decomposition failed: {e}")
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool = True, trans: str = 'N') -> np.ndarray:
        """
        Solve triangular system on GPU using CuPy.
        
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
            Solution x as NumPy array (transferred from GPU)
        """
        try:
            # Transfer to GPU
            gpu_a = self.cp.asarray(a)
            gpu_b = self.cp.asarray(b)
            
            # Use cupyx.scipy.linalg.solve_triangular
            # Convert trans parameter
            trans_map = {'N': 0, 'T': 1, 'C': 2}
            if trans not in trans_map:
                raise ValueError(f"trans must be 'N', 'T', or 'C', got {trans}")
            
            # Solve triangular system
            gpu_result = self.cu_linalg.solve_triangular(
                gpu_a, gpu_b,
                lower=not upper,
                trans=trans_map[trans]
            )
            
            # Transfer back to CPU
            return self.cp.asnumpy(gpu_result)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"CuPy triangular solve failed: {e}")
    
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant on GPU using CuPy.
        
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
        """
        try:
            # Transfer to GPU
            gpu_matrix = self.cp.asarray(matrix)
            
            # Use cupy.linalg.slogdet
            sign, logdet = self.cp.linalg.slogdet(gpu_matrix)
            
            # Convert to Python floats (automatically transfers from GPU)
            return float(sign), float(logdet)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"CuPy slogdet failed: {e}")
    
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse on GPU using CuPy.
        
        Parameters
        ----------
        matrix : np.ndarray
            Invertible square matrix
            
        Returns
        -------
        np.ndarray
            Matrix inverse as NumPy array (transferred from GPU)
        """
        try:
            # Transfer to GPU
            gpu_matrix = self.cp.asarray(matrix)
            
            # Use cupy.linalg.inv
            gpu_result = self.cp.linalg.inv(gpu_matrix)
            
            # Transfer back to CPU
            return self.cp.asnumpy(gpu_result)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"CuPy matrix inversion failed: {e}")
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication on GPU using CuPy.
        
        Parameters
        ----------
        a, b : np.ndarray
            Matrices to multiply
            
        Returns
        -------
        np.ndarray
            Product a @ b as NumPy array (transferred from GPU)
        """
        try:
            # Transfer to GPU
            gpu_a = self.cp.asarray(a)
            gpu_b = self.cp.asarray(b)
            
            # Matrix multiplication
            gpu_result = self.cp.matmul(gpu_a, gpu_b)
            
            # Transfer back to CPU
            return self.cp.asnumpy(gpu_result)
            
        except Exception as e:
            raise ValueError(f"CuPy matrix multiplication failed: {e}")
    
    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer array from GPU to CPU memory.
        
        Parameters
        ----------
        array : cupy.ndarray or np.ndarray
            Array in CuPy format or already on CPU
            
        Returns
        -------
        np.ndarray
            Array converted to NumPy format on CPU
        """
        if hasattr(array, 'get'):
            # CuPy array - transfer to CPU
            return array.get()
        else:
            # Already a NumPy array
            return np.asarray(array)
    
    def asarray(self, array: Union[np.ndarray, list], dtype=None) -> object:
        """
        Convert input to CuPy array on GPU.
        
        Parameters
        ----------
        array : array-like
            Input array or list
        dtype : data type, optional
            Desired data type
            
        Returns
        -------
        cupy.ndarray
            Array in CuPy format on GPU
        """
        return self.cp.asarray(array, dtype=dtype)
    
    def _get_device_info(self) -> dict:
        """Get detailed GPU device information."""
        try:
            devices = []
            total_memory = 0
            
            for i in range(self.device_count):
                with self.cp.cuda.Device(i):
                    props = self.cp.cuda.runtime.getDeviceProperties(i)
                    device_memory = props['totalGlobalMem']
                    
                    device_info = {
                        'id': i,
                        'name': props['name'].decode(),
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'memory_gb': device_memory // (1024**3),
                        'memory_bytes': device_memory,
                        'multiprocessors': props['multiProcessorCount'],
                        'max_threads_per_block': props['maxThreadsPerBlock']
                    }
                    devices.append(device_info)
                    total_memory += device_memory
            
            return {
                'device_type': 'cuda',
                'device_count': self.device_count,
                'devices': devices,
                'total_memory_gb': total_memory // (1024**3),
                'backend_version': self.cp.__version__,
                'cuda_version': self.cp.cuda.runtime.runtimeGetVersion(),
                'driver_version': self.cp.cuda.runtime.driverGetVersion()
            }
            
        except Exception as e:
            return {
                'device_type': 'cuda',
                'device_count': self.device_count if hasattr(self, 'device_count') else 0,
                'error': str(e),
                'backend_version': self.cp.__version__ if self.cp else 'unknown'
            }
    
    def get_device_info(self) -> dict:
        """
        Get GPU device information.
        
        Returns
        -------
        dict
            Device information including GPU details and memory
        """
        return self.device_info.copy()
    
    def __repr__(self) -> str:
        """String representation with GPU info."""
        if not self.available:
            return "CuPyBackend(unavailable)"
        
        if hasattr(self, 'device_info') and 'devices' in self.device_info:
            devices = self.device_info['devices']
            if devices:
                gpu_name = devices[0]['name']
                memory_gb = devices[0]['memory_gb']
                return f"CuPyBackend(gpu='{gpu_name}', memory={memory_gb}GB)"
        
        return f"CuPyBackend(devices={self.device_count})"