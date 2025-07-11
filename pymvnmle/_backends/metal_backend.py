"""
Metal backend for PyMVNMLE
Apple Silicon GPU acceleration using PyTorch MPS
"""

import numpy as np
from typing import Tuple, Union
from .base import BackendInterface


class MetalBackend(BackendInterface):
    """
    Apple Silicon GPU backend using PyTorch Metal Performance Shaders (MPS).
    
    Provides GPU acceleration for Apple M-series chips (M1, M2, M3, M4) using
    PyTorch's Metal backend. Features unified memory architecture where CPU and
    GPU share the same memory pool, eliminating transfer overhead.
    
    Features:
    - Unified memory architecture (no CPU↔GPU transfers)
    - Optimized for Apple Silicon M-series chips
    - Metal Performance Shaders acceleration
    - Seamless integration with PyTorch ecosystem
    - Memory efficient due to shared memory pool
    - Native macOS optimization
    """
    
    def __init__(self):
        """Initialize Metal backend."""
        super().__init__()
        self.name = "metal"
        
        try:
            import torch
            
            # Check if MPS (Metal Performance Shaders) is available
            if not torch.backends.mps.is_available():
                self.available = False
                return
            
            # Check if MPS is built (compiled with Metal support)
            if not torch.backends.mps.is_built():
                self.available = False
                return
            
            self.torch = torch
            self.device = torch.device("mps")
            
            # Test basic operation to ensure Metal is working
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=self.device)
            _ = torch.sum(test_tensor)  # This will fail if Metal is not working
            
            self.available = True
            
            # Get device information
            self.device_info = self._get_device_info()
            
        except ImportError:
            self.available = False
            self.torch = None
        except Exception:
            # Metal operations failed
            self.available = False
            self.torch = None
    
    @property
    def is_available(self) -> bool:
        """Check if Metal backend is available and working."""
        return self.available
    
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition using PyTorch Metal.
        
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
            Cholesky factor as NumPy array
        """
        try:
            # Convert to float32 (MPS doesn't support float64)
            matrix_f32 = matrix.astype(np.float32)
            
            # Convert to PyTorch tensor on Metal device
            metal_tensor = self.torch.from_numpy(matrix_f32).to(self.device)
            
            # PyTorch cholesky returns upper triangular by default
            if upper:
                metal_result = self.torch.linalg.cholesky_ex(metal_tensor, upper=True)[0]
            else:
                metal_result = self.torch.linalg.cholesky_ex(metal_tensor, upper=False)[0]
            
            # Convert back to NumPy (keep float32 for consistency)
            return metal_result.cpu().numpy().astype(np.float64)
            
        except RuntimeError as e:
            if "not positive definite" in str(e).lower() or "cholesky" in str(e).lower():
                from .base import NumericalError
                raise NumericalError(f"Matrix is not positive definite: {e}")
            else:
                from .base import NumericalError
                raise NumericalError(f"Metal Cholesky decomposition failed: {e}")
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"Metal Cholesky decomposition failed: {e}")
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool = True, trans: str = 'N') -> np.ndarray:
        """
        Solve triangular system using PyTorch Metal.
        
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
        """
        try:
            # Convert to float32 (MPS doesn't support float64)
            a_f32 = a.astype(np.float32)
            b_f32 = b.astype(np.float32)
            
            # Convert to PyTorch tensors on Metal device
            metal_a = self.torch.from_numpy(a_f32).to(self.device)
            metal_b = self.torch.from_numpy(b_f32).to(self.device)
            
            # Handle transpose options
            if trans == 'T':
                metal_a = metal_a.T
            elif trans == 'C':
                metal_a = metal_a.conj().T
            elif trans != 'N':
                raise ValueError(f"trans must be 'N', 'T', or 'C', got {trans}")
            
            # PyTorch triangular solve
            metal_result = self.torch.triangular_solve(
                metal_b, metal_a, upper=upper
            )[0]
            
            # Convert back to NumPy and restore float64
            return metal_result.cpu().numpy().astype(np.float64)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"Metal triangular solve failed: {e}")
    
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant using PyTorch Metal.
        
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
            # Convert to float32 (MPS doesn't support float64)
            matrix_f32 = matrix.astype(np.float32)
            
            # Convert to PyTorch tensor on Metal device
            metal_tensor = self.torch.from_numpy(matrix_f32).to(self.device)
            
            # Compute log-determinant
            sign, logdet = self.torch.linalg.slogdet(metal_tensor)
            
            # Convert to Python floats (automatically transfers from Metal)
            return float(sign.cpu()), float(logdet.cpu())
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"Metal slogdet failed: {e}")
    
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse using PyTorch Metal.
        
        Parameters
        ----------
        matrix : np.ndarray
            Invertible square matrix
            
        Returns
        -------
        np.ndarray
            Matrix inverse as NumPy array
        """
        try:
            # Convert to float32 (MPS doesn't support float64)
            matrix_f32 = matrix.astype(np.float32)
            
            # Convert to PyTorch tensor on Metal device
            metal_tensor = self.torch.from_numpy(matrix_f32).to(self.device)
            
            # Compute inverse
            metal_result = self.torch.linalg.inv(metal_tensor)
            
            # Convert back to NumPy and restore float64
            return metal_result.cpu().numpy().astype(np.float64)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"Metal matrix inversion failed: {e}")
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using PyTorch Metal.
        
        Parameters
        ----------
        a, b : np.ndarray
            Matrices to multiply
            
        Returns
        -------
        np.ndarray
            Product a @ b as NumPy array
        """
        try:
            # Convert to float32 (MPS doesn't support float64)
            a_f32 = a.astype(np.float32)
            b_f32 = b.astype(np.float32)
            
            # Convert to PyTorch tensors on Metal device
            metal_a = self.torch.from_numpy(a_f32).to(self.device)
            metal_b = self.torch.from_numpy(b_f32).to(self.device)
            
            # Matrix multiplication
            metal_result = self.torch.matmul(metal_a, metal_b)
            
            # Convert back to NumPy and restore float64
            return metal_result.cpu().numpy().astype(np.float64)
            
        except Exception as e:
            raise ValueError(f"Metal matrix multiplication failed: {e}")
    
    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer array from Metal to CPU memory.
        
        Parameters
        ----------
        array : torch.Tensor or np.ndarray
            Array in PyTorch format or already on CPU
            
        Returns
        -------
        np.ndarray
            Array converted to NumPy format on CPU
            
        Notes
        -----
        For Metal backend, this transfers from unified memory (GPU) to
        CPU representation, though both share the same physical memory.
        """
        if hasattr(array, 'cpu'):
            # PyTorch tensor - transfer to CPU
            return array.cpu().numpy()
        else:
            # Already a NumPy array
            return np.asarray(array)
    
    def asarray(self, array: Union[np.ndarray, list], dtype=None) -> object:
        """
        Convert input to PyTorch tensor on Metal device.
        
        Parameters
        ----------
        array : array-like
            Input array or list
        dtype : data type, optional
            Desired data type
            
        Returns
        -------
        torch.Tensor
            Tensor in PyTorch format on Metal device
        """
        if dtype is not None:
            # Map NumPy dtypes to PyTorch dtypes if needed
            return self.torch.from_numpy(np.asarray(array, dtype=dtype)).to(self.device)
        else:
            return self.torch.from_numpy(np.asarray(array)).to(self.device)
    
    def _get_device_info(self) -> dict:
        """Get Metal device information."""
        try:
            import platform
            import psutil
            
            # Get system information
            system_info = {
                'device_type': 'metal',
                'device_count': 1,  # Apple Silicon has one integrated GPU
                'backend_version': self.torch.__version__,
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.machine(),
            }
            
            # Try to get memory information
            try:
                memory = psutil.virtual_memory()
                system_info['total_memory_gb'] = memory.total // (1024**3)
                system_info['available_memory_gb'] = memory.available // (1024**3)
            except Exception:
                system_info['total_memory_gb'] = 'unknown'
                system_info['available_memory_gb'] = 'unknown'
            
            # Try to detect Apple Silicon chip
            try:
                processor = platform.processor()
                if 'arm' in processor.lower() or platform.machine() == 'arm64':
                    if 'M1' in processor or 'Apple M1' in processor:
                        system_info['chip'] = 'Apple M1'
                    elif 'M2' in processor or 'Apple M2' in processor:
                        system_info['chip'] = 'Apple M2'
                    elif 'M3' in processor or 'Apple M3' in processor:
                        system_info['chip'] = 'Apple M3'
                    elif 'M4' in processor or 'Apple M4' in processor:
                        system_info['chip'] = 'Apple M4'
                    else:
                        system_info['chip'] = 'Apple Silicon (unknown variant)'
                else:
                    system_info['chip'] = 'Intel (MPS not recommended)'
            except Exception:
                system_info['chip'] = 'unknown'
            
            # MPS specific information
            system_info['mps_available'] = self.torch.backends.mps.is_available()
            system_info['mps_built'] = self.torch.backends.mps.is_built()
            
            return system_info
            
        except Exception as e:
            return {
                'device_type': 'metal',
                'device_count': 1,
                'error': str(e),
                'backend_version': self.torch.__version__ if self.torch else 'unknown'
            }
    
    def get_device_info(self) -> dict:
        """
        Get Metal device information.
        
        Returns
        -------
        dict
            Device information including Apple Silicon details and memory
        """
        return self.device_info.copy()
    
    def __repr__(self) -> str:
        """String representation with Apple Silicon info."""
        if not self.available:
            return "MetalBackend(unavailable)"
        
        if hasattr(self, 'device_info'):
            chip = self.device_info.get('chip', 'unknown')
            memory_gb = self.device_info.get('total_memory_gb', 'unknown')
            return f"MetalBackend(chip='{chip}', memory={memory_gb}GB unified)"
        
        return "MetalBackend(Apple Silicon)"