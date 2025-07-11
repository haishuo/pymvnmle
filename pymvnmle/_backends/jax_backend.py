"""
JAX backend for PyMVNMLE
TPU/GPU acceleration using JAX with XLA compilation
"""

import numpy as np
from typing import Tuple, Union
from .base import BackendInterface


class JAXBackend(BackendInterface):
    """
    JAX backend with CPU/GPU/TPU support via XLA compilation.
    
    JAX provides automatic differentiation and XLA compilation for
    high-performance numerical computing. It works on CPU, GPU, and TPU
    with the same code, making it ideal for diverse hardware environments.
    
    Features:
    - XLA compilation for optimized performance
    - Automatic differentiation (useful for future extensions)
    - CPU/GPU/TPU support with same code
    - Functional programming paradigm
    - Good support for new GPU architectures via XLA
    - Memory-efficient for large computations
    """
    
    def __init__(self):
        """Initialize JAX backend."""
        super().__init__()
        self.name = "jax"
        
        try:
            import jax
            import jax.numpy as jnp
            
            self.jax = jax
            self.jnp = jnp
            
            # Check available devices
            devices = jax.devices()
            if len(devices) == 0:
                self.available = False
                return
            
            # Test basic operation to ensure JAX is working
            test_array = jnp.array([1.0, 2.0, 3.0])
            _ = jnp.sum(test_array)  # This will fail if JAX is not working
            
            self.available = True
            self.devices = devices
            
            # Get device information
            self.device_info = self._get_device_info()
            
        except ImportError:
            self.available = False
            self.jax = None
            self.jnp = None
        except Exception:
            # JAX operations failed
            self.available = False
            self.jax = None
            self.jnp = None
    
    @property
    def is_available(self) -> bool:
        """Check if JAX backend is available and working."""
        return self.available
    
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition using JAX.
        
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
            # Convert to JAX array
            jax_matrix = self.jnp.asarray(matrix)
            
            # JAX cholesky returns lower triangular by default
            if upper:
                # For upper triangular, compute lower then transpose
                jax_lower = self.jnp.linalg.cholesky(jax_matrix)
                jax_result = jax_lower.T
            else:
                # Lower triangular (default)
                jax_result = self.jnp.linalg.cholesky(jax_matrix)
            
            # Convert back to NumPy
            return np.asarray(jax_result)
            
        except Exception as e:
            # JAX doesn't have a specific LinAlgError, check error message
            if "not positive definite" in str(e).lower() or "cholesky" in str(e).lower():
                from .base import NumericalError
                raise NumericalError(f"Matrix is not positive definite: {e}")
            else:
                from .base import NumericalError
                raise NumericalError(f"JAX Cholesky decomposition failed: {e}")
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool = True, trans: str = 'N') -> np.ndarray:
        """
        Solve triangular system using JAX.
        
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
            # Convert to JAX arrays
            jax_a = self.jnp.asarray(a)
            jax_b = self.jnp.asarray(b)
            
            # Handle transpose options
            if trans == 'T':
                jax_a = jax_a.T
            elif trans == 'C':
                jax_a = jax_a.conj().T
            elif trans != 'N':
                raise ValueError(f"trans must be 'N', 'T', or 'C', got {trans}")
            
            # JAX solve_triangular
            jax_result = self.jax.scipy.linalg.solve_triangular(
                jax_a, jax_b, lower=not upper
            )
            
            # Convert back to NumPy
            return np.asarray(jax_result)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"JAX triangular solve failed: {e}")
    
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant using JAX.
        
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
            # Convert to JAX array
            jax_matrix = self.jnp.asarray(matrix)
            
            # Compute log-determinant
            sign, logdet = self.jnp.linalg.slogdet(jax_matrix)
            
            # Convert to Python floats
            return float(sign), float(logdet)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"JAX slogdet failed: {e}")
    
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse using JAX.
        
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
            # Convert to JAX array
            jax_matrix = self.jnp.asarray(matrix)
            
            # Compute inverse
            jax_result = self.jnp.linalg.inv(jax_matrix)
            
            # Convert back to NumPy
            return np.asarray(jax_result)
            
        except Exception as e:
            from .base import NumericalError
            raise NumericalError(f"JAX matrix inversion failed: {e}")
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using JAX.
        
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
            # Convert to JAX arrays
            jax_a = self.jnp.asarray(a)
            jax_b = self.jnp.asarray(b)
            
            # Matrix multiplication
            jax_result = self.jnp.matmul(jax_a, jax_b)
            
            # Convert back to NumPy
            return np.asarray(jax_result)
            
        except Exception as e:
            raise ValueError(f"JAX matrix multiplication failed: {e}")
    
    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer array to CPU memory as NumPy array.
        
        Parameters
        ----------
        array : jax.Array or np.ndarray
            Array in JAX format or already on CPU
            
        Returns
        -------
        np.ndarray
            Array converted to NumPy format on CPU
        """
        # JAX arrays can be converted directly to NumPy
        return np.asarray(array)
    
    def asarray(self, array: Union[np.ndarray, list], dtype=None) -> object:
        """
        Convert input to JAX array.
        
        Parameters
        ----------
        array : array-like
            Input array or list
        dtype : data type, optional
            Desired data type
            
        Returns
        -------
        jax.Array
            Array in JAX format
        """
        return self.jnp.asarray(array, dtype=dtype)
    
    def _get_device_info(self) -> dict:
        """Get detailed device information for all JAX devices."""
        try:
            devices_info = []
            device_types = set()
            
            for i, device in enumerate(self.devices):
                device_str = str(device)
                
                # Parse device information - fix the detection logic
                if 'cuda' in device_str.lower() or 'gpu' in device_str.lower():
                    device_type = 'cuda'
                    platform = 'cuda'
                elif 'tpu' in device_str.lower():
                    device_type = 'tpu'
                    platform = 'tpu'
                else:
                    device_type = 'cpu'
                    platform = 'cpu'
                
                device_types.add(device_type)
                
                device_info = {
                    'id': i,
                    'platform': platform,
                    'device_kind': device.device_kind,
                    'process_index': device.process_index,
                    'device_str': device_str
                }
                
                devices_info.append(device_info)
            
            # Determine primary device type
            if 'tpu' in device_types:
                primary_type = 'tpu'
            elif 'cuda' in device_types:
                primary_type = 'cuda'
            else:
                primary_type = 'cpu'
            
            return {
                'device_type': primary_type,
                'device_count': len(self.devices),
                'devices': devices_info,
                'backend_version': self.jax.__version__,
                'xla_flags': self._get_xla_flags(),
                'default_device': str(self.jax.devices()[0]) if self.devices else 'none'
            }
            
        except Exception as e:
            return {
                'device_type': 'unknown',
                'device_count': len(self.devices) if hasattr(self, 'devices') else 0,
                'error': str(e),
                'backend_version': self.jax.__version__ if self.jax else 'unknown'
            }
    
    def _get_xla_flags(self) -> dict:
        """Get XLA compilation flags and settings."""
        try:
            import os
            xla_info = {}
            
            # Check common XLA environment variables
            xla_vars = [
                'XLA_FLAGS',
                'XLA_PYTHON_CLIENT_PREALLOCATE',
                'XLA_PYTHON_CLIENT_MEM_FRACTION',
                'XLA_PYTHON_CLIENT_ALLOCATOR'
            ]
            
            for var in xla_vars:
                if var in os.environ:
                    xla_info[var] = os.environ[var]
            
            return xla_info
            
        except Exception:
            return {}
    
    def get_device_info(self) -> dict:
        """
        Get JAX device information.
        
        Returns
        -------
        dict
            Device information including available devices and XLA settings
        """
        return self.device_info.copy()
    
    def __repr__(self) -> str:
        """String representation with device info."""
        if not self.available:
            return "JAXBackend(unavailable)"
        
        if hasattr(self, 'device_info'):
            device_type = self.device_info.get('device_type', 'unknown')
            device_count = self.device_info.get('device_count', 0)
            return f"JAXBackend(devices={device_count}×{device_type})"
        
        return f"JAXBackend(devices={len(self.devices)})"