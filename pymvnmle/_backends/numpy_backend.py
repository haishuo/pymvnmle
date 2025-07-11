"""
NumPy backend for PyMVNMLE
CPU-based reference implementation using NumPy and SciPy
"""

import numpy as np
import scipy.linalg as linalg
from typing import Tuple, Union
from .base import BackendInterface


class NumPyBackend(BackendInterface):
    """
    CPU backend using NumPy and SciPy.
    
    This serves as the reference implementation that all other backends
    must match numerically. Uses mature, well-tested NumPy/SciPy routines
    for maximum reliability and compatibility.
    
    Features:
    - Always available (no optional dependencies)
    - Numerically stable algorithms from LAPACK
    - Comprehensive error handling
    - Optimized for Intel MKL when available
    - Serves as fallback for all other backends
    """
    
    def __init__(self):
        """Initialize NumPy backend."""
        super().__init__()
        self.name = "numpy"
        self.available = True  # NumPy is always available
        
        # Get device and version information
        self.device_info = {
            'device_type': 'cpu',
            'device_count': 1,
            'backend_version': np.__version__,
            'scipy_version': getattr(linalg, '__version__', 'unknown'),
            'blas_info': self._get_blas_info(),
            'processor_count': self._get_processor_count()
        }
    
    @property
    def is_available(self) -> bool:
        """NumPy backend is always available."""
        return True
    
    def cholesky(self, matrix: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute Cholesky decomposition using scipy.linalg.cholesky.
        
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
            Cholesky factor
            
        Notes
        -----
        Uses scipy.linalg.cholesky which provides better error handling
        than np.linalg.cholesky and supports both upper/lower triangular output.
        """
        try:
            # scipy.linalg.cholesky with lower=False gives upper triangular
            return linalg.cholesky(matrix, lower=not upper)
        except linalg.LinAlgError as e:
            if "not positive definite" in str(e).lower():
                from .base import NumericalError
                raise NumericalError(f"Matrix is not positive definite: {e}")
            else:
                raise
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool = True, trans: str = 'N') -> np.ndarray:
        """
        Solve triangular system using scipy.linalg.solve_triangular.
        
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
            Solution x
            
        Notes
        -----
        scipy.linalg.solve_triangular is optimized and numerically stable.
        Automatically detects if matrix is singular.
        """
        try:
            # Convert trans parameter to scipy format
            trans_map = {'N': 0, 'T': 1, 'C': 2}
            if trans not in trans_map:
                raise ValueError(f"trans must be 'N', 'T', or 'C', got {trans}")
            
            return linalg.solve_triangular(
                a, b, 
                lower=not upper, 
                trans=trans_map[trans],
                check_finite=True
            )
        except linalg.LinAlgError as e:
            from .base import NumericalError
            raise NumericalError(f"Failed to solve triangular system: {e}")
    
    def slogdet(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute sign and log-determinant using scipy.linalg.slogdet.
        
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
        scipy.linalg.slogdet is numerically stable for computing log-determinants
        and avoids overflow/underflow issues that plague direct determinant computation.
        """
        try:
            sign, logdet = linalg.slogdet(matrix)
            return float(sign), float(logdet)
        except linalg.LinAlgError as e:
            from .base import NumericalError
            raise NumericalError(f"Failed to compute log-determinant: {e}")
    
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse using scipy.linalg.inv.
        
        Parameters
        ----------
        matrix : np.ndarray
            Invertible square matrix
            
        Returns
        -------
        np.ndarray
            Matrix inverse
            
        Notes
        -----
        Uses LU decomposition via LAPACK for numerical stability.
        Automatically checks for singularity.
        """
        try:
            return linalg.inv(matrix, check_finite=True)
        except linalg.LinAlgError as e:
            from .base import NumericalError
            raise NumericalError(f"Matrix is singular and cannot be inverted: {e}")
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using NumPy's optimized implementation.
        
        Parameters
        ----------
        a, b : np.ndarray
            Matrices to multiply
            
        Returns
        -------
        np.ndarray
            Product a @ b
            
        Notes
        -----
        Uses np.matmul which automatically selects the best BLAS routine
        and handles broadcasting correctly.
        """
        try:
            return np.matmul(a, b)
        except ValueError as e:
            raise ValueError(f"Matrix multiplication failed: {e}")
    
    def to_cpu(self, array) -> np.ndarray:
        """
        Convert array to CPU (no-op for NumPy backend).
        
        Parameters
        ----------
        array : np.ndarray
            Array already on CPU
            
        Returns
        -------
        np.ndarray
            Same array (no conversion needed)
        """
        return np.asarray(array)
    
    def asarray(self, array: Union[np.ndarray, list], dtype=None) -> np.ndarray:
        """
        Convert input to NumPy array.
        
        Parameters
        ----------
        array : array-like
            Input array or list
        dtype : data type, optional
            Desired data type
            
        Returns
        -------
        np.ndarray
            NumPy array
        """
        return np.asarray(array, dtype=dtype)
    
    def _get_blas_info(self) -> dict:
        """Get BLAS/LAPACK configuration information."""
        try:
            # Try multiple methods to get BLAS info
            result = {
                'library': 'unknown',
                'version': 'unknown',
                'threading': 'unknown'
            }
            
            # Method 1: Try numpy.show_config() - most reliable
            try:
                config = np.show_config(mode='dicts')
                
                # Check different possible keys
                blas_keys = ['blas_info', 'blas_mkl_info', 'mkl_info', 'blas_opt_info']
                blas_info = None
                
                for key in blas_keys:
                    if key in config:
                        blas_info = config[key]
                        break
                
                if blas_info:
                    # Extract library name
                    if 'libraries' in blas_info:
                        libs = blas_info['libraries']
                        if isinstance(libs, list) and libs:
                            lib_str = ' '.join(libs).lower()
                        else:
                            lib_str = str(libs).lower()
                        
                        # Identify BLAS library
                        if 'mkl' in lib_str:
                            result['library'] = 'Intel MKL'
                        elif 'openblas' in lib_str:
                            result['library'] = 'OpenBLAS'
                        elif 'atlas' in lib_str:
                            result['library'] = 'ATLAS'
                        elif 'accelerate' in lib_str:
                            result['library'] = 'Accelerate'
                        elif 'blis' in lib_str:
                            result['library'] = 'BLIS'
                        else:
                            result['library'] = f"Custom ({lib_str})"
                    
                    # Try to extract version
                    if 'version' in blas_info:
                        result['version'] = str(blas_info['version'])
                
            except Exception:
                pass
            
            # Method 2: Check numpy.__config__ if available
            if result['library'] == 'unknown':
                try:
                    if hasattr(np, '__config__'):
                        config = np.__config__
                        if hasattr(config, 'blas_info'):
                            blas_info = config.blas_info
                            if blas_info and 'libraries' in blas_info:
                                libs = str(blas_info['libraries']).lower()
                                if 'mkl' in libs:
                                    result['library'] = 'Intel MKL'
                                elif 'openblas' in libs:
                                    result['library'] = 'OpenBLAS'
                except Exception:
                    pass
            
            # Method 3: Try importing scipy and checking its BLAS
            if result['library'] == 'unknown':
                try:
                    import scipy
                    if hasattr(scipy, 'show_config'):
                        config = scipy.show_config(mode='dicts')
                        if 'blas_info' in config:
                            blas_info = config['blas_info']
                            if 'libraries' in blas_info:
                                libs = str(blas_info['libraries']).lower()
                                if 'mkl' in libs:
                                    result['library'] = 'Intel MKL'
                                elif 'openblas' in libs:
                                    result['library'] = 'OpenBLAS'
                except Exception:
                    pass
            
            # Method 4: Try direct library detection via ctypes (last resort)
            if result['library'] == 'unknown':
                try:
                    import ctypes.util
                    # Look for common BLAS libraries
                    for lib_name, display_name in [
                        ('mkl_rt', 'Intel MKL'),
                        ('openblas', 'OpenBLAS'),
                        ('atlas', 'ATLAS'),
                        ('accelerate', 'Accelerate')
                    ]:
                        if ctypes.util.find_library(lib_name):
                            result['library'] = display_name
                            break
                except Exception:
                    pass
            
            return result
            
        except Exception:
            # If all methods fail, return basic unknown info
            return {
                'library': 'unknown',
                'version': 'unknown', 
                'threading': 'unknown'
            }
    
    def _get_processor_count(self) -> int:
        """Get number of available CPU cores."""
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1
    
    def get_device_info(self) -> dict:
        """
        Get CPU and BLAS information.
        
        Returns
        -------
        dict
            Device information including CPU count and BLAS library details
        """
        return self.device_info.copy()
    
    def __repr__(self) -> str:
        """String representation with BLAS info."""
        blas_lib = self.device_info.get('blas_info', {}).get('library', 'unknown')
        cpu_count = self.device_info.get('processor_count', 1)
        return f"NumPyBackend(cpus={cpu_count}, blas='{blas_lib}')"