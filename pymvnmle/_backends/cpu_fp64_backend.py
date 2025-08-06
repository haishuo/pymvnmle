"""
CPU backend using NumPy with FP64 precision.

This backend maintains exact R compatibility using NumPy operations.
Always uses FP64 for numerical precision required by Newton-type methods.
"""

import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional
from .base import CPUBackend


class NumpyBackendFP64(CPUBackend):
    """
    NumPy-based CPU backend with FP64 precision.
    
    This is the reference implementation that maintains exact
    compatibility with R's mvnmle package.
    
    Notes
    -----
    - Always uses float64 for R compatibility
    - Uses scipy.linalg for robust linear algebra
    - No GPU acceleration, but works everywhere
    - Optimized for correctness over speed
    """
    
    def __init__(self):
        """Initialize NumPy backend."""
        super().__init__()
        self.name = 'numpy_fp64'
        
        # Verify NumPy and SciPy versions
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Verify required dependencies are available."""
        try:
            import numpy as np
            import scipy
            
            # Check versions for known issues
            np_version = tuple(map(int, np.__version__.split('.')[:2]))
            sp_version = tuple(map(int, scipy.__version__.split('.')[:2]))
            
            if np_version < (1, 20):
                raise RuntimeError(
                    f"NumPy version {np.__version__} is too old. "
                    f"Please upgrade to NumPy >= 1.20.0"
                )
            
            if sp_version < (1, 7):
                raise RuntimeError(
                    f"SciPy version {scipy.__version__} is too old. "
                    f"Please upgrade to SciPy >= 1.7.0"
                )
                
        except ImportError as e:
            raise ImportError(f"Required dependency missing: {e}")
    
    def to_device(self, array: np.ndarray) -> np.ndarray:
        """
        Convert array to FP64 NumPy array.
        
        Parameters
        ----------
        array : np.ndarray
            Input array (any dtype)
            
        Returns
        -------
        np.ndarray
            Array with dtype=float64
        """
        if array.dtype != np.float64:
            return np.asarray(array, dtype=np.float64)
        return array
    
    def to_numpy(self, device_array: np.ndarray) -> np.ndarray:
        """
        No-op for CPU backend (already NumPy).
        
        Parameters
        ----------
        device_array : np.ndarray
            Already a NumPy array
            
        Returns
        -------
        np.ndarray
            Same array
        """
        return device_array
    
    def cholesky(self, matrix: np.ndarray, upper: bool) -> np.ndarray:
        """
        Compute Cholesky decomposition using SciPy.
        
        Parameters
        ----------
        matrix : np.ndarray
            Positive definite matrix
        upper : bool
            If True, return upper triangular factor
            
        Returns
        -------
        np.ndarray
            Cholesky factor
            
        Raises
        ------
        np.linalg.LinAlgError
            If matrix is not positive definite
        """
        try:
            # SciPy's cholesky is more robust than NumPy's
            factor = scipy.linalg.cholesky(matrix, lower=not upper)
            return factor
        except scipy.linalg.LinAlgError as e:
            # Convert to NumPy exception for consistency
            raise np.linalg.LinAlgError(str(e))
    
    def solve_triangular(self, a: np.ndarray, b: np.ndarray, 
                        upper: bool, trans: str) -> np.ndarray:
        """
        Solve triangular system using SciPy.
        
        Parameters
        ----------
        a : np.ndarray
            Triangular matrix
        b : np.ndarray
            Right-hand side
        upper : bool
            Whether a is upper triangular
        trans : str
            'N' for a @ x = b, 'T' for a.T @ x = b
            
        Returns
        -------
        np.ndarray
            Solution x
        """
        if trans not in ['N', 'T']:
            raise ValueError(f"trans must be 'N' or 'T', got {trans}")
        
        return scipy.linalg.solve_triangular(
            a, b, 
            lower=not upper, 
            trans=trans
        )
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using NumPy.
        
        Parameters
        ----------
        a, b : np.ndarray
            Matrices to multiply
            
        Returns
        -------
        np.ndarray
            Matrix product a @ b
        """
        return np.matmul(a, b)
    
    def inv(self, matrix: np.ndarray) -> np.ndarray:
        """
        Matrix inversion using NumPy.
        
        Parameters
        ----------
        matrix : np.ndarray
            Square matrix to invert
            
        Returns
        -------
        np.ndarray
            Matrix inverse
            
        Raises
        ------
        np.linalg.LinAlgError
            If matrix is singular
            
        Notes
        -----
        Uses numpy.linalg.inv which employs LU decomposition.
        For symmetric positive definite matrices, consider using
        Cholesky decomposition instead for better numerical stability.
        """
        return np.linalg.inv(matrix)
    
    def log_det(self, matrix: np.ndarray) -> float:
        """
        Compute log determinant using Cholesky decomposition.
        
        Parameters
        ----------
        matrix : np.ndarray
            Positive definite matrix
            
        Returns
        -------
        float
            Natural logarithm of determinant
            
        Notes
        -----
        For positive definite matrices, uses the identity:
        log(det(A)) = 2 * sum(log(diag(cholesky(A))))
        
        This is more numerically stable than computing det directly.
        """
        try:
            # Use Cholesky for numerical stability
            L = self.cholesky(matrix, upper=False)
            return 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fallback to slogdet if Cholesky fails
            sign, logdet = np.linalg.slogdet(matrix)
            if sign <= 0:
                raise ValueError("Matrix is not positive definite")
            return logdet
    
    def quadratic_form(self, x: np.ndarray, A: np.ndarray) -> float:
        """
        Compute quadratic form x.T @ A @ x.
        
        Parameters
        ----------
        x : np.ndarray
            Vector of shape (n,) or (n, 1)
        A : np.ndarray
            Positive definite matrix of shape (n, n)
            
        Returns
        -------
        float
            Scalar result of quadratic form
            
        Notes
        -----
        Computed as x.T @ A @ x, which is more efficient than
        creating intermediate matrices.
        """
        # Ensure x is 1D for efficient computation
        x = np.ravel(x)
        
        # Compute x.T @ A @ x efficiently
        # This is faster than x.T @ A @ x for large matrices
        Ax = np.dot(A, x)
        result = np.dot(x, Ax)
        
        return float(result)
    
    def solve_posdef(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve A @ x = b where A is positive definite.
        
        Parameters
        ----------
        A : np.ndarray
            Positive definite matrix
        b : np.ndarray
            Right-hand side
            
        Returns
        -------
        np.ndarray
            Solution x
            
        Notes
        -----
        Uses Cholesky decomposition for efficiency:
        A = L @ L.T, then solve L @ y = b and L.T @ x = y
        """
        try:
            # Cholesky decomposition
            L = self.cholesky(A, upper=False)
            
            # Forward substitution: L @ y = b
            y = self.solve_triangular(L, b, upper=False, trans='N')
            
            # Back substitution: L.T @ x = y
            x = self.solve_triangular(L, y, upper=False, trans='T')
            
            return x
            
        except np.linalg.LinAlgError:
            # Fallback to general solver
            return np.linalg.solve(A, b)
    
    def eigenvalues(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues of symmetric matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Symmetric matrix
            
        Returns
        -------
        np.ndarray
            Eigenvalues in ascending order
        """
        return np.linalg.eigvalsh(matrix)
    
    def is_positive_definite(self, matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if matrix is positive definite.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix to check
        tol : float
            Tolerance for smallest eigenvalue
            
        Returns
        -------
        bool
            True if all eigenvalues > tol
        """
        try:
            # Try Cholesky first (fastest)
            self.cholesky(matrix, upper=False)
            return True
        except np.linalg.LinAlgError:
            # Check eigenvalues as fallback
            eigenvals = self.eigenvalues(matrix)
            return np.all(eigenvals > tol)
    
    def make_positive_definite(self, matrix: np.ndarray, 
                              min_eigenval: float = 1e-6) -> np.ndarray:
        """
        Make matrix positive definite by adjusting eigenvalues.
        
        Parameters
        ----------
        matrix : np.ndarray
            Symmetric matrix
        min_eigenval : float
            Minimum eigenvalue to enforce
            
        Returns
        -------
        np.ndarray
            Positive definite matrix
            
        Notes
        -----
        Performs eigendecomposition and clips eigenvalues to min_eigenval.
        This is used for numerical stability in optimization.
        """
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Clip eigenvalues
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct matrix
        matrix_pd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure symmetry (numerical errors can break it)
        matrix_pd = 0.5 * (matrix_pd + matrix_pd.T)
        
        return matrix_pd
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information.
        
        Returns
        -------
        dict
            Backend details including versions and capabilities
        """
        import numpy as np
        import scipy
        import platform
        
        info = self.get_device_info()
        info.update({
            'backend_name': self.name,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
            'python_version': platform.python_version(),
            'blas_info': self._get_blas_info(),
            'precision': 'fp64',
            'optimization_method': 'BFGS',  # R-compatible
            'supports_autodiff': False
        })
        
        return info
    
    def _get_blas_info(self) -> Dict[str, str]:
        """Get BLAS/LAPACK configuration."""
        try:
            import numpy as np
            config = np.show_config(mode='dicts')
            
            # Extract BLAS/LAPACK info
            blas_info = {}
            if 'blas_opt_info' in config:
                blas_info['blas'] = config['blas_opt_info'].get('libraries', ['unknown'])[0]
            if 'lapack_opt_info' in config:
                blas_info['lapack'] = config['lapack_opt_info'].get('libraries', ['unknown'])[0]
            
            return blas_info
        except:
            return {'blas': 'unknown', 'lapack': 'unknown'}