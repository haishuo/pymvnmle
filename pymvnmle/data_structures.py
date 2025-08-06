"""
Data structures for PyMVNMLE.

This module defines the core data structures used throughout PyMVNMLE,
including the MLResult class that encapsulates estimation results.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class MLResult:
    """
    Result object from maximum likelihood estimation.
    
    This class encapsulates all results from the mlest() function,
    providing a clean interface for accessing estimates and diagnostics.
    
    Attributes
    ----------
    muhat : np.ndarray
        Estimated mean vector of shape (n_variables,)
    sigmahat : np.ndarray
        Estimated covariance matrix of shape (n_variables, n_variables)
    loglik : float
        Log-likelihood at convergence
    n_iter : int
        Number of iterations performed
    converged : bool
        Whether the optimization converged
    computation_time : float
        Total computation time in seconds
    backend : str
        Backend used ('cpu', 'gpu_fp32', 'gpu_fp64')
    method : str
        Optimization method used ('BFGS', 'Newton-CG', etc.)
    patterns : dict
        Missing data pattern information
    n_obs : int
        Number of observations
    n_vars : int
        Number of variables
    n_missing : int
        Total number of missing values
    grad_norm : float
        Final gradient norm
    message : str
        Convergence message or error description
    """
    
    # Required fields
    muhat: np.ndarray
    sigmahat: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    computation_time: float
    backend: str
    method: str
    patterns: Dict[str, Any]
    
    # Data dimensions
    n_obs: int
    n_vars: int
    n_missing: int
    
    # Convergence diagnostics
    grad_norm: float
    message: str = ""
    
    # Optional fields for extended diagnostics
    hessian: Optional[np.ndarray] = None
    standard_errors: Optional[np.ndarray] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        # Ensure arrays are numpy arrays
        self.muhat = np.asarray(self.muhat)
        self.sigmahat = np.asarray(self.sigmahat)
        
        # Validate dimensions
        if self.muhat.ndim != 1:
            raise ValueError(f"muhat must be 1-dimensional, got shape {self.muhat.shape}")
        
        if self.sigmahat.ndim != 2:
            raise ValueError(f"sigmahat must be 2-dimensional, got shape {self.sigmahat.shape}")
        
        if self.sigmahat.shape[0] != self.sigmahat.shape[1]:
            raise ValueError(f"sigmahat must be square, got shape {self.sigmahat.shape}")
        
        if len(self.muhat) != self.sigmahat.shape[0]:
            raise ValueError(
                f"Dimension mismatch: muhat has {len(self.muhat)} elements, "
                f"sigmahat is {self.sigmahat.shape[0]}x{self.sigmahat.shape[1]}"
            )
        
        # Compute AIC and BIC if not provided
        if self.aic is None:
            n_params = len(self.muhat) + len(self.muhat) * (len(self.muhat) + 1) // 2
            self.aic = -2 * self.loglik + 2 * n_params
        
        if self.bic is None:
            n_params = len(self.muhat) + len(self.muhat) * (len(self.muhat) + 1) // 2
            self.bic = -2 * self.loglik + n_params * np.log(self.n_obs)
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            "Maximum Likelihood Estimation Results",
            "=" * 40,
            f"Converged: {self.converged}",
            f"Iterations: {self.n_iter}",
            f"Log-likelihood: {self.loglik:.6f}",
            f"AIC: {self.aic:.6f}",
            f"BIC: {self.bic:.6f}",
            "",
            f"Data: {self.n_obs} observations, {self.n_vars} variables",
            f"Missing values: {self.n_missing} ({100*self.n_missing/(self.n_obs*self.n_vars):.1f}%)",
            f"Missing patterns: {len(self.patterns.get('pattern_indices', []))}",
            "",
            f"Backend: {self.backend}",
            f"Method: {self.method}",
            f"Computation time: {self.computation_time:.3f}s",
            "",
            "Mean estimates:",
            str(self.muhat),
            "",
            "Covariance matrix:",
            str(self.sigmahat)
        ]
        
        if not self.converged:
            lines.append("")
            lines.append(f"Warning: {self.message}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"MLResult(converged={self.converged}, "
            f"n_iter={self.n_iter}, "
            f"loglik={self.loglik:.6f}, "
            f"backend='{self.backend}', "
            f"method='{self.method}')"
        )
    
    @property
    def correlation_matrix(self) -> np.ndarray:
        """
        Compute correlation matrix from covariance matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        D = np.sqrt(np.diag(self.sigmahat))
        return self.sigmahat / np.outer(D, D)
    
    @property
    def standard_deviations(self) -> np.ndarray:
        """
        Extract standard deviations from covariance matrix.
        
        Returns
        -------
        np.ndarray
            Standard deviations for each variable
        """
        return np.sqrt(np.diag(self.sigmahat))
    
    def summary(self) -> None:
        """Print a detailed summary of results."""
        print(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all result fields
        """
        return {
            'muhat': self.muhat.tolist(),
            'sigmahat': self.sigmahat.tolist(),
            'loglik': self.loglik,
            'n_iter': self.n_iter,
            'converged': self.converged,
            'computation_time': self.computation_time,
            'backend': self.backend,
            'method': self.method,
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'n_missing': self.n_missing,
            'grad_norm': self.grad_norm,
            'message': self.message,
            'aic': self.aic,
            'bic': self.bic
        }
    
    def save(self, filename: str) -> None:
        """
        Save results to file.
        
        Parameters
        ----------
        filename : str
            Path to save file (supports .npz, .json)
        """
        import json
        from pathlib import Path
        
        path = Path(filename)
        suffix = path.suffix.lower()
        
        if suffix == '.npz':
            np.savez(
                path,
                muhat=self.muhat,
                sigmahat=self.sigmahat,
                loglik=self.loglik,
                n_iter=self.n_iter,
                converged=self.converged,
                computation_time=self.computation_time,
                backend=self.backend,
                method=self.method,
                n_obs=self.n_obs,
                n_vars=self.n_vars,
                n_missing=self.n_missing,
                grad_norm=self.grad_norm,
                message=self.message
            )
        elif suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @classmethod
    def load(cls, filename: str) -> 'MLResult':
        """
        Load results from file.
        
        Parameters
        ----------
        filename : str
            Path to saved results file
            
        Returns
        -------
        MLResult
            Loaded results object
        """
        import json
        from pathlib import Path
        
        path = Path(filename)
        suffix = path.suffix.lower()
        
        if suffix == '.npz':
            data = np.load(path, allow_pickle=True)
            return cls(
                muhat=data['muhat'],
                sigmahat=data['sigmahat'],
                loglik=float(data['loglik']),
                n_iter=int(data['n_iter']),
                converged=bool(data['converged']),
                computation_time=float(data['computation_time']),
                backend=str(data['backend']),
                method=str(data['method']),
                patterns={},  # Not saved in npz
                n_obs=int(data['n_obs']),
                n_vars=int(data['n_vars']),
                n_missing=int(data['n_missing']),
                grad_norm=float(data['grad_norm']),
                message=str(data['message'])
            )
        elif suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(
                muhat=np.array(data['muhat']),
                sigmahat=np.array(data['sigmahat']),
                loglik=data['loglik'],
                n_iter=data['n_iter'],
                converged=data['converged'],
                computation_time=data['computation_time'],
                backend=data['backend'],
                method=data['method'],
                patterns={},  # Not saved in json
                n_obs=data['n_obs'],
                n_vars=data['n_vars'],
                n_missing=data['n_missing'],
                grad_norm=data['grad_norm'],
                message=data['message']
            )
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


# Additional data structures can be added here as needed
__all__ = ['MLResult']