"""
Parameterization schemes for covariance matrices.

This module provides different parameterizations to ensure positive definiteness
during optimization. The key insight is that different backends and optimization
methods work better with different parameterizations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class CovarianceParameterization(ABC):
    """
    Abstract base class for covariance parameterizations.
    
    All parameterizations must provide pack/unpack operations
    to convert between (μ, Σ) and the parameter vector θ.
    """
    
    def __init__(self, n_vars: int):
        """
        Initialize parameterization.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (dimension of the multivariate normal)
        """
        self.n_vars = n_vars
        self.n_mean_params = n_vars
        self.n_cov_params = (n_vars * (n_vars + 1)) // 2
        self.n_params = self.n_mean_params + self.n_cov_params
    
    @abstractmethod
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack mean and covariance into parameter vector.
        
        Parameters
        ----------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix (must be positive definite)
            
        Returns
        -------
        theta : np.ndarray, shape (n_params,)
            Parameter vector
        """
        raise NotImplementedError
    
    @abstractmethod
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack parameter vector into mean and covariance.
        
        Parameters
        ----------
        theta : np.ndarray, shape (n_params,)
            Parameter vector
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix (guaranteed positive definite)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_initial_parameters(self, sample_mean: np.ndarray, 
                              sample_cov: np.ndarray) -> np.ndarray:
        """
        Get initial parameter values from sample statistics.
        
        Parameters
        ----------
        sample_mean : np.ndarray, shape (n_vars,)
            Sample mean
        sample_cov : np.ndarray, shape (n_vars, n_vars)
            Sample covariance
            
        Returns
        -------
        theta : np.ndarray, shape (n_params,)
            Initial parameter vector
        """
        raise NotImplementedError


class InverseCholeskyParameterization(CovarianceParameterization):
    """
    R-compatible inverse Cholesky parameterization.
    
    This parameterization uses Δ = L⁻¹ where Σ = L'L.
    Parameters: θ = [μ, log(diag(Δ)), off-diag(Δ)]
    
    This is the EXACT parameterization used by R's mvnmle package.
    Used for CPU backend to maintain R compatibility.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack using inverse Cholesky parameterization (R-compatible).
        
        Parameters
        ----------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
            
        Returns
        -------
        theta : np.ndarray
            Parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
        """
        n = self.n_vars
        
        # Compute Cholesky decomposition: Σ = L'L
        L = np.linalg.cholesky(sigma).T  # Upper triangular
        
        # Compute inverse: Δ = L⁻¹
        delta = np.linalg.inv(L)
        
        # Extract parameters
        theta = np.zeros(self.n_params)
        
        # Mean parameters
        theta[:n] = mu
        
        # Log diagonal elements of Δ
        theta[n:2*n] = np.log(np.diag(delta))
        
        # Off-diagonal elements (by column, then row within column)
        idx = 2 * n
        for j in range(n):
            for i in range(j):
                theta[idx] = delta[i, j]
                idx += 1
        
        return theta
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack from inverse Cholesky parameterization.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(Δ)), off-diag(Δ)]
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
        """
        n = self.n_vars
        
        # Extract mean
        mu = theta[:n].copy()
        
        # Reconstruct Δ matrix
        delta = np.zeros((n, n))
        
        # Diagonal elements (exponentiated)
        np.fill_diagonal(delta, np.exp(theta[n:2*n]))
        
        # Off-diagonal elements
        idx = 2 * n
        for j in range(n):
            for i in range(j):
                delta[i, j] = theta[idx]
                idx += 1
        
        # Compute Σ = (Δ⁻¹)'(Δ⁻¹)
        delta_inv = np.linalg.inv(delta)
        sigma = delta_inv.T @ delta_inv
        
        # Ensure symmetry (numerical errors can break it)
        sigma = 0.5 * (sigma + sigma.T)
        
        return mu, sigma
    
    def get_initial_parameters(self, sample_mean: np.ndarray,
                              sample_cov: np.ndarray) -> np.ndarray:
        """
        Get R-compatible initial parameters.
        
        Uses the same initialization as R's mvnmle:
        - μ = sample mean
        - Σ = sample covariance (regularized if needed)
        """
        # Regularize covariance if needed
        eigenvals = np.linalg.eigvalsh(sample_cov)
        if np.min(eigenvals) < 1e-6:
            sample_cov = sample_cov + (1e-6 - np.min(eigenvals) + 1e-8) * np.eye(self.n_vars)
        
        # Pack into parameters
        return self.pack(sample_mean, sample_cov)


class CholeskyParameterization(CovarianceParameterization):
    """
    Standard Cholesky parameterization.
    
    This parameterization uses L where Σ = LL'.
    Parameters: θ = [μ, log(diag(L)), off-diag(L)]
    
    Used for GPU backends as it's more natural for autodiff.
    More numerically stable with FP32 precision.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack using standard Cholesky parameterization.
        
        Parameters
        ----------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
            
        Returns
        -------
        theta : np.ndarray
            Parameter vector [μ, log(diag(L)), off-diag(L)]
        """
        n = self.n_vars
        
        # Compute Cholesky decomposition: Σ = LL'
        L = np.linalg.cholesky(sigma)  # Lower triangular
        
        # Extract parameters
        theta = np.zeros(self.n_params)
        
        # Mean parameters
        theta[:n] = mu
        
        # Log diagonal elements of L
        theta[n:2*n] = np.log(np.diag(L))
        
        # Off-diagonal elements (lower triangular, by column)
        idx = 2 * n
        for j in range(n):
            for i in range(j + 1, n):
                theta[idx] = L[i, j]
                idx += 1
        
        return theta
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack from standard Cholesky parameterization.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(L)), off-diag(L)]
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
        """
        n = self.n_vars
        
        # Extract mean
        mu = theta[:n].copy()
        
        # Reconstruct L matrix (lower triangular)
        L = np.zeros((n, n))
        
        # Diagonal elements (exponentiated)
        np.fill_diagonal(L, np.exp(theta[n:2*n]))
        
        # Off-diagonal elements
        idx = 2 * n
        for j in range(n):
            for i in range(j + 1, n):
                L[i, j] = theta[idx]
                idx += 1
        
        # Compute Σ = LL'
        sigma = L @ L.T
        
        # Ensure symmetry
        sigma = 0.5 * (sigma + sigma.T)
        
        return mu, sigma
    
    def get_initial_parameters(self, sample_mean: np.ndarray,
                              sample_cov: np.ndarray) -> np.ndarray:
        """
        Get initial parameters for Cholesky parameterization.
        
        Similar to R but using standard Cholesky.
        """
        # Regularize covariance if needed
        eigenvals = np.linalg.eigvalsh(sample_cov)
        if np.min(eigenvals) < 1e-6:
            sample_cov = sample_cov + (1e-6 - np.min(eigenvals) + 1e-8) * np.eye(self.n_vars)
        
        # Pack into parameters
        return self.pack(sample_mean, sample_cov)


class MatrixLogParameterization(CovarianceParameterization):
    """
    Matrix logarithm parameterization.
    
    Uses Σ = exp(A) where A is symmetric.
    This guarantees positive definiteness without Cholesky.
    
    Experimental - may be useful for certain optimization landscapes.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack using matrix logarithm.
        
        Parameters
        ----------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
            
        Returns
        -------
        theta : np.ndarray
            Parameter vector [μ, vech(log(Σ))]
        """
        n = self.n_vars
        
        # Compute matrix logarithm
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        log_eigenvals = np.log(eigenvals)
        log_sigma = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T
        
        # Extract parameters
        theta = np.zeros(self.n_params)
        
        # Mean parameters
        theta[:n] = mu
        
        # Vectorize lower triangle of log(Σ)
        idx = n
        for i in range(n):
            for j in range(i + 1):
                theta[idx] = log_sigma[i, j]
                idx += 1
        
        return theta
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack from matrix logarithm parameterization.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, vech(log(Σ))]
            
        Returns
        -------
        mu : np.ndarray, shape (n_vars,)
            Mean vector
        sigma : np.ndarray, shape (n_vars, n_vars)
            Covariance matrix
        """
        n = self.n_vars
        
        # Extract mean
        mu = theta[:n].copy()
        
        # Reconstruct log(Σ)
        log_sigma = np.zeros((n, n))
        idx = n
        for i in range(n):
            for j in range(i + 1):
                log_sigma[i, j] = theta[idx]
                log_sigma[j, i] = theta[idx]  # Symmetric
                idx += 1
        
        # Compute Σ = exp(log_sigma)
        eigenvals, eigenvecs = np.linalg.eigh(log_sigma)
        exp_eigenvals = np.exp(eigenvals)
        sigma = eigenvecs @ np.diag(exp_eigenvals) @ eigenvecs.T
        
        # Ensure symmetry
        sigma = 0.5 * (sigma + sigma.T)
        
        return mu, sigma
    
    def get_initial_parameters(self, sample_mean: np.ndarray,
                              sample_cov: np.ndarray) -> np.ndarray:
        """Get initial parameters for matrix log parameterization."""
        # Regularize if needed
        eigenvals = np.linalg.eigvalsh(sample_cov)
        if np.min(eigenvals) < 1e-6:
            sample_cov = sample_cov + (1e-6 - np.min(eigenvals) + 1e-8) * np.eye(self.n_vars)
        
        return self.pack(sample_mean, sample_cov)


def convert_parameters(theta: np.ndarray,
                      from_param: CovarianceParameterization,
                      to_param: CovarianceParameterization) -> np.ndarray:
    """
    Convert parameters between different parameterizations.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector in source parameterization
    from_param : CovarianceParameterization
        Source parameterization
    to_param : CovarianceParameterization
        Target parameterization
        
    Returns
    -------
    theta_new : np.ndarray
        Parameter vector in target parameterization
    """
    # Unpack using source parameterization
    mu, sigma = from_param.unpack(theta)
    
    # Pack using target parameterization
    return to_param.pack(mu, sigma)


def get_parameterization(name: str, n_vars: int) -> CovarianceParameterization:
    """
    Get parameterization by name.
    
    Parameters
    ----------
    name : str
        Parameterization name: 'inverse_cholesky', 'cholesky', or 'matrix_log'
    n_vars : int
        Number of variables
        
    Returns
    -------
    CovarianceParameterization
        Parameterization instance
    """
    name = name.lower()
    
    if name in ['inverse_cholesky', 'inverse-cholesky', 'r', 'cpu']:
        return InverseCholeskyParameterization(n_vars)
    elif name in ['cholesky', 'standard', 'gpu']:
        return CholeskyParameterization(n_vars)
    elif name in ['matrix_log', 'matrix-log', 'log']:
        return MatrixLogParameterization(n_vars)
    else:
        raise ValueError(
            f"Unknown parameterization: {name}. "
            f"Available: 'inverse_cholesky', 'cholesky', 'matrix_log'"
        )