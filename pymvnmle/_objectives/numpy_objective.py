"""
NumPy/CPU implementation of MLE objective.
This is the EXACT same implementation as the original _objective.py,
just cleaned of PyTorch code and inheriting shared preprocessing.

CRITICAL: ALL mathematical algorithms are preserved exactly.
"""

import numpy as np
from typing import Union, Tuple, List
from .base import MLEObjectiveBase, PatternData  # Import base and PatternData


class NumpyMLEObjective(MLEObjectiveBase):  # Changed name and inheritance
    """
    CPU implementation of MLE objective with finite differences.
    
    This is the reference implementation matching R's mvnmle exactly.
    All mathematical algorithms are preserved:
    1. Row shuffling for submatrix extraction
    2. Givens rotations for numerical stability
    3. Pattern-wise likelihood computation
    4. R's exact parameter ordering
    """
    
    def __init__(self, data: np.ndarray, backend=None):  # Keep backend param for compatibility
        """
        Initialize with R's exact preprocessing algorithm.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Data matrix with missing values as np.nan
        backend : ignored
            Kept for backward compatibility
        """
        super().__init__(data)  # Use base class preprocessing
        
        # Any additional initialization specific to NumPy implementation
        # (Currently none needed)
    
    def _reconstruct_delta_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Reconstruct upper triangular Δ matrix from parameter vector.
        
        EXACT copy from original implementation.
        """
        Delta = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements: exp(log-parameters) to ensure positivity
        log_diag = theta[self.n_vars:2*self.n_vars]
        np.fill_diagonal(Delta, np.exp(log_diag))
        
        # Off-diagonal elements in R's column-major order
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):
                Delta[i, j] = theta[idx]
                idx += 1
        
        return Delta
    
    def _apply_givens_rotations(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply Givens rotations for numerical stability.
        
        EXACT copy from original - this is CRITICAL for R compatibility.
        Matches R's evallf.c implementation precisely.
        """
        result = matrix.copy()
        
        # R's exact algorithm: bottom-up, left-to-right
        for i in range(self.n_vars - 1, -1, -1):  # Bottom to top
            for j in range(i):  # Left to diagonal
                a = result[i, j]
                b = result[i, j+1] if j+1 < self.n_vars else 0.0
                
                # R's exact threshold
                if np.abs(a) < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                # Compute rotation parameters
                r = np.sqrt(a*a + b*b)
                if r < 0.000001:
                    continue
                
                c = a / r
                d = b / r
                
                # Apply rotation to entire matrix
                for k in range(self.n_vars):
                    old_kj = result[k, j]
                    old_kj1 = result[k, j+1] if j+1 < self.n_vars else 0.0
                    
                    result[k, j] = d * old_kj - c * old_kj1
                    if j+1 < self.n_vars:
                        result[k, j+1] = c * old_kj + d * old_kj1
                
                result[i, j] = 0.0
        
        # Ensure positive diagonal (R's sign adjustment)
        for i in range(self.n_vars):
            if result[i, i] < 0:
                for j in range(i+1):
                    result[j, i] *= -1
        
        return result
    
    def __call__(self, theta: np.ndarray) -> float:
        """
        Compute negative log-likelihood using R's exact algorithm.
        
        PRESERVES all mathematical operations from original.
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)
        
        # Apply Givens rotations for numerical stability (R's evallf.c)
        Delta_stabilized = self._apply_givens_rotations(Delta)
        
        # Compute negative log-likelihood using pattern-wise formula
        neg_loglik = 0.0
        
        for pattern in self.patterns:
            if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                continue
            
            # Extract relevant submatrices (R's approach)
            obs_indices = pattern.observed_indices
            n_obs_vars = len(obs_indices)
            mu_k = mu[obs_indices]
            
            # CRITICAL: Implement R's row shuffling algorithm exactly
            # Create reordered Delta with observed rows first, missing rows last
            subdel = np.zeros((self.n_vars, self.n_vars))
            
            # Put observed variable rows FIRST
            pcount = 0
            for i in range(self.n_vars):
                if i in obs_indices:
                    subdel[pcount, :] = Delta_stabilized[i, :]
                    pcount += 1
            
            # Put missing variable rows LAST
            acount = 0
            for i in range(self.n_vars):
                if i not in obs_indices:
                    subdel[self.n_vars - acount - 1, :] = Delta_stabilized[i, :]
                    acount += 1
            
            # Apply Givens rotations to shuffled matrix
            subdel_rotated = self._apply_givens_rotations(subdel)
            
            # Extract top-left submatrix for observed variables
            Delta_k = subdel_rotated[:n_obs_vars, :n_obs_vars]
            
            try:
                # Use R's exact computation approach
                # Log-determinant of Delta_k (more stable than computing Sigma_k first)
                log_det_delta_k = np.sum(np.log(np.diag(Delta_k)))
                
                # Compute quadratic form efficiently
                # For each observation, compute (y - μ)'Σ^{-1}(y - μ)
                # Using Σ = (Δ^{-1})'Δ^{-1}, we get (Δ'(y - μ))'(Δ'(y - μ))
                obj_contribution = -2 * pattern.n_k * log_det_delta_k
                
                for i in range(pattern.n_k):
                    centered = pattern.data_k[i] - mu_k
                    # prod = Δ_k' @ centered
                    prod = Delta_k.T @ centered
                    obj_contribution += np.dot(prod, prod)
                
                neg_loglik += obj_contribution
                
            except (np.linalg.LinAlgError, RuntimeError):
                # Handle numerical issues
                return 1e20
        
        return neg_loglik
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract estimates from parameter vector.
        
        EXACT copy from original implementation.
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)
        
        # Convert to covariance matrix: Σ = (Δ⁻¹)ᵀ Δ⁻¹
        try:
            # Use triangular solve for numerical stability (R's approach)
            I = np.eye(self.n_vars)
            Delta_inv = np.linalg.solve(Delta, I)  # More stable than inv(Delta)
            sigma = Delta_inv.T @ Delta_inv
            
            # Ensure exact symmetry (R does this)
            sigma = (sigma + sigma.T) / 2.0
            
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            sigma = np.eye(self.n_vars)
        
        # Compute log-likelihood
        loglik = -self(theta) / 2.0  # Objective is negative log-likelihood
        
        return mu, sigma, loglik