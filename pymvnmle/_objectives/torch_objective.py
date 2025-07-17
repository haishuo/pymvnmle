"""
PyTorch MLE objective with COMPLETE FIX for both parameterization and log-likelihood.

Key fixes:
1. Proper conversion from R's inverse Cholesky to our Cholesky parameterization
2. Correct understanding that objective returns -2 log L (not -log L)
3. Proper log-likelihood computation in extract_parameters
"""

import torch
import numpy as np
from typing import Union, Tuple, List, Optional, Dict
from .base import MLEObjectiveBase, PatternData


class TorchMLEObjective(MLEObjectiveBase):
    """
    PyTorch implementation using Cholesky parameterization with GPU optimization.
    
    CRITICAL FIXES:
    1. Properly converts between R's inverse Cholesky and our Cholesky
    2. Returns -2 log L from objective (matching NumPy convention)
    3. Correctly computes log-likelihood in extract_parameters
    """
    
    def __init__(self, data: np.ndarray, device: Optional[str] = None):
        """Initialize with preprocessing and pattern grouping."""
        super().__init__(data)
        
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print(f"🚀 Using Apple Metal GPU")
            else:
                self.device = torch.device('cpu')
                print(f"💻 Using CPU")
        else:
            self.device = torch.device(device)
        
        # Pre-convert and group patterns for batching
        self._prepare_patterns()
        
        # Compile the objective for performance
        try:
            self._torch_objective_compiled = torch.compile(self._torch_objective)
        except:
            self._torch_objective_compiled = self._torch_objective
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Convert R's initial parameters to our Cholesky format.
        
        The base class returns parameters in R's inverse Cholesky format.
        We need to convert to standard Cholesky.
        """
        # Get R-format initial parameters from base class
        theta_r = super().get_initial_parameters()
        
        # Extract μ and Σ using R's parameterization
        mu, sigma = self._extract_from_r_parameterization(theta_r)
        
        # Pack into our Cholesky parameterization
        theta_cholesky = self.pack_parameters(mu, sigma)
        
        return theta_cholesky
    
    def _extract_from_r_parameterization(self, theta_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (μ, Σ) from R's inverse Cholesky parameterization.
        
        This is the EXACT algorithm used in NumpyMLEObjective.
        """
        # Extract mean
        mu = theta_r[:self.n_vars]
        
        # Reconstruct Δ matrix from R parameterization
        Delta = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements (exponential of log values)
        log_diag = theta_r[self.n_vars:2*self.n_vars]
        Delta[np.diag_indices(self.n_vars)] = np.exp(log_diag)
        
        # Off-diagonal elements (upper triangle, column by column)
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):
                Delta[i, j] = theta_r[idx]
                idx += 1
        
        # Convert to covariance: Σ = (Δ⁻¹)ᵀ(Δ⁻¹)
        try:
            Delta_inv = np.linalg.inv(Delta)
            sigma = Delta_inv.T @ Delta_inv
            # Ensure symmetry
            sigma = (sigma + sigma.T) / 2.0
        except np.linalg.LinAlgError:
            sigma = np.eye(self.n_vars)
        
        return mu, sigma
    
    def pack_parameters(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack (μ, Σ) into our Cholesky parameterization.
        
        Parameters: [μ₁, ..., μₚ, log(L₁₁), ..., log(Lₚₚ), L₂₁, L₃₁, L₃₂, ...]
        where Σ = LLᵀ
        """
        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # Add regularization if not positive definite
            eigvals = np.linalg.eigvalsh(sigma)
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                sigma = sigma + (1e-8 - min_eig + 1e-6) * np.eye(self.n_vars)
            L = np.linalg.cholesky(sigma)
        
        # Pack parameters
        theta = np.zeros(self.n_params)
        
        # Mean
        theta[:self.n_vars] = mu
        
        # Log-diagonal of L
        theta[self.n_vars:2*self.n_vars] = np.log(np.diag(L))
        
        # Lower triangular elements (row by row, excluding diagonal)
        idx = 2 * self.n_vars
        for i in range(1, self.n_vars):
            for j in range(i):
                theta[idx] = L[i, j]
                idx += 1
        
        return theta
    
    def unpack_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack parameter vector to (μ, Σ) from our Cholesky format."""
        # Convert to tensor
        theta_tensor = torch.tensor(theta, dtype=torch.float64, device=self.device)
        
        # Unpack using Cholesky parameterization
        mu, L = self._unpack_theta_cholesky(theta_tensor)
        
        # Compute covariance: Σ = LLᵀ
        Sigma = torch.matmul(L, L.T)
        
        # Convert to numpy
        return mu.cpu().numpy(), Sigma.cpu().numpy()
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mean, covariance, and log-likelihood.
        
        CRITICAL FIX: The objective returns -2 log L, not -log L.
        """
        # Unpack parameters
        mu, sigma = self.unpack_parameters(theta)
        
        # Compute log-likelihood
        # The objective returns -2 log L (matching NumPy)
        neg_2_loglik = self(theta)
        loglik = -neg_2_loglik / 2.0
        
        return mu, sigma, loglik
    
    def _unpack_theta_cholesky(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack parameter vector using Cholesky parameterization.
        """
        # Extract mean
        mu = theta[:self.n_vars]
        
        # Initialize L matrix
        L = torch.zeros((self.n_vars, self.n_vars), dtype=theta.dtype, device=theta.device)
        
        # Set diagonal (exponential to ensure positive)
        log_diag = theta[self.n_vars:2*self.n_vars]
        L.diagonal().copy_(torch.exp(log_diag))
        
        # Fill lower triangle
        idx = 2 * self.n_vars
        for i in range(1, self.n_vars):
            for j in range(i):
                L[i, j] = theta[idx]
                idx += 1
        
        return mu, L
    
    def _torch_objective(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute NEGATIVE 2 × LOG-LIKELIHOOD.
        
        Returns: -2 log L(θ|Y)
        
        This matches the NumPy implementation's convention.
        NO constant term is included (following R's implementation).
        """
        # Unpack parameters
        mu, L = self._unpack_theta_cholesky(theta)
        
        # Initialize objective value
        neg_2_loglik = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        # Process each pattern
        for pattern in self.patterns:
            if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                continue
            
            # Convert pattern data to tensors
            obs_idx = torch.tensor(pattern.observed_indices, device=self.device)
            data_k = torch.tensor(pattern.data_k, dtype=torch.float64, device=self.device)
            
            # Extract parameters for observed variables
            mu_k = mu[obs_idx]
            
            # Extract L submatrix for observed variables
            L_k = L[obs_idx][:, obs_idx]
            
            # Compute log-determinant of Σ_k = L_k L_k^T
            # log|Σ_k| = log|L_k L_k^T| = 2 log|L_k| = 2 Σ log(L_kii)
            log_det_L_k = torch.sum(torch.log(L_k.diagonal()))
            log_det_Sigma_k = 2 * log_det_L_k
            
            # Center the data
            centered = data_k - mu_k  # (n_k, p_obs)
            
            # Solve L_k z = centered^T for z
            # Then ||z||^2 = (y-μ)' Σ^{-1} (y-μ)
            z = torch.linalg.solve_triangular(L_k, centered.T, upper=False)
            quadratic = torch.sum(z * z)
            
            # Pattern contribution to -2 log L
            # -2 log L_k = n_k log|Σ_k| + Σᵢ (yᵢ-μ)'Σ⁻¹(yᵢ-μ)
            pattern_contrib = pattern.n_k * log_det_Sigma_k + quadratic
            
            neg_2_loglik = neg_2_loglik + pattern_contrib
        
        return neg_2_loglik
    
    def __call__(self, theta: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute objective value (-2 log L).
        
        Returns: -2 log L(θ|Y)
        """
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            # Use compiled version if available
            try:
                return self._torch_objective_compiled(theta).item()
            except:
                return self._torch_objective(theta).item()
    
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute analytical gradient of -2 log L using autodiff.
        
        Note: This is the gradient of the objective (-2 log L), not log L.
        """
        theta_tensor = torch.tensor(theta, dtype=torch.float64, device=self.device, requires_grad=True)
        
        # Forward pass
        loss = self._torch_objective(theta_tensor)
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        return theta_tensor.grad.detach().cpu().numpy()
    
    def _prepare_patterns(self):
        """Convert pattern data to PyTorch tensors for efficient computation."""
        # For now, keep patterns as is - batching can be added later
        # This ensures the base implementation works correctly first
        pass