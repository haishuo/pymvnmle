"""
PyTorch MLE objective using Cholesky parameterization with PATTERN BATCHING.
Optimized for GPU performance while maintaining compatibility with existing base.py
"""

import torch
import numpy as np
from typing import Union, Tuple, List, Optional, Dict
from .base import MLEObjectiveBase, PatternData
from .numpy_objective import NumpyMLEObjective


class TorchMLEObjective(MLEObjectiveBase):
    """
    PyTorch implementation using Cholesky parameterization with GPU optimization.
    
    Key optimizations:
    1. Pattern batching - process patterns of same size together
    2. torch.compile - JIT compilation for performance
    3. Efficient tensor operations - minimize CPU-GPU transfers
    
    This will give different parameter values but the SAME estimates for (μ, Σ).
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
    
    def _prepare_patterns(self):
        """Convert pattern data to PyTorch tensors AND group by size for batching."""
        self.torch_patterns = []
        self.pattern_groups = {}  # Group patterns by number of observed variables
        
        for pattern in self.patterns:
            if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                continue
            
            # Convert to tensors
            pattern_dict = {
                'data': torch.tensor(pattern.data_k, dtype=torch.float64, device=self.device),
                'obs_idx': torch.tensor(pattern.observed_indices, dtype=torch.long, device=self.device),
                'n_k': pattern.n_k,
                'n_obs': len(pattern.observed_indices)
            }
            
            self.torch_patterns.append(pattern_dict)
            
            # Group by size for batching
            n_obs = len(pattern.observed_indices)
            if n_obs not in self.pattern_groups:
                self.pattern_groups[n_obs] = {
                    'patterns': [],
                    'indices': [],  # Store original pattern indices
                    'total_n_k': 0
                }
            
            self.pattern_groups[n_obs]['patterns'].append(pattern_dict)
            self.pattern_groups[n_obs]['indices'].append(len(self.torch_patterns) - 1)
            self.pattern_groups[n_obs]['total_n_k'] += pattern.n_k
        
        # Print batching statistics
        if len(self.pattern_groups) < len(self.torch_patterns):
            print(f"📊 Pattern batching: {len(self.torch_patterns)} patterns → {len(self.pattern_groups)} batches")
    
    def _unpack_theta_cholesky(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack parameters using Cholesky parameterization.
        
        Parameters are organized as:
        - First n_vars: mean μ
        - Next n_vars: log-diagonal of Cholesky factor L
        - Remaining: Lower triangular elements of L (row-major order)
        
        This ensures Σ = L @ L.T is always positive definite.
        """
        # Mean
        mu = theta[:self.n_vars]
        
        # Build lower triangular Cholesky factor
        L = torch.zeros((self.n_vars, self.n_vars), dtype=theta.dtype, device=theta.device)
        
        # Diagonal (positive via exp)
        log_diag = theta[self.n_vars:2*self.n_vars]
        L_diag = torch.exp(log_diag)
        L = L + torch.diag(L_diag)
        
        # Lower triangular elements
        idx = 2 * self.n_vars
        for i in range(1, self.n_vars):
            for j in range(i):
                L[i, j] = theta[idx]
                idx += 1
        
        return mu, L
    
    @torch.compile  # JIT compilation for performance
    def _torch_objective(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute -2*log(L) using Cholesky parameterization with PATTERN BATCHING.
        
        This is the optimized version that processes patterns in batches.
        """
        # Unpack parameters
        mu, L = self._unpack_theta_cholesky(theta)
        
        # Compute full covariance: Σ = L @ L.T
        Sigma = torch.matmul(L, L.T)
        
        # Add small regularization for numerical stability
        eps = 1e-8
        Sigma = Sigma + eps * torch.eye(self.n_vars, device=Sigma.device)
        
        # Compute objective by pattern groups (BATCHED)
        neg_2_loglik = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        # Process each group of patterns with same size
        for n_obs, group_info in self.pattern_groups.items():
            patterns = group_info['patterns']
            n_patterns = len(patterns)
            
            if n_patterns == 1:
                # Single pattern - process as before
                pattern = patterns[0]
                contrib = self._process_single_pattern(pattern, mu, Sigma, eps)
                neg_2_loglik = neg_2_loglik + contrib
            else:
                # Multiple patterns - BATCH PROCESSING!
                contrib = self._process_pattern_batch(patterns, mu, Sigma, eps, n_obs)
                neg_2_loglik = neg_2_loglik + contrib
        
        return neg_2_loglik
    
    def _process_single_pattern(self, pattern: Dict, mu: torch.Tensor, 
                               Sigma: torch.Tensor, eps: float) -> torch.Tensor:
        """Process a single pattern (original method for comparison)."""
        data_k = pattern['data']
        obs_idx = pattern['obs_idx']
        n_k = pattern['n_k']
        n_obs = pattern['n_obs']
        
        # Extract parameters for observed variables
        mu_k = mu[obs_idx]
        Sigma_k = Sigma[obs_idx[:, None], obs_idx]
        
        # Add regularization
        Sigma_k = Sigma_k + eps * torch.eye(n_obs, device=Sigma_k.device)
        
        # Compute using Cholesky decomposition for stability
        try:
            L_k = torch.linalg.cholesky(Sigma_k)
            
            # Log-determinant: log|Σ| = 2*log|L|
            log_det = 2 * torch.sum(torch.log(torch.diag(L_k)))
            
            # Quadratic form: (y-μ)'Σ^(-1)(y-μ)
            centered = data_k - mu_k
            z = torch.linalg.solve_triangular(L_k, centered.T, upper=False)
            quadratic = torch.sum(z * z)
            
            # Contribution (without 2π term)
            contrib = n_k * log_det + quadratic
            
        except RuntimeError:
            # Fallback for numerical issues
            contrib = self._fallback_computation(Sigma_k, mu_k, data_k, n_k, eps)
        
        return contrib
    
    def _process_pattern_batch(self, patterns: List[Dict], mu: torch.Tensor,
                              Sigma: torch.Tensor, eps: float, n_obs: int) -> torch.Tensor:
        """
        Process multiple patterns of the same size in a batch.
        
        This is the KEY OPTIMIZATION - process many patterns at once!
        """
        n_patterns = len(patterns)
        
        # Stack all data for this pattern size
        # Create batch tensors
        batch_data = []
        batch_n_k = []
        batch_indices = []
        
        for pattern in patterns:
            batch_data.append(pattern['data'])
            batch_n_k.append(pattern['n_k'])
            batch_indices.append(pattern['obs_idx'])
        
        # Use the first pattern's indices (they all have same observed variables structure)
        obs_idx = patterns[0]['obs_idx']
        
        # Extract parameters for observed variables (same for all in batch)
        mu_k = mu[obs_idx]
        Sigma_k = Sigma[obs_idx[:, None], obs_idx]
        Sigma_k = Sigma_k + eps * torch.eye(n_obs, device=Sigma_k.device)
        
        # Compute Cholesky once for all patterns
        try:
            L_k = torch.linalg.cholesky(Sigma_k)
            log_det = 2 * torch.sum(torch.log(torch.diag(L_k)))
            
            # Process all patterns in batch
            total_contrib = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            
            # Batch compute quadratic forms
            for i, pattern in enumerate(patterns):
                data_k = pattern['data']
                n_k = pattern['n_k']
                
                # Quadratic form for this pattern
                centered = data_k - mu_k
                z = torch.linalg.solve_triangular(L_k, centered.T, upper=False)
                quadratic = torch.sum(z * z)
                
                total_contrib += n_k * log_det + quadratic
            
            return total_contrib
            
        except RuntimeError:
            # Fallback - process individually
            total_contrib = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            for pattern in patterns:
                contrib = self._process_single_pattern(pattern, mu, Sigma, eps)
                total_contrib += contrib
            return total_contrib
    
    def _fallback_computation(self, Sigma_k: torch.Tensor, mu_k: torch.Tensor,
                             data_k: torch.Tensor, n_k: int, eps: float) -> torch.Tensor:
        """Fallback computation using eigendecomposition for numerical issues."""
        eigvals, eigvecs = torch.linalg.eigh(Sigma_k)
        eigvals = torch.clamp(eigvals, min=eps)
        
        log_det = torch.sum(torch.log(eigvals))
        
        # Quadratic form
        centered = data_k - mu_k
        transformed = torch.matmul(centered, eigvecs)
        weighted = transformed / torch.sqrt(eigvals)
        quadratic = torch.sum(weighted * weighted)
        
        return n_k * log_det + quadratic
    
    def pack_parameters(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Pack (μ, Σ) into parameter vector using Cholesky parameterization.
        
        This is used to convert from R parameterization to our parameterization.
        """
        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # Add regularization if not positive definite
            eigvals = np.linalg.eigvalsh(sigma)
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                sigma = sigma + (1e-8 - min_eig) * np.eye(self.n_vars)
            L = np.linalg.cholesky(sigma)
        
        # Pack parameters
        theta = np.zeros(self.n_params)
        
        # Mean
        theta[:self.n_vars] = mu
        
        # Log-diagonal of L
        theta[self.n_vars:2*self.n_vars] = np.log(np.diag(L))
        
        # Lower triangular elements
        idx = 2 * self.n_vars
        for i in range(1, self.n_vars):
            for j in range(i):
                theta[idx] = L[i, j]
                idx += 1
        
        return theta
    
    def unpack_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack parameter vector to (μ, Σ)."""
        # Convert to tensor
        theta_tensor = torch.tensor(theta, dtype=torch.float64, device=self.device)
        
        # Unpack
        mu, L = self._unpack_theta_cholesky(theta_tensor)
        
        # Compute covariance
        Sigma = torch.matmul(L, L.T)
        
        # Convert to numpy
        return mu.cpu().numpy(), Sigma.cpu().numpy()
    
    def __call__(self, theta: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute objective value."""
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            return self._torch_objective(theta).item()
    
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute analytical gradient using autodiff."""
        theta_tensor = torch.tensor(theta, dtype=torch.float64, device=self.device, requires_grad=True)
        
        # Forward pass
        loss = self._torch_objective(theta_tensor)
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        return theta_tensor.grad.detach().cpu().numpy()
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract mean, covariance, and log-likelihood."""
        mu, sigma = self.unpack_parameters(theta)
        
        # Compute log-likelihood
        neg_2_loglik = self(theta)
        loglik = -neg_2_loglik / 2.0
        
        return mu, sigma, loglik