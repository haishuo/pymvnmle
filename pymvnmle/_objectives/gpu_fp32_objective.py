"""
GPU FP32 objective function using standard Cholesky parameterization.

Optimized for consumer GPUs (RTX series, Apple Metal) that have limited
or no FP64 support. Uses PyTorch autodiff for analytical gradients.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any

from pymvnmle._objectives.base import MLEObjectiveBase
from pymvnmle._objectives.parameterizations import (
    CholeskyParameterization,
    BoundedCholeskyParameterization
)


class GPUObjectiveFP32(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective using FP32 precision.
    
    Designed for consumer GPUs where FP64 is severely limited.
    Uses standard Cholesky parameterization for autodiff compatibility.
    """
    
    def __init__(self, data: np.ndarray,
                 device: Optional[str] = None,
                 compile_objective: bool = False,
                 use_bounded: bool = False):
        """
        Initialize GPU FP32 objective.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with missing values as np.nan
        device : str or None
            Device to use ('cuda', 'mps', or None for auto)
        compile_objective : bool
            Whether to compile objective with torch.compile (requires PyTorch 2.0+)
        use_bounded : bool
            Whether to use bounded parameterization for stability
        """
        super().__init__(data, skip_validation=False)
        
        # Import PyTorch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU objectives. "
                "Install with: pip install torch"
            )
        
        # Create parameterization
        if use_bounded:
            self.parameterization = BoundedCholeskyParameterization(
                self.n_vars,
                var_min=0.01,
                var_max=100.0,
                corr_max=0.95
            )
        else:
            self.parameterization = CholeskyParameterization(self.n_vars)
        
        self.n_params = self.parameterization.n_params
        
        # Select device
        self.device = self._select_device(device)
        
        # FP32 settings
        self.dtype = torch.float32
        self.eps = 1e-6  # Looser epsilon for FP32
        
        # Transfer pattern data to GPU
        self._prepare_gpu_data()
        
        # Compile objective if requested and available
        self.use_compiled = False
        if compile_objective:
            if hasattr(torch, 'compile'):
                try:
                    self._compiled_objective = torch.compile(self._torch_objective)
                    self.use_compiled = True
                except Exception as e:
                    warnings.warn(f"Failed to compile objective: {e}")
            else:
                warnings.warn("torch.compile requires PyTorch 2.0+")
    
    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device."""
        torch = self.torch
        
        if requested_device:
            return torch.device(requested_device)
        
        # Auto-select
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            warnings.warn("No GPU available, using CPU (will be slow)")
            return torch.device('cpu')
    
    def _prepare_gpu_data(self) -> None:
        """Transfer pattern data to GPU."""
        torch = self.torch
        
        self.gpu_patterns = []
        for pattern in self.patterns:
            gpu_pattern = {
                'n_obs': pattern.n_obs,
                'n_observed': len(pattern.observed_indices),
                'observed_indices': torch.tensor(
                    pattern.observed_indices,
                    device=self.device,
                    dtype=torch.long
                ),
                'data': torch.tensor(
                    pattern.data,
                    device=self.device,
                    dtype=self.dtype
                )
            }
            self.gpu_patterns.append(gpu_pattern)
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameters with same regularization strategy as CPU.
        
        This ensures fair comparison between CPU and GPU implementations.
        """
        sample_cov_regularized = self.sample_cov.copy()
        
        # Apply same regularization as CPU for consistency
        try:
            eigenvals = np.linalg.eigvalsh(sample_cov_regularized)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)
            
            # If poorly conditioned or non-PD, regularize
            if min_eig < 1e-6 or max_eig / min_eig > 1e10:
                reg_amount = max(1e-4, abs(min_eig) + 1e-4)
                sample_cov_regularized += reg_amount * np.eye(self.n_vars)
                
                # Shrink off-diagonals for stability
                for i in range(self.n_vars):
                    for j in range(i + 1, self.n_vars):
                        sample_cov_regularized[i, j] *= 0.95
                        sample_cov_regularized[j, i] *= 0.95
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal
            sample_cov_regularized = np.diag(np.diag(self.sample_cov))
            sample_cov_regularized += 0.1 * np.eye(self.n_vars)
        
        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            sample_cov_regularized
        )
    
    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute -2 * log-likelihood using GPU.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [μ, log(diag(L)), off-diag(L)]
            
        Returns
        -------
        float
            -2 * log-likelihood value
        """
        torch = self.torch
        
        # Convert to GPU tensor (no gradient needed for objective)
        with torch.no_grad():
            theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
            
            if self.use_compiled:
                obj_value = self._compiled_objective(theta_gpu)
            else:
                obj_value = self._torch_objective(theta_gpu)
        
        return obj_value.item()
    
    def _torch_objective(self, theta_gpu: Any) -> Any:
        """
        PyTorch implementation of objective function.
        
        Parameters
        ----------
        theta_gpu : torch.Tensor
            Parameter vector on GPU
            
        Returns
        -------
        torch.Tensor
            Scalar objective value (-2 * log-likelihood for R compatibility)
        """
        torch = self.torch
        
        # Unpack parameters on GPU
        mu_gpu, sigma_gpu = self._unpack_gpu(theta_gpu)
        
        # Initialize objective
        obj_value = torch.zeros(1, device=self.device, dtype=self.dtype)
        
        # Process each pattern
        for gpu_pattern in self.gpu_patterns:
            if gpu_pattern['n_observed'] == 0:
                continue
            
            # Extract observed submatrices - use index_select for gradient preservation
            obs_idx = gpu_pattern['observed_indices']
            mu_k = torch.index_select(mu_gpu, 0, obs_idx)
            sigma_k_rows = torch.index_select(sigma_gpu, 0, obs_idx)
            sigma_k = torch.index_select(sigma_k_rows, 1, obs_idx)
            
            # Add small diagonal for FP32 stability
            sigma_k = sigma_k + self.eps * torch.eye(
                gpu_pattern['n_observed'],
                device=self.device,
                dtype=self.dtype
            )
            
            # Compute pattern contribution
            contrib = self._compute_pattern_contribution_gpu(
                gpu_pattern, mu_k, sigma_k
            )
            
            # Add contribution directly
            obj_value = obj_value + contrib
        
        return obj_value.squeeze()
    
    def _compute_pattern_contribution_gpu(self, pattern: Dict, 
                                         mu_k: Any, 
                                         sigma_k: Any) -> Any:
        """
        Compute pattern contribution on GPU.
        
        CRITICAL: The CPU objective does NOT include the constant term n*p*log(2π)!
        It only computes: n_k * [log|Σ_k| + tr(Σ_k^-1 * S_k)]
        
        Parameters
        ----------
        pattern : dict
            GPU pattern data
        mu_k : torch.Tensor
            Mean for observed variables
        sigma_k : torch.Tensor
            Covariance for observed variables
            
        Returns
        -------
        torch.Tensor
            Pattern contribution to -2*log-likelihood (WITHOUT constant term)
        """
        torch = self.torch
        
        n_obs = pattern['n_obs']
        data = pattern['data']  # shape: (n_obs, n_observed)
        
        # Log determinant term: log|Σ_k|
        L_k = torch.linalg.cholesky(sigma_k)
        log_det_term = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        
        # Compute sample covariance to match CPU formulation
        data_centered = data - mu_k.unsqueeze(0)
        S_k = (data_centered.T @ data_centered) / n_obs
        
        # Trace term: tr(Σ_k^-1 * S_k)
        sigma_inv_S = torch.linalg.solve(sigma_k, S_k)
        trace_term = torch.trace(sigma_inv_S)
        
        # Total contribution: n_obs * [log|Σ| + tr(Σ^-1 S)]
        # NO CONSTANT TERM to match CPU!
        total_contribution = n_obs * (log_det_term + trace_term)
        
        return total_contribution
    
    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """
        Unpack parameters on GPU maintaining gradient flow.
        
        Parameters
        ----------
        theta_gpu : torch.Tensor
            Parameter vector on GPU
            
        Returns
        -------
        mu : torch.Tensor
            Mean vector on GPU
        sigma : torch.Tensor
            Covariance matrix on GPU
        """
        torch = self.torch
        n = self.n_vars
        
        # Extract mean directly
        mu = theta_gpu[:n]
        
        # Build L matrix based on parameterization type
        if isinstance(self.parameterization, BoundedCholeskyParameterization):
            # Apply bounded transformations on GPU
            L = torch.zeros((n, n), device=self.device, dtype=self.dtype)
            
            # Diagonal: sigmoid transformation
            diag_unbounded = theta_gpu[n:2*n]
            var_min = self.parameterization.var_min
            var_max = self.parameterization.var_max
            diag_vars = var_min + (var_max - var_min) * torch.sigmoid(diag_unbounded)
            L.diagonal().copy_(torch.sqrt(diag_vars))
            
            # Off-diagonal: tanh transformation  
            idx = 2 * n
            tril_indices = torch.tril_indices(n, n, offset=-1, device=self.device)
            if len(tril_indices[0]) > 0:
                tril_unbounded = theta_gpu[idx:]
                corr_max = self.parameterization.corr_max
                corr_vals = corr_max * torch.tanh(tril_unbounded)
                
                # Scale by diagonal (now properly set)
                i, j = tril_indices
                L[tril_indices[0], tril_indices[1]] = corr_vals * torch.sqrt(L[i, i] * L[j, j])
        else:
            # Standard Cholesky
            L = torch.zeros((n, n), device=self.device, dtype=self.dtype)
            
            # Diagonal elements
            L.diagonal().copy_(torch.exp(theta_gpu[n:2*n]))
            
            # Off-diagonal elements
            idx = 2 * n
            tril_indices = torch.tril_indices(n, n, offset=-1, device=self.device)
            if len(tril_indices[0]) > 0:
                L[tril_indices[0], tril_indices[1]] = theta_gpu[idx:]
        
        # Compute Σ = LL'
        sigma = torch.matmul(L, L.T)
        
        # Ensure symmetry
        sigma = 0.5 * (sigma + sigma.T)
        
        return mu, sigma
    
    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient using automatic differentiation.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Gradient vector
        """
        torch = self.torch
        
        # Convert to GPU tensor with gradient tracking
        theta_gpu = torch.tensor(
            theta, 
            device=self.device, 
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Compute objective WITH gradient tracking (no torch.no_grad!)
        obj_value = self._torch_objective(theta_gpu)
        
        # Compute gradient via autodiff
        grad_tensor = torch.autograd.grad(
            outputs=obj_value,
            inputs=theta_gpu,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Return as NumPy array
        return grad_tensor.cpu().numpy()
    
    def compute_hessian(self, theta: np.ndarray) -> np.ndarray:
        """
        Hessian not implemented for FP32 (use BFGS instead).
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Raises NotImplementedError
        """
        raise NotImplementedError(
            "Hessian computation not implemented for FP32. "
            "Use BFGS or L-BFGS-B optimization which only requires gradients."
        )
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mean and covariance from parameter vector.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        mu : np.ndarray
            Mean estimate
        sigma : np.ndarray
            Covariance estimate
        loglik : float
            Log-likelihood value
        """
        # Unpack using parameterization
        mu, sigma = self.parameterization.unpack(theta)
        
        # Compute objective
        neg2_loglik = self.compute_objective(theta)
        loglik = -0.5 * neg2_loglik
        
        return mu, sigma, loglik
    
    def get_optimization_bounds(self):
        """Get bounds if using bounded parameterization."""
        if isinstance(self.parameterization, BoundedCholeskyParameterization):
            # All parameters unbounded in transformed space
            return None
        else:
            return None
    
    def check_convergence(self, theta: np.ndarray,
                         grad: Optional[np.ndarray] = None,
                         tol: float = 1e-5) -> bool:
        """
        Check convergence for BFGS.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        grad : np.ndarray or None
            Current gradient
        tol : float
            Convergence tolerance (looser for FP32)
            
        Returns
        -------
        bool
            True if converged
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Use looser tolerance for FP32
        max_grad = np.max(np.abs(grad))
        return max_grad < tol
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        torch = self.torch
        
        info = {
            'device': str(self.device),
            'dtype': 'float32',
            'parameterization': self.parameterization.__class__.__name__
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'memory_reserved': torch.cuda.memory_reserved(self.device)
            })
        elif self.device.type == 'mps':
            info['gpu_name'] = 'Apple Metal Performance Shaders'
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()