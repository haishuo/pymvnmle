"""
GPU objective function using standard Cholesky parameterization with FP32.

Optimized for consumer GPUs (RTX series, Apple Metal) where FP64 is gimped.
Uses PyTorch autodiff for analytical gradients and BFGS-friendly parameterization.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, Union
from .base import MLEObjectiveBase, PatternData
from .parameterizations import CholeskyParameterization, BoundedCholeskyParameterization


class GPUObjectiveFP32(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective with FP32 precision.
    
    Designed for consumer GPUs where FP32 runs at full speed.
    Uses standard Cholesky parameterization which is more natural
    for autodiff and more stable with lower precision.
    
    Key differences from CPU objective:
    - Standard Cholesky (L where Σ = LL') instead of inverse
    - Analytical gradients via autodiff instead of finite differences
    - FP32 precision with numerical safeguards
    - Batched operations for GPU efficiency
    """
    
    def __init__(self, data: np.ndarray, 
                 device: Optional[str] = None,
                 validate: bool = True,
                 use_bounded: bool = False):
        """
        Initialize GPU FP32 objective.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data with missing values as np.nan
        device : str or None
            PyTorch device ('cuda', 'mps', or None for auto)
        validate : bool
            Whether to validate input data
        use_bounded : bool
            Use bounded parameterization for FP32 stability
        """
        # Initialize base class (handles preprocessing)
        super().__init__(data, validate)
        
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
            self.parameterization = BoundedCholeskyParameterization(self.n_vars)
        else:
            self.parameterization = CholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        
        # Select device
        self.device = self._select_device(device)
        
        # FP32 settings
        self.dtype = torch.float32
        self.eps = 1e-6  # Larger epsilon for FP32 stability
        
        # Transfer pattern data to GPU
        self._prepare_gpu_data()
    
    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device."""
        torch = self.torch
        
        if requested_device:
            return torch.device(requested_device)
        
        # Auto-select best available device
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            warnings.warn("No GPU available, using CPU (will be slower)")
            return torch.device('cpu')
    
    def _prepare_gpu_data(self) -> None:
        """Transfer pattern data to GPU for efficient computation."""
        torch = self.torch
        
        self.gpu_patterns = []
        
        for pattern in self.patterns:
            # Convert data to GPU tensors
            gpu_pattern = {
                'n_obs': pattern.n_obs,
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
            
            # Store number of observed variables
            if len(pattern.observed_indices) > 0:
                gpu_pattern['n_observed'] = len(pattern.observed_indices)
            else:
                gpu_pattern['n_observed'] = 0
            
            self.gpu_patterns.append(gpu_pattern)
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameters using standard Cholesky.
        
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            self.sample_cov
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
        
        # Convert to GPU tensor (no gradient needed for objective only)
        theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
        
        # Compute objective without gradient tracking
        with torch.no_grad():
            obj_value = self._torch_objective(theta_gpu)
        
        # Return as Python float
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
        
        # Center data and compute quadratic form
        data_centered = data - mu_k.unsqueeze(0)  # Broadcasting
        
        # Compute quadratic form: sum_i (x_i - μ)^T Σ^{-1} (x_i - μ)
        Y = torch.linalg.solve_triangular(L_k, data_centered.T, upper=False)
        quad_form = torch.sum(Y * Y)  # ||Y||^2_F
        
        # Total contribution: n_obs * log|Σ| + quad_form
        # NO CONSTANT TERM to match CPU!
        total_contribution = n_obs * log_det_term + quad_form

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
            L = L + torch.diag(torch.sqrt(diag_vars))
            
            # Off-diagonal: tanh transformation  
            idx = 2 * n
            tril_indices = torch.tril_indices(n, n, offset=-1, device=self.device)
            if len(tril_indices[0]) > 0:
                tril_unbounded = theta_gpu[idx:]
                corr_max = self.parameterization.corr_max
                corr_vals = corr_max * torch.tanh(tril_unbounded)
                
                # Scale by diagonal
                i, j = tril_indices
                L[tril_indices[0], tril_indices[1]] = corr_vals * torch.sqrt(L[i, i] * L[j, j])
        else:
            # Standard Cholesky
            L = torch.zeros((n, n), device=self.device, dtype=self.dtype)
            
            # Diagonal elements
            L_diag = torch.exp(theta_gpu[n:2*n])
            L = L + torch.diag(L_diag)
            
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
            # Return appropriate bounds for L-BFGS-B
            bounds = []
            n = self.n_vars
            
            # Mean parameters: unbounded
            for i in range(n):
                bounds.append((None, None))
            
            # Remaining parameters handled by bounded parameterization
            # which uses sigmoid/tanh transformations internally
            n_cov_params = self.n_params - n
            for i in range(n_cov_params):
                bounds.append((None, None))  # Unbounded in transformed space
            
            return bounds
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