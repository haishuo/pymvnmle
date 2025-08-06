"""
GPU objective function using standard Cholesky parameterization with FP32.

Optimized for consumer GPUs (RTX series, Apple Metal) where FP64 is gimped.
Uses PyTorch autodiff for analytical gradients and BFGS-friendly parameterization.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, Union
from .base import MLEObjectiveBase, PatternData
from .parameterizations import CholeskyParameterization


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
                 compile_objective: bool = False):  # DISABLED due to trace issues
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
        compile_objective : bool
            Whether to compile objective with torch.compile for speed
            (DISABLED by default due to trace operation issues)
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
        
        # Create parameterization (standard Cholesky for GPU)
        self.parameterization = CholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        
        # Select device
        self.device = self._select_device(device)
        
        # FP32 settings
        self.dtype = torch.float32
        self.eps = 1e-6  # Larger epsilon for FP32 stability
        
        # Transfer pattern data to GPU
        self._prepare_gpu_data()
        
        # Disable compilation due to trace backward issues
        self.use_compiled = False
        if compile_objective:
            warnings.warn(
                "torch.compile is disabled for GPU objectives due to trace operation compatibility issues",
                RuntimeWarning
            )
    
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
        
        # Precompute constants on GPU
        self.log_2pi_gpu = torch.tensor(
            np.log(2 * np.pi), 
            device=self.device, 
            dtype=self.dtype
        )
    
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
        
        # Convert to GPU tensor
        theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
        
        # Compute objective
        if self.use_compiled:
            obj_value = self._compiled_objective(theta_gpu)
        else:
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
            
            # Extract observed submatrices
            obs_idx = gpu_pattern['observed_indices']
            mu_k = mu_gpu[obs_idx]
            sigma_k = sigma_gpu[obs_idx][:, obs_idx]
            
            # Add small diagonal for FP32 stability
            sigma_k = sigma_k + self.eps * torch.eye(
                gpu_pattern['n_observed'],
                device=self.device,
                dtype=self.dtype
            )
            
            # Compute pattern contribution
            # CRITICAL: This returns the TOTAL contribution for all n_obs in the pattern
            # Do NOT multiply by n_obs again!
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
        n_obs_vars = pattern['n_observed']

        # NO CONSTANT TERM! The CPU doesn't include n*p*log(2π)
        # const_term = n_obs_vars * self.log_2pi_gpu  # REMOVED!
        
        # Log determinant term: log|Σ_k|
        try:
            L_k = torch.linalg.cholesky(sigma_k)
            log_det_term = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        except:
            # Fallback for near-singular matrices
            eigenvals = torch.linalg.eigvalsh(sigma_k)
            eigenvals = torch.clamp(eigenvals, min=self.eps)
            log_det_term = torch.sum(torch.log(eigenvals))
        
        # Compute sample covariance matrix
        # S_k = (1/n) * Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
        data_centered = pattern['data'] - mu_k  # shape: (n_obs, n_observed)
        S_k = (data_centered.T @ data_centered) / n_obs
        
        # Trace term: tr(Σ_k^-1 * S_k)
        try:
            # Solve Σ_k * X = S_k for X, then tr(X) = tr(Σ_k^-1 * S_k)
            X = torch.linalg.solve(sigma_k, S_k)
            trace_term = torch.trace(X)
        except:
            # Fallback using Cholesky solve
            L_k = torch.linalg.cholesky(sigma_k)
            Y = torch.linalg.solve_triangular(L_k, S_k, upper=False)
            Z = torch.linalg.solve_triangular(L_k.T, Y, upper=True)
            trace_term = torch.trace(Z)
        
        # Total contribution: n_obs * [log|Σ| + tr(Σ^-1 S)]
        # NO CONSTANT TERM to match CPU!
        total_contribution = n_obs * (log_det_term + trace_term)

        return total_contribution
    
    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """
        Unpack parameters on GPU.
        
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
        
        # Extract mean
        mu = theta_gpu[:n]
        
        # Reconstruct L matrix (lower triangular)
        L = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        
        # Diagonal elements (exponentiated)
        L.diagonal().copy_(torch.exp(theta_gpu[n:2*n]))
        
        # Off-diagonal elements
        idx = 2 * n
        for j in range(n):
            for i in range(j + 1, n):
                L[i, j] = theta_gpu[idx]
                idx += 1
        
        # Compute Σ = LL'
        sigma = L @ L.T
        
        # Ensure symmetry (important for FP32)
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
        
        # Compute objective (never use compiled version for gradients)
        obj_value = self._torch_objective(theta_gpu)
        
        # Compute gradient via autodiff
        obj_value.backward()
        
        # Return as NumPy array
        return theta_gpu.grad.cpu().numpy()
    
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
            "Use BFGS optimization which only requires gradients."
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
        # Unpack using CPU (for returning NumPy arrays)
        mu, sigma = self.parameterization.unpack(theta)
        
        # Compute objective
        neg2_loglik = self.compute_objective(theta)
        loglik = -0.5 * neg2_loglik
        
        return mu, sigma, loglik
    
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
            'using_compiled': self.use_compiled
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
        # MPS doesn't have cache control
    
    def to_cpu(self) -> 'CPUObjectiveFP64':
        """
        Convert to CPU objective for comparison.
        
        Returns
        -------
        CPUObjectiveFP64
            Equivalent CPU objective
        """
        from .cpu_fp64_objective import CPUObjectiveFP64
        return CPUObjectiveFP64(self.original_data)