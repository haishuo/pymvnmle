"""
GPU objective function using standard Cholesky parameterization with FP64.

For data center GPUs (A100, H100, V100) with full-speed FP64 support.
Enables Newton-CG optimization with analytical Hessians.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, Union
from .base import MLEObjectiveBase, PatternData
from .parameterizations import CholeskyParameterization


class GPUObjectiveFP64(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective with FP64 precision.
    
    Designed for data center GPUs with full FP64 support.
    Enables Newton-CG optimization with analytical second-order derivatives.
    
    Key features:
    - Full FP64 precision throughout
    - Analytical gradients and Hessians via autodiff
    - Newton-CG optimization support
    - Tighter numerical tolerances than FP32
    """
    
    def __init__(self, data: np.ndarray,
                 device: Optional[str] = None,
                 validate: bool = True,
                 verify_fp64_performance: bool = True,
                 compile_objective: bool = False):  # DISABLED due to trace issues
        """
        Initialize GPU FP64 objective.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data with missing values as np.nan
        device : str or None
            PyTorch device (must be CUDA for FP64)
        validate : bool
            Whether to validate input data
        verify_fp64_performance : bool
            Whether to check FP64 performance on initialization
        compile_objective : bool
            Whether to compile objective with torch.compile
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
        
        # Select device (must be CUDA for FP64)
        self.device = self._select_device(device)
        
        # FP64 settings
        self.dtype = torch.float64
        self.eps = 1e-12  # Tighter epsilon for FP64
        
        # Check FP64 performance
        if verify_fp64_performance and self.device.type == 'cuda':
            self._check_fp64_performance()
        
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
        """Select appropriate GPU device for FP64."""
        torch = self.torch
        
        if requested_device:
            device = torch.device(requested_device)
            if device.type != 'cuda':
                warnings.warn(
                    f"FP64 objective requested on {device.type}. "
                    f"Performance may be poor. Consider using FP32 objective."
                )
            return device
        
        # Auto-select: prefer CUDA for FP64
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            warnings.warn(
                "No CUDA device available for FP64. "
                "Using CPU fallback (will be very slow)."
            )
            return torch.device('cpu')
    
    def _check_fp64_performance(self) -> None:
        """Check if GPU has good FP64 performance."""
        torch = self.torch
        import time
        
        # Quick benchmark
        n = 512
        A = torch.randn(n, n, device=self.device, dtype=torch.float32)
        B = torch.randn(n, n, device=self.device, dtype=torch.float32)
        
        # FP32 timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            C32 = torch.matmul(A, B)
        torch.cuda.synchronize()
        time_fp32 = time.perf_counter() - start
        
        # FP64 timing
        A64 = A.to(torch.float64)
        B64 = B.to(torch.float64)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            C64 = torch.matmul(A64, B64)
        torch.cuda.synchronize()
        time_fp64 = time.perf_counter() - start
        
        ratio = time_fp32 / time_fp64
        
        if ratio < 0.1:  # FP64 is >10x slower
            gpu_name = torch.cuda.get_device_name(self.device)
            warnings.warn(
                f"Poor FP64 performance detected on {gpu_name}. "
                f"FP64 is {1/ratio:.1f}x slower than FP32. "
                f"Consider using FP32 objective for better performance."
            )
    
    def _prepare_gpu_data(self) -> None:
        """Transfer pattern data to GPU with FP64 precision."""
        torch = self.torch
        
        self.gpu_patterns = []
        
        for pattern in self.patterns:
            # Convert data to GPU tensors with FP64
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
                    dtype=self.dtype  # FP64
                )
            }
            
            gpu_pattern['n_observed'] = len(pattern.observed_indices)
            self.gpu_patterns.append(gpu_pattern)
        
        # Note: We do NOT store log(2π) since CPU doesn't use it
        # self.log_2pi_gpu = torch.tensor(...)  # REMOVED
    
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
        Compute -2 * log-likelihood using GPU with FP64.
        
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
        
        # Convert to GPU tensor (FP64)
        theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
        
        # Compute objective
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
            Parameter vector on GPU (FP64)
            
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
            
            # Compute pattern contribution (no regularization needed for FP64)
            # CRITICAL: This returns the TOTAL contribution for all n_obs in the pattern
            contrib = self._compute_pattern_contribution_gpu(
                gpu_pattern, mu_k, sigma_k
            )
            
            # Add contribution directly - do NOT multiply by n_obs!
            obj_value = obj_value + contrib
        
        return obj_value.squeeze()
    
    def _compute_pattern_contribution_gpu(self, pattern: Dict,
                                         mu_k: Any,
                                         sigma_k: Any) -> Any:
        """
        Compute pattern contribution with FP64 precision.
        
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
        # This is the key fix that makes GPU match CPU
        
        # Log determinant term: log|Σ_k|
        L_k = torch.linalg.cholesky(sigma_k)
        log_det_term = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        
        # Compute sample covariance matrix
        # S_k = (1/n) * Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
        data_centered = pattern['data'] - mu_k
        S_k = (data_centered.T @ data_centered) / n_obs
        
        # Trace term: tr(Σ_k^-1 * S_k)
        X = torch.linalg.solve(sigma_k, S_k)
        trace_term = torch.trace(X)
        
        # Total contribution: n_obs * [log|Σ| + tr(Σ^-1 S)]
        # NO CONSTANT TERM to match CPU!
        total_contribution = n_obs * (log_det_term + trace_term)
        
        return total_contribution
    
    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """
        Unpack parameters on GPU with FP64.
        
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
        
        # Compute objective
        obj_value = self._torch_objective(theta_gpu)
        
        # Compute gradient via autodiff
        obj_value.backward()
        
        # Return as NumPy array
        return theta_gpu.grad.cpu().numpy()
    
    def compute_hessian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix using automatic differentiation.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Hessian matrix
        """
        torch = self.torch
        
        # Convert to GPU tensor
        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Compute gradient with grad graph retained
        obj_value = self._torch_objective(theta_gpu)
        grad = torch.autograd.grad(
            obj_value, theta_gpu, create_graph=True
        )[0]
        
        # Compute Hessian row by row
        n = len(theta)
        hessian = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        
        for i in range(n):
            # Second derivatives
            grad2 = torch.autograd.grad(
                grad[i], theta_gpu, retain_graph=True
            )[0]
            hessian[i] = grad2
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        return hessian.cpu().numpy()
    
    def compute_newton_direction(self, theta: np.ndarray,
                                grad: Optional[np.ndarray] = None,
                                hess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Newton direction for Newton-CG optimization.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        grad : np.ndarray or None
            Current gradient
        hess : np.ndarray or None
            Current Hessian
            
        Returns
        -------
        np.ndarray
            Newton direction
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        if hess is None:
            hess = self.compute_hessian(theta)
        
        # Regularize Hessian if needed
        eigenvals = np.linalg.eigvalsh(hess)
        if eigenvals.min() < 1e-8:
            hess = hess + (1e-8 - eigenvals.min()) * np.eye(len(theta))
        
        # Solve Hessian * direction = -gradient
        try:
            direction = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            # Fall back to gradient descent
            direction = -grad
        
        return direction
    
    def line_search(self, theta: np.ndarray,
                   direction: np.ndarray,
                   grad: Optional[np.ndarray] = None,
                   alpha_init: float = 1.0) -> float:
        """
        Perform line search along Newton direction.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        direction : np.ndarray
            Search direction
        grad : np.ndarray or None
            Current gradient
        alpha_init : float
            Initial step size
            
        Returns
        -------
        float
            Optimal step size
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Armijo parameters
        c1 = 1e-4
        rho = 0.5
        
        # Initial objective and directional derivative
        f0 = self.compute_objective(theta)
        df0 = np.dot(grad, direction)
        
        # Backtracking line search
        alpha = alpha_init
        max_iter = 20
        
        for _ in range(max_iter):
            theta_new = theta + alpha * direction
            f_new = self.compute_objective(theta_new)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * df0:
                return alpha
            
            alpha *= rho
        
        return alpha
    
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
        # Unpack using CPU
        mu, sigma = self.parameterization.unpack(theta)
        
        # Compute objective
        neg2_loglik = self.compute_objective(theta)
        loglik = -0.5 * neg2_loglik
        
        return mu, sigma, loglik
    
    def check_convergence(self, theta: np.ndarray,
                         grad: Optional[np.ndarray] = None,
                         tol: float = 1e-8) -> bool:
        """
        Check convergence for Newton-CG.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        grad : np.ndarray or None
            Current gradient
        tol : float
            Convergence tolerance (tighter for FP64)
            
        Returns
        -------
        bool
            True if converged
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Tighter tolerance for FP64
        max_grad = np.max(np.abs(grad))
        return max_grad < tol
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        torch = self.torch
        
        info = {
            'device': str(self.device),
            'dtype': 'float64',
            'using_compiled': self.use_compiled
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'memory_reserved': torch.cuda.memory_reserved(self.device),
                'fp64_capable': True
            })
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()
    
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