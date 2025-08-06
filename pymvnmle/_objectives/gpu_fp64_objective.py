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
        
        # Constants in FP64
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
            contrib = self._compute_pattern_contribution_gpu(
                gpu_pattern, mu_k, sigma_k
            )
            
            # Weight by number of observations
            obj_value = obj_value + gpu_pattern['n_obs'] * contrib
        
        return obj_value.squeeze()
    
    def _compute_pattern_contribution_gpu(self, pattern: Dict,
                                         mu_k: Any,
                                         sigma_k: Any) -> Any:
        """
        Compute pattern contribution with FP64 precision.
        
        Uses R-compatible formula: n_k * [const + log|Σ| + tr(Σ⁻¹S)]
        where S is the sample covariance matrix.
        
        Parameters
        ----------
        pattern : dict
            GPU pattern data
        mu_k : torch.Tensor
            Mean for observed variables (FP64)
        sigma_k : torch.Tensor
            Covariance for observed variables (FP64)
            
        Returns
        -------
        torch.Tensor
            Pattern contribution to objective (total, not per observation)
        """
        torch = self.torch
        
        n_obs = pattern['n_obs']
        n_obs_vars = pattern['n_observed']
        
        # Constant term for all observations in this pattern
        const_term = n_obs * n_obs_vars * self.log_2pi_gpu
        
        # Log determinant term for all observations
        L_k = torch.linalg.cholesky(sigma_k)
        log_det = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        log_det_term = n_obs * log_det
        
        # Compute sample covariance matrix (R-style)
        data_centered = pattern['data'] - mu_k  # shape: (n_obs, n_observed)
        S_k = (data_centered.T @ data_centered) / n_obs
        
        # Trace term: tr(Σ_k^-1 * S_k)
        # Use Cholesky solve for numerical stability
        Y = torch.linalg.solve_triangular(L_k, S_k, upper=False)
        Z = torch.linalg.solve_triangular(L_k.T, Y, upper=True)
        trace_term = torch.trace(Z)
        trace_term = n_obs * trace_term
        
        # Total contribution (not averaged)
        total_contribution = const_term + log_det_term + trace_term
        
        return total_contribution / n_obs  # Return per-observation average for consistency
    
    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """
        Unpack parameters on GPU with FP64.
        
        Parameters
        ----------
        theta_gpu : torch.Tensor
            Parameter vector on GPU (FP64)
            
        Returns
        -------
        mu : torch.Tensor
            Mean vector on GPU (FP64)
        sigma : torch.Tensor
            Covariance matrix on GPU (FP64)
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
        
        # Ensure exact symmetry (important even for FP64)
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
        Compute Hessian matrix using automatic differentiation.
        
        Enables Newton-CG optimization for FP64.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray
            Hessian matrix of shape (n_params, n_params)
        """
        torch = self.torch
        
        # Convert to GPU tensor
        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Compute gradient with create_graph=True for second derivatives
        if self.use_compiled:
            obj_value = self._compiled_objective(theta_gpu)
        else:
            obj_value = self._torch_objective(theta_gpu)
        
        # First derivatives
        grad = torch.autograd.grad(
            obj_value, theta_gpu, 
            create_graph=True, retain_graph=True
        )[0]
        
        # Compute Hessian row by row
        hessian = []
        for i in range(self.n_params):
            # Second derivative with respect to theta[i]
            grad2 = torch.autograd.grad(
                grad[i], theta_gpu,
                retain_graph=True
            )[0]
            hessian.append(grad2.cpu().numpy())
        
        return np.array(hessian)
    
    def compute_newton_direction(self, theta: np.ndarray,
                                grad: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Newton direction for Newton-CG.
        
        Solves: H d = -g where H is Hessian, g is gradient
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameter vector
        grad : np.ndarray or None
            Current gradient (computed if None)
            
        Returns
        -------
        np.ndarray
            Newton direction vector
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        hessian = self.compute_hessian(theta)
        
        # Solve H d = -g using Cholesky decomposition
        try:
            # Add small regularization for numerical stability
            hessian_reg = hessian + self.eps * np.eye(self.n_params)
            L = np.linalg.cholesky(hessian_reg)
            y = np.linalg.solve(L, -grad)
            direction = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse if Hessian is singular
            warnings.warn("Hessian is singular, using pseudoinverse")
            direction = -np.linalg.pinv(hessian) @ grad
        
        return direction
    
    def line_search(self, theta: np.ndarray, direction: np.ndarray,
                   grad: Optional[np.ndarray] = None,
                   alpha_init: float = 1.0) -> float:
        """
        Backtracking line search for Newton-CG.
        
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