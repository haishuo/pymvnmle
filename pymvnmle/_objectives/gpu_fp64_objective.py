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
                 compile_objective: bool = True):
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
        
        # Verify FP64 performance if requested
        if verify_fp64_performance:
            self._verify_fp64_performance()
        
        # FP64 settings
        self.dtype = torch.float64
        self.eps = 1e-10  # Much smaller epsilon for FP64
        
        # Transfer pattern data to GPU
        self._prepare_gpu_data()
        
        # Compile objective for performance
        if compile_objective and hasattr(torch, 'compile'):
            try:
                self._compiled_objective = torch.compile(self._torch_objective)
                self._compiled_gradient = torch.compile(self._compute_gradient_torch)
                self.use_compiled = True
            except:
                self.use_compiled = False
                warnings.warn("Failed to compile objective, using eager mode")
        else:
            self.use_compiled = False
    
    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select CUDA device for FP64 computation."""
        torch = self.torch
        
        # FP64 requires CUDA (not Metal)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FP64 GPU objective requires CUDA. No CUDA devices found. "
                "Use FP32 objective for Apple Silicon or CPU objective."
            )
        
        if requested_device:
            if not requested_device.startswith('cuda'):
                raise ValueError(
                    f"FP64 objective requires CUDA device, got {requested_device}"
                )
            return torch.device(requested_device)
        
        # Auto-select best CUDA device for FP64
        device_count = torch.cuda.device_count()
        best_device = 0
        best_memory = 0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # Prefer devices with more memory for FP64
            if props.total_memory > best_memory:
                best_memory = props.total_memory
                best_device = i
        
        return torch.device(f'cuda:{best_device}')
    
    def _verify_fp64_performance(self) -> None:
        """Check if GPU has acceptable FP64 performance."""
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
            Scalar objective value
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
            Pattern contribution to objective
        """
        torch = self.torch
        
        n_obs_vars = pattern['n_observed']
        
        # Constant term
        const_term = n_obs_vars * self.log_2pi_gpu
        
        # Log determinant using Cholesky (stable with FP64)
        L_k = torch.linalg.cholesky(sigma_k)
        log_det = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        
        # Compute sample covariance
        data_centered = pattern['data'] - mu_k
        S_k = (data_centered.T @ data_centered) / pattern['n_obs']
        
        # Trace term: tr(Σ_k^-1 * S_k)
        # Use Cholesky solve for numerical stability
        Y = torch.linalg.solve_triangular(L_k, S_k, upper=False)
        X = torch.linalg.solve_triangular(L_k.T, Y, upper=True)
        trace_term = torch.trace(X)
        
        return const_term + log_det + trace_term
    
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
        
        if self.use_compiled:
            grad_gpu = self._compiled_gradient(theta_gpu)
        else:
            grad_gpu = self._compute_gradient_torch(theta_gpu)
        
        return grad_gpu.cpu().numpy()
    
    def _compute_gradient_torch(self, theta_gpu: Any) -> Any:
        """Compute gradient on GPU."""
        # Compute objective
        obj_value = self._torch_objective(theta_gpu)
        
        # Compute gradient via autodiff
        grad = self.torch.autograd.grad(obj_value, theta_gpu)[0]
        
        return grad
    
    def compute_hessian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute full Hessian matrix using automatic differentiation.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
            
        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix
            
        Notes
        -----
        This is the key capability that FP64 enables - accurate
        Hessian computation for Newton-CG optimization.
        """
        torch = self.torch
        
        # Convert to GPU tensor
        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Compute gradient with computation graph
        obj_value = self._torch_objective(theta_gpu)
        grad = torch.autograd.grad(obj_value, theta_gpu, create_graph=True)[0]
        
        # Compute Hessian row by row
        hessian = []
        for i in range(self.n_params):
            # Second derivative with respect to theta[i]
            grad2 = torch.autograd.grad(
                grad[i],
                theta_gpu,
                retain_graph=True
            )[0]
            hessian.append(grad2)
        
        # Stack into matrix
        hessian = torch.stack(hessian)
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        return hessian.cpu().numpy()
    
    def compute_newton_direction(self, theta: np.ndarray,
                                grad: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Newton direction for Newton-CG.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameter vector
        grad : np.ndarray or None
            Current gradient (computed if None)
            
        Returns
        -------
        np.ndarray
            Newton direction: -H^{-1} @ grad
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Compute Hessian
        hessian = self.compute_hessian(theta)
        
        # Add small regularization for numerical stability
        hessian += self.eps * np.eye(self.n_params)
        
        # Solve H @ d = -grad for Newton direction
        try:
            direction = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            # Fall back to gradient descent if Hessian is singular
            warnings.warn("Hessian near singular, using gradient direction")
            direction = -grad
        
        return direction
    
    def line_search(self, theta: np.ndarray, direction: np.ndarray,
                   alpha_init: float = 1.0) -> float:
        """
        Backtracking line search for Newton-CG.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameters
        direction : np.ndarray
            Search direction
        alpha_init : float
            Initial step size
            
        Returns
        -------
        float
            Optimal step size
        """
        # Armijo backtracking parameters
        c1 = 1e-4
        backtrack = 0.5
        
        # Current objective value
        f0 = self.compute_objective(theta)
        
        # Gradient in search direction
        grad = self.compute_gradient(theta)
        slope = np.dot(grad, direction)
        
        alpha = alpha_init
        
        # Backtracking loop
        for _ in range(20):  # Max iterations
            theta_new = theta + alpha * direction
            f_new = self.compute_objective(theta_new)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * slope:
                break
            
            alpha *= backtrack
        
        return alpha
    
    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mean, covariance, and log-likelihood.
        
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
            Convergence tolerance (tight for FP64)
            
        Returns
        -------
        bool
            True if converged
        """
        if grad is None:
            grad = self.compute_gradient(theta)
        
        # Tight tolerance for FP64
        max_grad = np.max(np.abs(grad))
        return max_grad < tol
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        torch = self.torch
        
        props = torch.cuda.get_device_properties(self.device)
        
        info = {
            'device': str(self.device),
            'dtype': 'float64',
            'gpu_name': props.name,
            'memory_total': props.total_memory,
            'memory_allocated': torch.cuda.memory_allocated(self.device),
            'multiprocessor_count': props.multi_processor_count,
            'using_compiled': self.use_compiled,
            'supports_newton_cg': True
        }
        
        # Check if this is a data center GPU
        gpu_name_upper = props.name.upper()
        is_datacenter = any(
            model in gpu_name_upper
            for model in ['A100', 'A800', 'H100', 'H800', 'V100']
        )
        info['is_datacenter_gpu'] = is_datacenter
        
        if not is_datacenter:
            info['warning'] = 'This GPU may have poor FP64 performance'
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        self.torch.cuda.empty_cache()