"""
GPU backend using PyTorch with FP64 precision.

For data center GPUs (A100, H100, V100) with full-speed FP64 support.
Uses Newton-CG optimization which requires FP64 for Hessian stability.
"""

import numpy as np
import warnings
import time
from typing import Dict, Any, Optional, Union
from .base import GPUBackendFP64


class PyTorchBackendFP64(GPUBackendFP64):
    """
    PyTorch-based GPU backend with FP64 precision.
    
    Designed for:
    - NVIDIA A100 (Ampere data center)
    - NVIDIA H100 (Hopper data center)
    - NVIDIA V100 (Volta data center)
    - Google TPUs (via PyTorch/XLA)
    
    Notes
    -----
    - Uses float64 for maximum numerical precision
    - Only efficient on GPUs with full FP64 support
    - Enables Newton-CG optimization with analytical Hessians
    - Will warn if used on gimped FP64 hardware
    """
    
    def __init__(self, device: Optional[str] = None, verify_performance: bool = True):
        """
        Initialize PyTorch FP64 backend.
        
        Parameters
        ----------
        device : str or None
            Device to use: 'cuda:0', 'cuda:1', etc., or None for auto
        verify_performance : bool
            If True, benchmark FP64 performance on initialization
        """
        super().__init__()
        self.name = 'pytorch_fp64'
        
        # Import PyTorch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU acceleration. "
                "Install with: pip install torch"
            )
        
        # Set default dtype to float64
        self.torch.set_default_dtype(torch.float64)
        
        # Select device (must be CUDA for FP64)
        self.device = self._select_device(device)
        self.device_str = str(self.device)
        
        # Verify this is appropriate hardware for FP64
        if verify_performance:
            self._verify_fp64_performance()
        
        # Disable TF32 for maximum precision
        self._configure_for_precision()
    
    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select CUDA device for FP64 computation."""
        torch = self.torch
        
        # FP64 requires CUDA (not Metal)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FP64 backend requires CUDA. No CUDA devices found. "
                "Use FP32 backend for Apple Silicon or CPU."
            )
        
        if requested_device:
            # User specified device
            if not requested_device.startswith('cuda'):
                raise ValueError(
                    f"FP64 backend requires CUDA device, got {requested_device}"
                )
            return torch.device(requested_device)
        
        # Auto-select best CUDA device
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
        
        # Try to find A100/H100/V100 by name
        best_device = 0
        best_score = 0
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i).upper()
            score = self._score_gpu_for_fp64(name)
            if score > best_score:
                best_score = score
                best_device = i
        
        if best_score == 0:
            warnings.warn(
                f"No data center GPU detected. Using {torch.cuda.get_device_name(best_device)} "
                f"which may have gimped FP64 performance."
            )
        
        return torch.device(f'cuda:{best_device}')
    
    def _score_gpu_for_fp64(self, gpu_name: str) -> int:
        """Score GPU for FP64 suitability (higher is better)."""
        # Data center GPUs with full FP64
        if any(model in gpu_name for model in ['A100', 'A800', 'H100', 'H800']):
            return 100  # Latest data center
        if 'V100' in gpu_name:
            return 90   # Previous gen data center
        if any(model in gpu_name for model in ['P100', 'TESLA']):
            return 80   # Older data center
        if 'TITAN V' in gpu_name:
            return 70   # Titan V has good FP64
        
        # Consumer GPUs with gimped FP64
        return 0
    
    def _configure_for_precision(self) -> None:
        """Configure PyTorch for maximum FP64 precision."""
        torch = self.torch
        
        # Disable TF32 for maximum precision
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        
        # Enable highest precision cuDNN algorithms
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = False  # Deterministic
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = True
    
    def _verify_fp64_performance(self) -> None:
        """Verify FP64 performance is acceptable."""
        ratio = self.check_fp64_performance()
        
        if ratio < 0.1:  # Less than 1/10 of FP32 speed
            warnings.warn(
                f"FP64 performance is only {ratio:.1%} of FP32 on {self.get_device_info()['gpu_name']}. "
                f"This indicates gimped FP64 hardware. "
                f"Consider using FP32 backend for better performance."
            )
        elif ratio < 0.3:  # Less than 1/3 of FP32 speed
            warnings.warn(
                f"FP64 performance is {ratio:.1%} of FP32 on {self.get_device_info()['gpu_name']}. "
                f"This GPU has limited FP64 capabilities."
            )
    
    def check_fp64_performance(self) -> float:
        """
        Benchmark FP64 vs FP32 performance.
        
        Returns
        -------
        float
            Ratio of FP64 to FP32 throughput (0.5 is ideal)
        """
        torch = self.torch
        
        # Test matrix size
        n = 1024
        iterations = 100
        
        # Benchmark FP32
        A_fp32 = torch.randn(n, n, device=self.device, dtype=torch.float32)
        B_fp32 = torch.randn(n, n, device=self.device, dtype=torch.float32)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            C_fp32 = torch.matmul(A_fp32, B_fp32)
        torch.cuda.synchronize()
        time_fp32 = time.perf_counter() - start
        
        # Clear cache
        del A_fp32, B_fp32, C_fp32
        torch.cuda.empty_cache()
        
        # Benchmark FP64
        A_fp64 = torch.randn(n, n, device=self.device, dtype=torch.float64)
        B_fp64 = torch.randn(n, n, device=self.device, dtype=torch.float64)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            C_fp64 = torch.matmul(A_fp64, B_fp64)
        torch.cuda.synchronize()
        time_fp64 = time.perf_counter() - start
        
        # Calculate ratio
        ratio = time_fp32 / time_fp64  # Inverted because time is inverse of throughput
        
        return ratio
    
    def is_available(self) -> bool:
        """Check if CUDA with FP64 is available."""
        return self.device.type == 'cuda'
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        torch = self.torch
        
        props = torch.cuda.get_device_properties(self.device)
        
        info = {
            'backend_name': self.name,
            'device_type': self.device.type,
            'device_str': self.device_str,
            'precision': 'fp64',
            'optimization_method': 'Newton-CG',
            'supports_autodiff': True,
            'torch_version': torch.__version__,
            'gpu_name': props.name,
            'gpu_capability': (props.major, props.minor),
            'memory_total': props.total_memory,
            'memory_allocated': torch.cuda.memory_allocated(self.device),
            'multiprocessor_count': props.multi_processor_count,
            'cuda_cores': props.multi_processor_count * self._cores_per_sm(props.major),
            'tf32_enabled': False,  # Disabled for FP64 precision
            'fp64_fp32_ratio': self._theoretical_fp64_ratio(props.name)
        }
        
        return info
    
    def _cores_per_sm(self, major: int) -> int:
        """Get CUDA cores per streaming multiprocessor."""
        # Based on compute capability
        if major == 5:  # Maxwell
            return 128
        elif major == 6:  # Pascal
            return 64
        elif major == 7:  # Volta/Turing
            return 64
        elif major == 8:  # Ampere
            return 64
        elif major == 9:  # Hopper
            return 128
        else:
            return 64  # Default
    
    def _theoretical_fp64_ratio(self, gpu_name: str) -> float:
        """Get theoretical FP64:FP32 ratio for known GPUs."""
        gpu_upper = gpu_name.upper()
        
        # Full FP64 GPUs
        if any(x in gpu_upper for x in ['A100', 'H100', 'V100', 'P100']):
            return 0.5
        # Gimped consumer GPUs
        elif 'RTX 40' in gpu_upper or 'RTX 50' in gpu_upper:
            return 1/64
        elif 'RTX 30' in gpu_upper:
            return 1/64
        elif 'RTX 20' in gpu_upper:
            return 1/32
        else:
            return 1/32  # Conservative estimate
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        torch = self.torch
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device),
            'reserved': torch.cuda.memory_reserved(self.device),
            'total': torch.cuda.get_device_properties(self.device).total_memory
        }
    
    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer NumPy array to GPU as FP64 tensor.
        
        Parameters
        ----------
        array : np.ndarray
            NumPy array to transfer
            
        Returns
        -------
        torch.Tensor
            FP64 tensor on GPU
        """
        torch = self.torch
        
        # Convert to FP64 tensor
        tensor = torch.from_numpy(array.astype(np.float64))
        
        # Move to device
        return tensor.to(self.device)
    
    def to_numpy(self, device_array: Any) -> np.ndarray:
        """
        Transfer GPU tensor back to NumPy array.
        
        Parameters
        ----------
        device_array : torch.Tensor
            Tensor on GPU
            
        Returns
        -------
        np.ndarray
            NumPy array on CPU with float64 dtype
        """
        return device_array.detach().cpu().numpy().astype(np.float64)
    
    def cholesky(self, matrix: Any, upper: bool) -> Any:
        """
        Compute Cholesky decomposition with FP64 precision.
        
        Parameters
        ----------
        matrix : torch.Tensor
            Positive definite matrix on GPU
        upper : bool
            If True, return upper triangular factor
            
        Returns
        -------
        torch.Tensor
            Cholesky factor on GPU
        """
        torch = self.torch
        
        # Ensure FP64
        if matrix.dtype != torch.float64:
            matrix = matrix.to(torch.float64)
        
        # Compute Cholesky
        L = torch.linalg.cholesky(matrix)
        
        if upper:
            return L.T
        return L
    
    def solve_triangular(self, a: Any, b: Any, upper: bool, trans: str) -> Any:
        """
        Solve triangular system with FP64 precision.
        
        Parameters
        ----------
        a : torch.Tensor
            Triangular matrix on GPU
        b : torch.Tensor
            Right-hand side on GPU
        upper : bool
            Whether a is upper triangular
        trans : str
            'N' for a @ x = b, 'T' for a.T @ x = b
            
        Returns
        -------
        torch.Tensor
            Solution x on GPU
        """
        torch = self.torch
        
        # Ensure FP64
        if a.dtype != torch.float64:
            a = a.to(torch.float64)
        if b.dtype != torch.float64:
            b = b.to(torch.float64)
        
        # Apply transpose if needed
        if trans == 'T':
            a = a.T
            upper = not upper
        elif trans != 'N':
            raise ValueError(f"trans must be 'N' or 'T', got {trans}")
        
        # Solve
        return torch.linalg.solve_triangular(
            a, b,
            upper=upper,
            left=True,
            unitriangular=False
        )
    
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Matrix multiplication with FP64 precision.
        
        Parameters
        ----------
        a, b : torch.Tensor
            Matrices on GPU
            
        Returns
        -------
        torch.Tensor
            Matrix product on GPU
        """
        # Ensure FP64
        if a.dtype != self.torch.float64:
            a = a.to(self.torch.float64)
        if b.dtype != self.torch.float64:
            b = b.to(self.torch.float64)
        
        return self.torch.matmul(a, b)
    
    def inv(self, matrix: Any) -> Any:
        """
        Matrix inversion with FP64 precision.
        
        Parameters
        ----------
        matrix : torch.Tensor
            Square matrix on GPU
            
        Returns
        -------
        torch.Tensor
            Inverse matrix on GPU
        """
        # Ensure FP64
        if matrix.dtype != self.torch.float64:
            matrix = matrix.to(self.torch.float64)
        
        return self.torch.linalg.inv(matrix)
    
    def log_det(self, matrix: Any) -> float:
        """
        Compute log determinant with FP64 precision.
        
        Parameters
        ----------
        matrix : torch.Tensor
            Positive definite matrix on GPU
            
        Returns
        -------
        float
            Natural logarithm of determinant
        """
        torch = self.torch
        
        # Ensure FP64
        if matrix.dtype != torch.float64:
            matrix = matrix.to(torch.float64)
        
        # Use Cholesky for numerical stability
        try:
            L = self.cholesky(matrix, upper=False)
            return 2.0 * torch.sum(torch.log(torch.diag(L))).item()
        except torch.linalg.LinAlgError:
            # Fallback to slogdet
            sign, logdet = torch.linalg.slogdet(matrix)
            if sign <= 0:
                raise ValueError("Matrix is not positive definite")
            return logdet.item()
    
    def quadratic_form(self, x: Any, A: Any) -> float:
        """
        Compute quadratic form with FP64 precision.
        
        Parameters
        ----------
        x : torch.Tensor
            Vector on GPU
        A : torch.Tensor
            Positive definite matrix on GPU
            
        Returns
        -------
        float
            Scalar result
        """
        # Ensure FP64
        if x.dtype != self.torch.float64:
            x = x.to(self.torch.float64)
        if A.dtype != self.torch.float64:
            A = A.to(self.torch.float64)
        
        # Ensure x is a column vector
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Compute x.T @ A @ x
        result = x.T @ A @ x
        
        return result.item()
    
    def solve_posdef(self, A: Any, b: Any) -> Any:
        """
        Solve A @ x = b with FP64 precision where A is positive definite.
        
        Parameters
        ----------
        A : torch.Tensor
            Positive definite matrix on GPU
        b : torch.Tensor
            Right-hand side on GPU
            
        Returns
        -------
        torch.Tensor
            Solution x on GPU
        """
        torch = self.torch
        
        # Ensure FP64
        if A.dtype != torch.float64:
            A = A.to(torch.float64)
        if b.dtype != torch.float64:
            b = b.to(torch.float64)
        
        # Use Cholesky decomposition
        L = self.cholesky(A, upper=False)
        
        # Forward substitution
        y = self.solve_triangular(L, b, upper=False, trans='N')
        
        # Back substitution
        x = self.solve_triangular(L, y, upper=False, trans='T')
        
        return x
    
    def supports_autodiff(self) -> bool:
        """PyTorch supports automatic differentiation."""
        return True
    
    def enable_gradient_computation(self, tensor: Any) -> Any:
        """
        Enable gradient computation for FP64 tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to track gradients for
            
        Returns
        -------
        torch.Tensor
            Tensor with requires_grad=True
        """
        if tensor.dtype != self.torch.float64:
            tensor = tensor.to(self.torch.float64)
        return tensor.requires_grad_(True)
    
    def compute_gradient(self, loss: Any, parameters: Any) -> Any:
        """
        Compute gradients using automatic differentiation.
        
        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value
        parameters : torch.Tensor
            Parameters to compute gradients for
            
        Returns
        -------
        torch.Tensor
            Gradients with respect to parameters
        """
        torch = self.torch
        
        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            parameters,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return grads
    
    def compute_hessian(self, loss: Any, parameters: Any) -> Any:
        """
        Compute Hessian matrix for Newton-CG optimization.
        
        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value
        parameters : torch.Tensor
            Parameters to compute Hessian for
            
        Returns
        -------
        torch.Tensor
            Hessian matrix
            
        Notes
        -----
        This is why we need FP64 - Hessian computation is numerically sensitive.
        """
        torch = self.torch
        
        # First-order gradients
        grads = torch.autograd.grad(
            loss, 
            parameters, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second-order gradients (Hessian)
        hessian = []
        for i in range(len(grads)):
            grad2 = torch.autograd.grad(
                grads[i],
                parameters,
                retain_graph=True
            )[0]
            hessian.append(grad2)
        
        return torch.stack(hessian)
    
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        self.torch.cuda.synchronize(self.device)
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        self.torch.cuda.empty_cache()