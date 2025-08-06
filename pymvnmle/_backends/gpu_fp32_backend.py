"""
GPU backend using PyTorch with FP32 precision.

Optimized for consumer GPUs (RTX series, Apple Metal) where FP64 is gimped.
Uses BFGS optimization which is more stable with FP32 precision.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, Union
from .base import GPUBackendFP32


class PyTorchBackendFP32(GPUBackendFP32):
    """
    PyTorch-based GPU backend with FP32 precision.
    
    Designed for:
    - Consumer NVIDIA GPUs (RTX 4090, RTX 5070 Ti, etc.)
    - Apple Silicon GPUs (M1/M2/M3/M4)
    - Any GPU where FP64 is gimped or unavailable
    
    Notes
    -----
    - Uses float32 for optimal GPU throughput
    - Supports both CUDA and Metal Performance Shaders
    - Enables automatic differentiation for gradients
    - Falls back to CPU if GPU runs out of memory
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch FP32 backend.
        
        Parameters
        ----------
        device : str or None
            Device to use: 'cuda', 'mps', or None for auto-detect
        """
        super().__init__()
        self.name = 'pytorch_fp32'
        
        # Import PyTorch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU acceleration. "
                "Install with: pip install torch"
            )
        
        # Set default dtype to float32
        self.torch.set_default_dtype(torch.float32)
        
        # Select device
        self.device = self._select_device(device)
        self.device_str = str(self.device)
        
        # Enable optimizations
        self._configure_optimizations()
    
    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device."""
        torch = self.torch
        
        if requested_device:
            # User specified device
            if requested_device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if requested_device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("Metal requested but not available")
            return torch.device(requested_device)
        
        # Auto-detect best device
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            warnings.warn(
                "No GPU available, falling back to CPU. "
                "This will be slower than the dedicated CPU backend."
            )
            return torch.device('cpu')
    
    def _configure_optimizations(self) -> None:
        """Configure PyTorch optimizations for FP32."""
        torch = self.torch
        
        # Enable TF32 on Ampere GPUs (RTX 30xx and newer)
        if self.device.type == 'cuda':
            if hasattr(torch.backends.cuda, 'matmul'):
                # TF32 gives speedup with minimal accuracy loss
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
        # Set number of threads for CPU operations
        if self.device.type == 'cpu':
            torch.set_num_threads(4)  # Limit CPU threads when using GPU backend
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self.device.type in ['cuda', 'mps']
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        torch = self.torch
        info = {
            'backend_name': self.name,
            'device_type': self.device.type,
            'device_str': self.device_str,
            'precision': 'fp32',
            'optimization_method': 'BFGS',
            'supports_autodiff': True,
            'torch_version': torch.__version__
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'gpu_capability': torch.cuda.get_device_capability(self.device),
                'memory_total': torch.cuda.get_device_properties(self.device).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'tf32_enabled': torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda, 'matmul') else False
            })
        elif self.device.type == 'mps':
            info.update({
                'gpu_name': 'Apple Metal Performance Shaders',
                'memory_allocated': torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 'unknown'
            })
        
        return info
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        torch = self.torch
        
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device),
                'reserved': torch.cuda.memory_reserved(self.device),
                'total': torch.cuda.get_device_properties(self.device).total_memory
            }
        elif self.device.type == 'mps':
            return {
                'allocated': torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0,
                'reserved': 0,  # MPS doesn't report reserved memory
                'total': 0  # MPS doesn't report total memory
            }
        else:
            return {'allocated': 0, 'reserved': 0, 'total': 0}
    
    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer NumPy array to GPU as FP32 tensor.
        
        Parameters
        ----------
        array : np.ndarray
            NumPy array to transfer
            
        Returns
        -------
        torch.Tensor
            FP32 tensor on GPU
        """
        torch = self.torch
        
        # Convert to FP32 tensor
        tensor = torch.from_numpy(array.astype(np.float32))
        
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
            NumPy array on CPU
        """
        # Move to CPU and convert to NumPy
        return device_array.detach().cpu().numpy()
    
    def cholesky(self, matrix: Any, upper: bool) -> Any:
        """
        Compute Cholesky decomposition on GPU.
        
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
        
        try:
            # PyTorch's cholesky always returns lower triangular
            L = torch.linalg.cholesky(matrix)
            
            if upper:
                # Return transpose for upper triangular
                return L.T
            return L
            
        except torch.linalg.LinAlgError as e:
            # Try adding small diagonal for numerical stability
            eps = 1e-6
            matrix_stabilized = matrix + eps * torch.eye(
                matrix.shape[0], 
                device=self.device, 
                dtype=torch.float32
            )
            
            try:
                L = torch.linalg.cholesky(matrix_stabilized)
                warnings.warn(
                    f"Matrix near singular, added {eps} to diagonal for stability"
                )
                return L.T if upper else L
            except:
                raise e
    
    def solve_triangular(self, a: Any, b: Any, upper: bool, trans: str) -> Any:
        """
        Solve triangular system on GPU.
        
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
        
        # Apply transpose if needed
        if trans == 'T':
            a = a.T
            upper = not upper
        elif trans != 'N':
            raise ValueError(f"trans must be 'N' or 'T', got {trans}")
        
        # Solve using torch.linalg.solve_triangular
        return torch.linalg.solve_triangular(
            a, b, 
            upper=upper,
            left=True,
            unitriangular=False
        )
    
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Matrix multiplication on GPU.
        
        Parameters
        ----------
        a, b : torch.Tensor
            Matrices on GPU
            
        Returns
        -------
        torch.Tensor
            Matrix product on GPU
        """
        return self.torch.matmul(a, b)
    
    def inv(self, matrix: Any) -> Any:
        """
        Matrix inversion on GPU.
        
        Parameters
        ----------
        matrix : torch.Tensor
            Square matrix on GPU
            
        Returns
        -------
        torch.Tensor
            Inverse matrix on GPU
        """
        return self.torch.linalg.inv(matrix)
    
    def log_det(self, matrix: Any) -> float:
        """
        Compute log determinant on GPU.
        
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
        
        # Use slogdet for numerical stability
        sign, logdet = torch.linalg.slogdet(matrix)
        
        if sign <= 0:
            # Try Cholesky-based computation
            try:
                L = self.cholesky(matrix, upper=False)
                return 2.0 * torch.sum(torch.log(torch.diag(L))).item()
            except:
                raise ValueError("Matrix is not positive definite")
        
        return logdet.item()
    
    def quadratic_form(self, x: Any, A: Any) -> float:
        """
        Compute quadratic form x.T @ A @ x on GPU.
        
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
        # Ensure x is a column vector
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Compute x.T @ A @ x
        result = x.T @ A @ x
        
        return result.item()
    
    def solve_posdef(self, A: Any, b: Any) -> Any:
        """
        Solve A @ x = b where A is positive definite, on GPU.
        
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
        
        try:
            # Use Cholesky decomposition for efficiency
            L = self.cholesky(A, upper=False)
            
            # Forward substitution
            y = self.solve_triangular(L, b, upper=False, trans='N')
            
            # Back substitution
            x = self.solve_triangular(L, y, upper=False, trans='T')
            
            return x
            
        except torch.linalg.LinAlgError:
            # Fallback to general solver
            return torch.linalg.solve(A, b)
    
    def supports_autodiff(self) -> bool:
        """PyTorch supports automatic differentiation."""
        return True
    
    def enable_gradient_computation(self, tensor: Any) -> Any:
        """
        Enable gradient computation for a tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to track gradients for
            
        Returns
        -------
        torch.Tensor
            Tensor with requires_grad=True
        """
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
    
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            # MPS doesn't have explicit sync, operations are synchronous
            pass
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            # MPS doesn't expose cache control
            pass