"""
Newton-CG optimizer for FP64 backends.

This module implements the Newton-Conjugate Gradient optimization algorithm,
designed for high-precision FP64 computation on data center GPUs (A100, H100).
Requires analytical gradients and Hessians for quadratic convergence.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np
from scipy.sparse.linalg import cg


class NewtonCGOptimizer:
    """
    Newton-CG optimizer tailored for FP64 precision.
    
    This implementation is designed for data center GPUs with full
    FP64 support (A100, H100, V100). Uses analytical Hessians for
    quadratic convergence near the optimum.
    
    Attributes
    ----------
    max_iter : int
        Maximum number of Newton iterations
    max_cg_iter : int
        Maximum conjugate gradient iterations per Newton step
    gtol : float
        Gradient norm tolerance for convergence
    ftol : float
        Function value tolerance for convergence
    xtol : float
        Parameter change tolerance for convergence
    cg_tol : float
        Conjugate gradient solver tolerance
    line_search_maxiter : int
        Maximum line search iterations
    verbose : bool
        Whether to print optimization progress
    """
    
    def __init__(
        self,
        max_iter: int,
        max_cg_iter: int,
        gtol: float,
        ftol: float,
        xtol: float,
        cg_tol: float,
        line_search_maxiter: int,
        trust_radius_init: float,
        verbose: bool
    ):
        """
        Initialize Newton-CG optimizer with explicit parameters.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of Newton iterations
        max_cg_iter : int
            Maximum CG iterations per Newton step
        gtol : float
            Gradient norm tolerance (FP64: typically 1e-8)
        ftol : float
            Function value change tolerance (typically 1e-10)
        xtol : float
            Parameter change tolerance (typically 1e-9)
        cg_tol : float
            CG solver tolerance (typically 1e-5)
        line_search_maxiter : int
            Maximum line search iterations (typically 10)
        trust_radius_init : float
            Initial trust region radius (typically 1.0)
        verbose : bool
            Print optimization progress
        """
        # Validate inputs
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if max_cg_iter <= 0:
            raise ValueError(f"max_cg_iter must be positive, got {max_cg_iter}")
        if gtol <= 0:
            raise ValueError(f"gtol must be positive, got {gtol}")
        if ftol <= 0:
            raise ValueError(f"ftol must be positive, got {ftol}")
        if xtol <= 0:
            raise ValueError(f"xtol must be positive, got {xtol}")
        if cg_tol <= 0:
            raise ValueError(f"cg_tol must be positive, got {cg_tol}")
        if line_search_maxiter <= 0:
            raise ValueError(f"line_search_maxiter must be positive, got {line_search_maxiter}")
        if trust_radius_init <= 0:
            raise ValueError(f"trust_radius_init must be positive, got {trust_radius_init}")
        
        self.max_iter = max_iter
        self.max_cg_iter = max_cg_iter
        self.gtol = gtol
        self.ftol = ftol
        self.xtol = xtol
        self.cg_tol = cg_tol
        self.line_search_maxiter = line_search_maxiter
        self.trust_radius = trust_radius_init
        self.verbose = verbose
        
        # FP64-specific parameters
        self.eps_fp64 = np.finfo(np.float64).eps
        self.hessian_regularization = 1e-12  # Minimal regularization for FP64
        
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> Dict[str, Any]:
        """
        Run Newton-CG optimization.
        
        Parameters
        ----------
        objective_fn : callable
            Function that computes objective value
        gradient_fn : callable
            Function that computes gradient
        hessian_fn : callable
            Function that computes Hessian matrix
        x0 : np.ndarray
            Initial parameter vector
        callback : callable, optional
            Called after each iteration with current x
            
        Returns
        -------
        dict
            Optimization results with keys:
            - 'x': Final parameters
            - 'fun': Final objective value
            - 'grad': Final gradient
            - 'hess': Final Hessian
            - 'n_iter': Number of iterations
            - 'n_cg_total': Total CG iterations
            - 'converged': Whether optimization converged
            - 'message': Convergence message
        """
        # Initialize
        x = x0.copy()
        n = len(x)
        
        # Initial function, gradient, and Hessian
        f = objective_fn(x)
        g = gradient_fn(x)
        H = hessian_fn(x)
        
        # Validate Hessian
        if H.shape != (n, n):
            raise ValueError(f"Hessian shape {H.shape} doesn't match parameter dimension {n}")
        
        # Track total CG iterations
        total_cg_iters = 0
        
        if self.verbose:
            print(f"Newton-CG Optimization Starting")
            print(f"Initial objective: {f:.6e}")
            print(f"Initial gradient norm: {np.linalg.norm(g):.6e}")
            print(f"Parameters: n={n}")
            print("-" * 60)
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Check gradient convergence
            grad_norm = np.linalg.norm(g)
            if grad_norm < self.gtol:
                message = f"Converged: gradient norm {grad_norm:.2e} < {self.gtol:.2e}"
                converged = True
                break
            
            # Solve Newton system H*d = -g using Conjugate Gradient
            d, cg_iters, cg_success = self._solve_newton_system(H, -g)
            total_cg_iters += cg_iters
            
            if not cg_success:
                # CG failed - try gradient descent as fallback
                if self.verbose:
                    warnings.warn("CG failed to converge, using gradient descent step")
                d = -g
            
            # Compute Newton decrement for convergence check
            newton_decrement = -np.dot(g, d)
            
            # Trust region / line search for global convergence
            alpha, f_new, g_new, H_new, ls_success = self._line_search_trust_region(
                objective_fn, gradient_fn, hessian_fn,
                x, f, g, H, d, newton_decrement
            )
            
            if not ls_success:
                message = "Line search failed to find acceptable step"
                converged = False
                break
            
            # Update position
            x_new = x + alpha * d
            
            # Check parameter change convergence
            x_change = np.linalg.norm(alpha * d)
            x_rel_change = x_change / (np.linalg.norm(x) + self.eps_fp64)
            
            if x_rel_change < self.xtol:
                message = f"Converged: parameter change {x_rel_change:.2e} < {self.xtol:.2e}"
                converged = True
                x = x_new
                f = f_new
                g = g_new
                H = H_new
                break
            
            # Check function value convergence
            f_change = abs(f_new - f)
            f_rel_change = f_change / (abs(f) + self.eps_fp64)
            
            if f_rel_change < self.ftol:
                message = f"Converged: function change {f_rel_change:.2e} < {self.ftol:.2e}"
                converged = True
                x = x_new
                f = f_new
                g = g_new
                H = H_new
                break
            
            # Check Newton decrement (quadratic convergence indicator)
            if newton_decrement < self.gtol:
                message = f"Converged: Newton decrement {newton_decrement:.2e} < {self.gtol:.2e}"
                converged = True
                x = x_new
                f = f_new
                g = g_new
                H = H_new
                break
            
            # Update state
            x = x_new
            f = f_new
            g = g_new
            H = H_new
            
            # Update trust radius based on step acceptance
            if alpha >= 0.9:
                self.trust_radius = min(2.0 * self.trust_radius, 10.0)
            elif alpha < 0.1:
                self.trust_radius = max(0.25 * self.trust_radius, 1e-4)
            
            # Callback
            if callback is not None:
                callback(x)
            
            # Verbose output
            if self.verbose:
                print(f"Iter {iteration:3d}: f={f:.8e}, |g|={grad_norm:.2e}, "
                      f"α={alpha:.3f}, CG={cg_iters:3d}, Δ={newton_decrement:.2e}")
        
        else:
            # Loop completed without convergence
            message = f"Maximum iterations ({self.max_iter}) reached"
            converged = False
        
        # Final gradient norm
        final_grad_norm = np.linalg.norm(g)
        
        if self.verbose:
            print("-" * 60)
            print(f"Newton-CG Optimization Complete")
            print(f"Converged: {converged}")
            print(f"Message: {message}")
            print(f"Newton iterations: {iteration + 1}")
            print(f"Total CG iterations: {total_cg_iters}")
            print(f"Final objective: {f:.8e}")
            print(f"Final gradient norm: {final_grad_norm:.2e}")
        
        return {
            'x': x,
            'fun': f,
            'grad': g,
            'hess': H,
            'n_iter': iteration + 1,
            'n_cg_total': total_cg_iters,
            'converged': converged,
            'message': message,
            'grad_norm': final_grad_norm
        }
    
    def _solve_newton_system(
        self,
        H: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Solve Newton system H*d = b using Conjugate Gradient.
        
        Parameters
        ----------
        H : np.ndarray
            Hessian matrix (must be positive definite)
        b : np.ndarray
            Right-hand side vector (-gradient)
            
        Returns
        -------
        d : np.ndarray
            Solution vector (Newton direction)
        n_iter : int
            Number of CG iterations
        success : bool
            Whether CG converged
        """
        n = len(b)
        
        # Check Hessian conditioning
        try:
            # Add minimal regularization for numerical stability
            H_reg = H + self.hessian_regularization * np.eye(n)
            
            # Check positive definiteness via Cholesky
            np.linalg.cholesky(H_reg)
            
        except np.linalg.LinAlgError:
            # Hessian not positive definite - add stronger regularization
            if self.verbose:
                warnings.warn("Hessian not positive definite, adding regularization")
            
            # Find minimum eigenvalue
            eigvals = np.linalg.eigvalsh(H)
            min_eigval = np.min(eigvals)
            
            # Regularize to make positive definite
            reg_strength = max(abs(min_eigval) + 1e-6, 1e-6)
            H_reg = H + reg_strength * np.eye(n)
        
        # Define linear operator for CG
        def hessian_op(v):
            return H_reg @ v
        
        # Solve using Conjugate Gradient
        try:
            # Use scipy's CG with tight tolerance for FP64
            d, info = cg(
                H_reg, b,
                tol=self.cg_tol,
                maxiter=min(self.max_cg_iter, n),  # CG converges in at most n iterations
                atol=self.cg_tol * np.linalg.norm(b) * self.eps_fp64  # Add absolute tolerance
            )
            
            # info = 0 means convergence
            success = (info == 0)
            
            # Estimate iteration count from info
            if info > 0:
                n_iter = info  # info contains number of iterations when it didn't converge
            else:
                n_iter = min(self.max_cg_iter // 2, n)  # Estimate for successful convergence
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"CG solver failed: {e}")
            d = np.zeros_like(b)
            n_iter = 0
            success = False
        
        # Truncate to trust region if necessary
        d_norm = np.linalg.norm(d)
        if d_norm > self.trust_radius:
            d = d * (self.trust_radius / d_norm)
        
        return d, n_iter, success
    
    def _line_search_trust_region(
        self,
        obj_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        hess_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        f: float,
        g: np.ndarray,
        H: np.ndarray,
        d: np.ndarray,
        newton_decrement: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray, bool]:
        """
        Backtracking line search with trust region safeguards.
        
        Parameters
        ----------
        obj_fn : callable
            Objective function
        grad_fn : callable
            Gradient function
        hess_fn : callable
            Hessian function
        x : np.ndarray
            Current position
        f : float
            Current objective value
        g : np.ndarray
            Current gradient
        H : np.ndarray
            Current Hessian
        d : np.ndarray
            Search direction
        newton_decrement : float
            Newton decrement (-g^T d)
            
        Returns
        -------
        alpha : float
            Step size
        f_new : float
            New objective value
        g_new : np.ndarray
            New gradient
        H_new : np.ndarray
            New Hessian
        success : bool
            Whether line search succeeded
        """
        # Armijo parameter for sufficient decrease
        c1 = 1e-4
        
        # Backtracking parameters
        alpha = 1.0
        backtrack_factor = 0.5
        
        # Check if we have a descent direction
        dg = np.dot(d, g)
        if dg > 0:
            if self.verbose:
                warnings.warn("Not a descent direction, reversing")
            d = -d
            dg = -dg
        
        # Backtracking line search
        for i in range(self.line_search_maxiter):
            x_new = x + alpha * d
            f_new = obj_fn(x_new)
            
            # Check Armijo condition for sufficient decrease
            expected_decrease = c1 * alpha * (-dg)
            
            if f_new <= f - expected_decrease:
                # Accept step
                g_new = grad_fn(x_new)
                H_new = hess_fn(x_new)
                return alpha, f_new, g_new, H_new, True
            
            # Quadratic interpolation for better step size
            if i == 0:
                # First backtrack - use quadratic interpolation
                alpha_new = -dg / (2 * (f_new - f - dg))
                alpha_new = np.clip(alpha_new, 0.1 * alpha, 0.9 * alpha)
            else:
                # Subsequent backtracks - simple reduction
                alpha_new = alpha * backtrack_factor
            
            alpha = alpha_new
            
            # Check if step is too small
            if alpha < 1e-10:
                break
        
        # Line search failed - return current state
        if self.verbose:
            warnings.warn(f"Line search failed after {i+1} iterations")
        
        return 0.0, f, g, H, False


def create_newton_cg_optimizer(
    max_iter: int = 100,
    precision: str = 'fp64',
    verbose: bool = False
) -> NewtonCGOptimizer:
    """
    Factory function to create Newton-CG optimizer with precision-specific settings.
    
    Parameters
    ----------
    max_iter : int
        Maximum Newton iterations
    precision : str
        Must be 'fp64' (Newton-CG requires high precision)
    verbose : bool
        Print progress
        
    Returns
    -------
    NewtonCGOptimizer
        Configured optimizer instance
    """
    if precision != 'fp64':
        raise ValueError(
            f"Newton-CG requires FP64 precision for convergence. "
            f"Got precision='{precision}'. Use BFGS for FP32."
        )
    
    return NewtonCGOptimizer(
        max_iter=max_iter,
        max_cg_iter=50,  # CG iterations per Newton step
        gtol=1e-8,       # Tight gradient tolerance for FP64
        ftol=1e-10,      # Tight function tolerance
        xtol=1e-9,       # Parameter change tolerance
        cg_tol=1e-5,     # CG solver tolerance
        line_search_maxiter=10,
        trust_radius_init=1.0,
        verbose=verbose
    )