"""
BFGS optimizer for FP32 backends.

This module implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) 
quasi-Newton optimization algorithm, optimized for FP32 precision
and consumer GPU hardware. Uses analytical gradients when available.

Author: Statistical Software Engineer
Date: January 2025
License: MIT
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np


class BFGSOptimizer:
    """
    BFGS optimizer tailored for FP32 precision.
    
    This implementation is designed for consumer GPUs (RTX series, 
    Apple Metal) where FP32 is optimal. Uses analytical gradients
    via autodiff when available, with appropriate numerical 
    safeguards for single precision.
    
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    gtol : float
        Gradient norm tolerance for convergence
    ftol : float
        Function value tolerance for convergence
    step_size_init : float
        Initial step size for line search
    armijo_c1 : float
        Armijo condition parameter
    wolfe_c2 : float
        Wolfe condition parameter for curvature
    max_line_search : int
        Maximum line search iterations
    verbose : bool
        Whether to print optimization progress
    """
    
    def __init__(
        self,
        max_iter: int,
        gtol: float,
        ftol: float,
        step_size_init: float,
        armijo_c1: float,
        wolfe_c2: float,
        max_line_search: int,
        verbose: bool
    ):
        """
        Initialize BFGS optimizer with explicit parameters.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of BFGS iterations
        gtol : float
            Gradient norm tolerance (FP32: typically 1e-5)
        ftol : float
            Function value change tolerance
        step_size_init : float
            Initial step size for line search (typically 1.0)
        armijo_c1 : float
            Armijo condition parameter (typically 1e-4)
        wolfe_c2 : float
            Wolfe curvature condition (typically 0.9)
        max_line_search : int
            Maximum line search iterations (typically 20)
        verbose : bool
            Print optimization progress
        """
        # Validate inputs
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if gtol <= 0:
            raise ValueError(f"gtol must be positive, got {gtol}")
        if ftol <= 0:
            raise ValueError(f"ftol must be positive, got {ftol}")
        if step_size_init <= 0:
            raise ValueError(f"step_size_init must be positive, got {step_size_init}")
        if not (0 < armijo_c1 < 1):
            raise ValueError(f"armijo_c1 must be in (0, 1), got {armijo_c1}")
        if not (0 < wolfe_c2 < 1):
            raise ValueError(f"wolfe_c2 must be in (0, 1), got {wolfe_c2}")
        if not (armijo_c1 < wolfe_c2):
            raise ValueError(f"wolfe_c2 must be greater than armijo_c1, got c1={armijo_c1}, c2={wolfe_c2}")
        if max_line_search <= 0:
            raise ValueError(f"max_line_search must be positive, got {max_line_search}")
        
        self.max_iter = max_iter
        self.gtol = gtol
        self.ftol = ftol
        self.step_size_init = step_size_init
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.max_line_search = max_line_search
        self.verbose = verbose
        
        # FP32-specific parameters
        self.eps_fp32 = np.finfo(np.float32).eps
        self.regularization = 1e-6  # For Hessian approximation stability
        
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> Dict[str, Any]:
        """
        Run BFGS optimization.
        
        Parameters
        ----------
        objective_fn : callable
            Function that computes objective value
        gradient_fn : callable
            Function that computes gradient
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
            - 'n_iter': Number of iterations
            - 'converged': Whether optimization converged
            - 'message': Convergence message
        """
        # Initialize
        x = x0.copy()
        n = len(x)
        
        # Initial function and gradient
        f = objective_fn(x)
        g = gradient_fn(x)
        
        # Initialize inverse Hessian approximation (identity matrix)
        H = np.eye(n, dtype=np.float64)  # Use FP64 for accumulation
        
        # Optimization history
        f_prev = f
        
        if self.verbose:
            print(f"BFGS Optimization Starting")
            print(f"Initial objective: {f:.6e}")
            print(f"Initial gradient norm: {np.linalg.norm(g):.6e}")
            print("-" * 50)
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Check gradient convergence
            grad_norm = np.linalg.norm(g)
            if grad_norm < self.gtol:
                message = f"Converged: gradient norm {grad_norm:.2e} < {self.gtol:.2e}"
                converged = True
                break
            
            # Compute search direction: d = -H @ g
            d = -H @ g
            
            # Line search
            alpha, f_new, g_new, ls_iters = self._line_search(
                objective_fn, gradient_fn, x, f, g, d
            )
            
            # Check line search failure
            if alpha is None:
                message = "Line search failed to find acceptable step"
                converged = False
                break
            
            # Update position
            s = alpha * d  # Step
            x_new = x + s
            
            # Compute gradient difference
            y = g_new - g
            
            # Update inverse Hessian approximation using BFGS formula
            H = self._update_inverse_hessian(H, s, y)
            
            # Check function value convergence
            f_change = abs(f_new - f)
            rel_change = f_change / (abs(f) + self.eps_fp32)
            
            if rel_change < self.ftol:
                message = f"Converged: relative change {rel_change:.2e} < {self.ftol:.2e}"
                converged = True
                x = x_new
                f = f_new
                g = g_new
                break
            
            # Update state
            x = x_new
            f = f_new
            g = g_new
            
            # Callback
            if callback is not None:
                callback(x)
            
            # Verbose output
            if self.verbose and (iteration % 10 == 0 or iteration < 10):
                print(f"Iter {iteration:4d}: f={f:.6e}, |g|={grad_norm:.2e}, "
                      f"α={alpha:.2e}, ls={ls_iters}")
            
            f_prev = f
        
        else:
            # Loop completed without convergence
            message = f"Maximum iterations ({self.max_iter}) reached"
            converged = False
        
        # Final gradient norm
        final_grad_norm = np.linalg.norm(g)
        
        if self.verbose:
            print("-" * 50)
            print(f"BFGS Optimization Complete")
            print(f"Converged: {converged}")
            print(f"Message: {message}")
            print(f"Iterations: {iteration + 1}")
            print(f"Final objective: {f:.6e}")
            print(f"Final gradient norm: {final_grad_norm:.6e}")
        
        return {
            'x': x,
            'fun': f,
            'grad': g,
            'n_iter': iteration + 1,
            'converged': converged,
            'message': message,
            'grad_norm': final_grad_norm
        }
    
    def _line_search(
        self,
        obj_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        f: float,
        g: np.ndarray,
        d: np.ndarray
    ) -> Tuple[Optional[float], float, np.ndarray, int]:
        """
        Backtracking line search with Armijo-Wolfe conditions.
        
        Parameters
        ----------
        obj_fn : callable
            Objective function
        grad_fn : callable
            Gradient function
        x : np.ndarray
            Current position
        f : float
            Current objective value
        g : np.ndarray
            Current gradient
        d : np.ndarray
            Search direction
            
        Returns
        -------
        alpha : float or None
            Step size (None if failed)
        f_new : float
            New objective value
        g_new : np.ndarray
            New gradient
        n_iters : int
            Number of line search iterations
        """
        alpha = self.step_size_init
        backtrack_factor = 0.5
        expand_factor = 2.0
        
        # Directional derivative
        dg = np.dot(d, g)
        
        if dg >= 0:
            warnings.warn("Search direction is not a descent direction")
            return None, f, g, 0
        
        # Try to bracket the minimum first
        alpha_prev = 0
        f_prev = f
        
        for i in range(self.max_line_search):
            x_new = x + alpha * d
            f_new = obj_fn(x_new)
            
            # Check Armijo condition
            if f_new > f + self.armijo_c1 * alpha * dg:
                # Overshot - backtrack
                alpha = (alpha_prev + alpha) / 2
            else:
                # Armijo satisfied, check Wolfe curvature condition
                g_new = grad_fn(x_new)
                dg_new = np.dot(d, g_new)
                
                if dg_new < self.wolfe_c2 * dg:
                    # Curvature condition not met - need larger step
                    alpha_prev = alpha
                    f_prev = f_new
                    alpha = min(alpha * expand_factor, 1.0)
                else:
                    # Both conditions satisfied
                    return alpha, f_new, g_new, i + 1
        
        # Fallback: just use Armijo condition
        for i in range(self.max_line_search):
            x_new = x + alpha * d
            f_new = obj_fn(x_new)
            
            if f_new <= f + self.armijo_c1 * alpha * dg:
                g_new = grad_fn(x_new)
                return alpha, f_new, g_new, self.max_line_search + i + 1
            
            alpha *= backtrack_factor
        
        # Line search failed
        return None, f, g, 2 * self.max_line_search
    
    def _update_inverse_hessian(
        self,
        H: np.ndarray,
        s: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Update inverse Hessian approximation using BFGS formula.
        
        Uses the Sherman-Morrison-Woodbury formula for numerical
        stability in FP32 environments.
        
        Parameters
        ----------
        H : np.ndarray
            Current inverse Hessian approximation
        s : np.ndarray
            Step vector (x_new - x_old)
        y : np.ndarray
            Gradient difference (g_new - g_old)
            
        Returns
        -------
        np.ndarray
            Updated inverse Hessian approximation
        """
        # Curvature condition check
        sy = np.dot(s, y)
        
        # Skip update if curvature condition violated (for stability)
        if sy < self.regularization * np.dot(s, s):
            if self.verbose:
                warnings.warn("Skipping BFGS update: insufficient curvature")
            return H
        
        # BFGS update formula with damping for FP32 stability
        # H_new = (I - rho * s * y^T) * H * (I - rho * y * s^T) + rho * s * s^T
        # where rho = 1 / (y^T * s)
        
        rho = 1.0 / sy
        
        # For numerical stability in FP32, use two-step update
        # Step 1: V = I - rho * s * y^T
        V = np.eye(len(s)) - rho * np.outer(s, y)
        
        # Step 2: H_new = V^T * H * V + rho * s * s^T
        H_new = V.T @ H @ V + rho * np.outer(s, s)
        
        # Ensure symmetry (important for FP32)
        H_new = 0.5 * (H_new + H_new.T)
        
        # Add small regularization for FP32 stability
        H_new += self.regularization * np.eye(len(s))
        
        return H_new


def create_bfgs_optimizer(
    max_iter: int = 1000,
    precision: str = 'fp32',
    verbose: bool = False
) -> BFGSOptimizer:
    """
    Factory function to create BFGS optimizer with precision-specific settings.
    
    Parameters
    ----------
    max_iter : int
        Maximum iterations
    precision : str
        Either 'fp32' or 'fp64' (affects tolerances)
    verbose : bool
        Print progress
        
    Returns
    -------
    BFGSOptimizer
        Configured optimizer instance
    """
    if precision == 'fp32':
        # Looser tolerances for FP32
        gtol = 1e-5
        ftol = 1e-6
    elif precision == 'fp64':
        # Tighter tolerances for FP64
        gtol = 1e-8
        ftol = 1e-10
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    return BFGSOptimizer(
        max_iter=max_iter,
        gtol=gtol,
        ftol=ftol,
        step_size_init=1.0,
        armijo_c1=1e-4,
        wolfe_c2=0.9,
        max_line_search=20,
        verbose=verbose
    )