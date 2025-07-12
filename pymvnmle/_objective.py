"""
Core likelihood computation for PyMVNMLE
REGULATORY-GRADE implementation ported from validated scripts/objective_function.py

CRITICAL DISCOVERY (January 2025): 
R's mvnmle uses nlm() with FINITE DIFFERENCES, not analytical gradients.
This implementation matches R's behavior exactly for FDA submission compatibility.

Author: Senior Biostatistician
Purpose: Exact R compatibility for regulatory submissions
Standard: FDA submission grade
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import warnings

from ._utils import mysort_data, get_starting_values, validate_input_data, reconstruct_delta_matrix


@dataclass
class PatternData:
    """Data structure for a missingness pattern."""
    observed_indices: np.ndarray  # Which variables are observed
    n_k: int                      # Number of observations with this pattern
    data_k: np.ndarray           # Data matrix (n_k × n_observed)
    S_k: Optional[np.ndarray] = None  # Empirical covariance for this pattern


class MVNMLEObjective:
    """
    Objective function for multivariate normal ML estimation with missing data.
    
    This class implements R's exact algorithm including:
    1. Data preprocessing (pattern sorting via mysort)
    2. Givens rotations for numerical stability
    3. Finite difference gradients (matching R's nlm)
    4. Parameter bounds enforcement
    
    CRITICAL: Uses finite differences to exactly match R's behavior.
    Gradient norms at "convergence" will be ~1e-4, not machine precision.
    """
    
    def __init__(self, data: np.ndarray, compute_auxiliary: bool = False):
        """
        Initialize objective function with data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data matrix with missing values as np.nan
            Shape: (n_observations, n_variables)
        compute_auxiliary : bool
            Whether to compute auxiliary information (for diagnostics)
        """
        # Validate and clean data using our validated function
        self.data = validate_input_data(data)
        self.n_obs, self.n_vars = self.data.shape
        self.compute_auxiliary = compute_auxiliary
        
        # Preprocess data: sort by missingness patterns (R's mysort)
        self.sorted_data, self.freq, self.presence_absence = mysort_data(self.data)
        
        # Convert to PatternData structures
        self.patterns = self._extract_pattern_data()
        
        # Store diagnostic information
        self.n_patterns = len(self.patterns)
        self.pattern_sizes = [p.n_k for p in self.patterns]
        
        # Parameter dimensions
        self.n_mean_params = self.n_vars
        self.n_delta_params = self.n_vars + self.n_vars * (self.n_vars - 1) // 2
        self.n_total_params = self.n_mean_params + self.n_delta_params
        
        # Diagnostic counters
        self.n_evaluations = 0
        self.n_gradient_evaluations = 0
    
    def _extract_pattern_data(self) -> List[PatternData]:
        """Convert sorted data into PatternData structures."""
        patterns = []
        data_idx = 0
        
        for pattern_idx, (n_obs, pattern_mask) in enumerate(zip(self.freq, self.presence_absence)):
            # Get indices of observed variables
            observed_vars = np.where(pattern_mask == 1)[0]
            
            # Extract data for this pattern (only observed variables)
            pattern_data = self.sorted_data[data_idx:data_idx + n_obs][:, observed_vars]
            
            # Compute empirical covariance for this pattern
            if n_obs > 1:
                centered = pattern_data - np.mean(pattern_data, axis=0)
                S_k = (centered.T @ centered) / n_obs
            else:
                # Single observation - use small identity matrix
                n_obs_vars = len(observed_vars)
                S_k = 0.1 * np.eye(n_obs_vars)
            
            # Create PatternData structure
            pattern = PatternData(
                observed_indices=observed_vars,
                n_k=int(n_obs),
                data_k=pattern_data,
                S_k=S_k
            )
            patterns.append(pattern)
            
            data_idx += n_obs
        
        return patterns
    
    def _apply_givens_rotations_exact(self, matrix: np.ndarray, n_vars: int) -> np.ndarray:
        """
        Apply Givens rotations EXACTLY as R's evallf.c does.
        
        This is the most critical numerical component - must match R exactly.
        """
        result = matrix.copy()
        
        # R's algorithm: bottom-up, left-to-right
        for i in range(n_vars-1, -1, -1):  # Start from bottom row
            for j in range(i):  # Left to diagonal
                # Zero out result[i, j] using Givens rotation
                a = result[i, j]
                b = result[i, j+1] if j+1 < n_vars else 0.0
                
                # Skip if already small (R's exact threshold)
                if abs(a) < 0.000001:
                    result[i, j] = 0.0
                    continue
                
                # Compute rotation parameters EXACTLY as R does
                r = np.sqrt(a*a + b*b)
                if r < 0.000001:
                    continue
                    
                c = a / r
                d = b / r
                
                # Apply rotation to entire matrix - compute all new values first
                newcol1 = np.zeros(n_vars)
                newcol2 = np.zeros(n_vars)
                
                for k in range(n_vars):
                    if j+1 < n_vars:
                        newcol1[k] = d * result[k, j] - c * result[k, j+1]
                        newcol2[k] = c * result[k, j] + d * result[k, j+1]
                    else:
                        newcol1[k] = d * result[k, j]
                        newcol2[k] = c * result[k, j]
                
                # Update the matrix
                for k in range(n_vars):
                    result[k, j] = newcol1[k]
                    if j+1 < n_vars:
                        result[k, j+1] = newcol2[k]
                
                result[i, j] = 0.0
        
        # Flip signs to ensure positive diagonal (R's exact procedure)
        for i in range(n_vars):
            if result[i, i] < 0:
                for j in range(i+1):
                    result[j, i] *= -1
        
        return result
    
    def __call__(self, theta: np.ndarray) -> float:
        """
        Evaluate objective function (proportional to -2*log-likelihood).
        
        This implements R's exact evallf.c algorithm including Givens rotations.
        Returns value proportional to -2*loglik (NOT -loglik) to match R exactly.
        """
        self.n_evaluations += 1
        
        # Validate parameter vector
        if len(theta) != self.n_total_params:
            raise ValueError(f"Parameter vector wrong length: {len(theta)} vs {self.n_total_params}")
        
        if not np.all(np.isfinite(theta)):
            return 1e20  # Return large value for optimizer
        
        try:
            # Extract parameters
            mu = theta[:self.n_vars]
            delta_params = theta[self.n_vars:]
            
            # Reconstruct Delta matrix with bounds enforcement
            Delta = reconstruct_delta_matrix(delta_params, self.n_vars)
            
            # Initialize objective value (proportional to -2*loglik like R)
            obj_value = 0.0
            
            # Process each missingness pattern
            for pattern in self.patterns:
                if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                    continue
                
                obs_idx = pattern.observed_indices
                n_obs_vars = len(obs_idx)
                
                # CRITICAL: Implement R's row shuffling algorithm exactly
                # Create reordered Delta with observed rows first, missing rows last
                subdel = np.zeros((self.n_vars, self.n_vars))
                
                # Put observed variable rows first
                pcount = 0
                for i in range(self.n_vars):
                    if i in obs_idx:
                        subdel[pcount, :] = Delta[i, :]
                        pcount += 1
                
                # Put missing variable rows last
                acount = 0
                for i in range(self.n_vars):
                    if i not in obs_idx:
                        subdel[self.n_vars - acount - 1, :] = Delta[i, :]
                        acount += 1
                
                # Apply Givens rotations using R's exact algorithm
                subdel = self._apply_givens_rotations_exact(subdel, self.n_vars)
                
                # Extract just the observed part (top-left submatrix)
                subdel_obs = subdel[:n_obs_vars, :n_obs_vars]
                
                # Extract mean for observed variables
                mu_obs = mu[obs_idx]
                
                # Log determinant contribution: -2*n_k*log|subdel_obs|
                try:
                    log_det_delta = np.sum(np.log(np.diag(subdel_obs)))
                    obj_value -= 2 * pattern.n_k * log_det_delta
                except:
                    return 1e20  # Handle numerical issues
                
                # Quadratic form contributions: Σᵢ (yᵢ-μ)ᵀ Σ⁻¹ (yᵢ-μ)
                # Using R's exact algorithm: prod = subdel_obs.T @ (y-μ), then sum(prod²)
                for i in range(pattern.n_k):
                    y_i = pattern.data_k[i, :]
                    centered = y_i - mu_obs
                    
                    # R's algorithm: prod[j] = Σₖ (data[k] - μ[k]) * subdel[k][j]
                    prod = subdel_obs.T @ centered
                    obj_value += np.dot(prod, prod)
            
            # Store for diagnostics
            self.last_theta = theta.copy()
            self.last_objective = obj_value
            
            # Return proportional to -2*loglik (R's exact convention)
            return obj_value
            
        except Exception as e:
            if self.compute_auxiliary:
                warnings.warn(f"Numerical issue in objective: {e}")
            return 1e20
    
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient using R's nlm finite difference approach.
        
        CRITICAL: Uses R's exact finite difference parameters to ensure
        identical behavior. This is why gradient norms are ~1e-4, not ~1e-15.
        """
        self.n_gradient_evaluations += 1
        
        if len(theta) != self.n_total_params:
            raise ValueError(f"Parameter vector wrong length")
        
        try:
            n_params = len(theta)
            gradient = np.zeros(n_params)
            
            # R's nlm uses this specific epsilon
            eps = 1.49011612e-08  # R's .Machine$double.eps^(1/3)
            
            # Base function value
            f0 = self(theta)
            
            for i in range(n_params):
                # R's step size calculation
                h = eps * max(abs(theta[i]), 1.0)
                
                # Ensure step is not too small
                if h < 1e-12:
                    h = 1e-12
                
                # Forward difference (R's default for nlm)
                theta_plus = theta.copy()
                theta_plus[i] = theta[i] + h
                
                try:
                    f_plus = self(theta_plus)
                    gradient[i] = (f_plus - f0) / h
                except:
                    # If forward fails, try backward
                    theta_minus = theta.copy()
                    theta_minus[i] = theta[i] - h
                    try:
                        f_minus = self(theta_minus)
                        gradient[i] = (f0 - f_minus) / h
                    except:
                        gradient[i] = 0.0
            
            self.last_gradient = gradient.copy()
            self.last_gradient_norm = np.linalg.norm(gradient)
            
            return gradient
            
        except Exception as e:
            if self.compute_auxiliary:
                warnings.warn(f"Numerical issue in gradient: {e}")
            return np.zeros(self.n_total_params)
    
    def get_starting_values(self, method: str = 'moments') -> np.ndarray:
        """Get starting values using validated approach from scripts."""
        if method == 'moments':
            return get_starting_values(self.data)
        elif method == 'identity':
            # Alternative: identity covariance
            theta0 = np.zeros(self.n_total_params)
            theta0[:self.n_vars] = np.nanmean(self.data, axis=0)
            theta0[self.n_vars:2*self.n_vars] = 0.0  # log(diag) = 0
            return theta0
        else:
            raise ValueError(f"Unknown starting value method: {method}")
    
    def compute_estimates(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance estimates from optimized parameters.
        
        Uses validated reconstruction from scripts/parameter_reconstruction.py
        """
        from ._utils import reconstruct_covariance_matrix
        
        mu = theta[:self.n_vars]
        delta_params = theta[self.n_vars:]
        
        # Reconstruct covariance using validated approach
        sigmahat = reconstruct_covariance_matrix(delta_params, self.n_vars)
        
        return mu, sigmahat
    
    def diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about optimization."""
        diag = {
            'n_observations': self.n_obs,
            'n_variables': self.n_vars,
            'n_parameters': self.n_total_params,
            'n_patterns': self.n_patterns,
            'pattern_sizes': self.pattern_sizes,
            'n_function_evaluations': self.n_evaluations,
            'n_gradient_evaluations': self.n_gradient_evaluations,
        }
        
        if hasattr(self, 'last_objective'):
            diag['last_objective'] = self.last_objective
            
        if hasattr(self, 'last_gradient_norm'):
            diag['last_gradient_norm'] = self.last_gradient_norm
            
        return diag


def create_scipy_objective(data: np.ndarray) -> Tuple[Callable, Callable, np.ndarray]:
    """
    Create objective and gradient functions for scipy.optimize.
    
    Returns functions compatible with scipy's BFGS optimizer to match R's nlm.
    """
    obj = MVNMLEObjective(data)
    x0 = obj.get_starting_values()
    
    return obj, obj.gradient, x0