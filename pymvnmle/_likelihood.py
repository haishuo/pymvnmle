"""
Core likelihood computation for PyMVNMLE
Ports the R evallf.c functionality to Python with GPU backend support
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from ._backends import BackendInterface


def compute_log_likelihood(params: np.ndarray, 
                          sorted_data: np.ndarray,
                          freq: np.ndarray, 
                          presence_absence: np.ndarray,
                          backend: BackendInterface) -> float:
    """
    Compute the negative log-likelihood exactly like R's evallf.c
    
    CRITICAL: This must match R's evallf.c computation exactly or we'll get
    wrong optimization results. The algorithm returns a value proportional 
    to TWICE the negative log-likelihood.
    """
    n_vars = sorted_data.shape[1]
    n_patterns = len(freq)
    
    # Extract mean parameters
    mu = params[:n_vars]
    
    # Reconstruct the full inverse Cholesky factor (del matrix)
    del_matrix = _reconstruct_del_matrix(params[n_vars:], n_vars, backend)
    
    # Check for numerical issues in del_matrix
    if not np.all(np.isfinite(del_matrix)):
        return 1e10
    
    # Initialize total value (R returns value proportional to 2*negative log-likelihood)
    total_value = 0.0
    data_counter = 0
    
    # Process each missingness pattern (exactly like R's evallf.c)
    for pattern_idx in range(n_patterns):
        pattern_freq = freq[pattern_idx]
        pattern_presence = presence_absence[pattern_idx]
        
        # Get indices of observed variables
        observed_indices = np.where(pattern_presence == 1)[0]
        n_observed = len(observed_indices)
        
        if n_observed == 0:
            continue  # Skip completely missing observations
        
        # Extract submatrix for observed variables
        sub_del = del_matrix[np.ix_(observed_indices, observed_indices)]
        
        # Apply Givens rotations (R's core algorithm)
        try:
            rotated_del = _apply_givens_rotations_r_style(sub_del, backend)
        except:
            return 1e10  # Return large value if rotations fail
        
        # Check for numerical issues
        if not np.all(np.isfinite(rotated_del)) or np.any(np.diag(rotated_del) <= 0):
            return 1e10
        
        # Compute log-determinant contribution (R: -2*freq*log_det)
        log_det_contrib = np.sum(np.log(np.diag(rotated_del)))
        total_value -= 2 * pattern_freq * log_det_contrib
        
        # Extract relevant mean parameters
        pattern_mu = mu[observed_indices]
        
        # Process observations in this pattern (R's quadratic form computation)
        pattern_data = sorted_data[data_counter:data_counter + pattern_freq, observed_indices]
        
        # Compute quadratic form exactly like R's evallf.c
        for obs_idx in range(pattern_freq):
            centered_obs = pattern_data[obs_idx] - pattern_mu
            
            # R's algorithm: prod[j] = Σₖ (data[k] - μ[k]) * subdel[k][j]
            # This is matrix multiplication: prod = rotated_del.T @ centered_obs
            # Then quadratic form = sum(prod²)
            try:
                prod = rotated_del.T @ centered_obs
                quad_form = np.sum(prod * prod)
                total_value += quad_form
            except:
                return 1e10
        
        data_counter += pattern_freq
    
    return total_value


def _apply_givens_rotations_r_style(matrix: np.ndarray, 
                                   backend: BackendInterface) -> np.ndarray:
    """
    Apply Givens rotations exactly like R's evallf.c
    
    This is the most critical part - must match R's algorithm exactly.
    """
    n = matrix.shape[0]
    result = matrix.copy()
    
    # R's algorithm: bottom-up, left-to-right
    for i in range(n-1, -1, -1):        # Start from bottom row
        for j in range(i):               # Left to diagonal
            # Zero out result[i, j] using Givens rotation
            a = result[i, j]
            b = result[i, j+1] if j+1 < n else 0.0
            
            # Skip if already small enough
            if abs(a) < 1e-12:
                result[i, j] = 0.0
                continue
            
            # Compute Givens rotation parameters
            r = np.sqrt(a*a + b*b)
            if r < 1e-12:
                continue
                
            c = a / r
            s = b / r
            
            # Apply rotation to entire matrix (R's approach)
            for k in range(n):
                if j+1 < n:
                    temp1 = s * result[k, j] - c * result[k, j+1]
                    temp2 = c * result[k, j] + s * result[k, j+1]
                    result[k, j] = temp1
                    result[k, j+1] = temp2
            
            # Ensure target element is zero
            result[i, j] = 0.0
    
    # Flip signs to ensure positive diagonal (R does this)
    for i in range(n):
        if result[i, i] < 0:
            for j in range(i+1):
                result[j, i] *= -1
    
    return result


def _reconstruct_del_matrix(del_params: np.ndarray, n_vars: int, 
                           backend: BackendInterface) -> np.ndarray:
    """
    Reconstruct the full inverse Cholesky factor from parameter vector.
    
    This mirrors the R make.del() function but uses GPU backends.
    
    Parameters
    ----------
    del_params : np.ndarray
        Parameters for inverse Cholesky factor:
        [log(del11), log(del22), ..., log(delpp), del12, del13, del23, ...]
    n_vars : int
        Number of variables
    backend : BackendInterface
        Computational backend
        
    Returns
    -------
    np.ndarray
        Upper triangular inverse Cholesky factor matrix
    """
    # Initialize with zeros
    del_matrix = np.zeros((n_vars, n_vars))
    
    # Set diagonal elements (exponentiated to ensure positivity)
    diagonal_params = del_params[:n_vars]
    # Clamp diagonal parameters to prevent overflow
    diagonal_params = np.clip(diagonal_params, -10, 10)
    for i in range(n_vars):
        del_matrix[i, i] = np.exp(diagonal_params[i])
    
    # Set upper triangular elements
    param_idx = n_vars
    for j in range(1, n_vars):  # Column index
        for i in range(j):      # Row index (i < j for upper triangular)
            if param_idx < len(del_params):
                # Clamp off-diagonal elements too
                del_matrix[i, j] = np.clip(del_params[param_idx], -100, 100)
                param_idx += 1
    
    return del_matrix


def _extract_submatrix(matrix: np.ndarray, indices: np.ndarray, 
                      backend: BackendInterface) -> np.ndarray:
    """Extract submatrix corresponding to observed variables."""
    # Use advanced indexing to extract submatrix
    submatrix = matrix[np.ix_(indices, indices)]
    return backend.asarray(submatrix)


def _apply_givens_rotations(matrix: np.ndarray, 
                           backend: BackendInterface) -> np.ndarray:
    """
    Apply Givens rotations to zero out below-diagonal elements.
    
    This is the core numerical algorithm from evallf.c that maintains
    the upper triangular structure after extracting submatrices.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be triangularized
    backend : BackendInterface
        Computational backend
        
    Returns
    -------
    np.ndarray
        Upper triangular matrix after Givens rotations
    """
    # Convert to backend format for computation
    gpu_matrix = backend.asarray(matrix.copy())
    n = matrix.shape[0]
    
    # Apply Givens rotations from bottom-up, left-to-right
    # This mirrors the evallf.c algorithm exactly
    for i in range(n-1, -1, -1):        # Start from bottom row, move up
        for j in range(i):               # From left to diagonal
            # Zero out matrix[i, j] using Givens rotation
            if abs(matrix[i, j]) > 1e-12:  # Only rotate if element is significant
                
                # Compute Givens rotation parameters
                a = matrix[i, j]
                b = matrix[i, j+1]
                r = np.sqrt(a*a + b*b)
                
                if r > 1e-12:  # Avoid division by very small numbers
                    c = a / r
                    s = b / r
                    
                    # Apply rotation to entire matrix
                    # This updates two columns simultaneously
                    for k in range(n):
                        temp1 = s * matrix[k, j] - c * matrix[k, j+1]
                        temp2 = c * matrix[k, j] + s * matrix[k, j+1]
                        matrix[k, j] = temp1
                        matrix[k, j+1] = temp2
                    
                    # Ensure the target element is exactly zero
                    matrix[i, j] = 0.0
    
    # Flip signs of columns where diagonal elements are negative
    # This ensures all diagonal elements are positive
    for i in range(n):
        if matrix[i, i] < 0:
            matrix[:i+1, i] *= -1
    
    return backend.to_cpu(matrix)


def _compute_log_determinant(triangular_matrix: np.ndarray, 
                            backend: BackendInterface) -> float:
    """
    Compute log-determinant of triangular matrix.
    
    For upper triangular matrix, log(det(A)) = sum(log(diag(A)))
    """
    diagonal = np.diag(triangular_matrix)
    
    # Check for non-positive diagonal elements
    if np.any(diagonal <= 0):
        return -np.inf  # Matrix is singular
    
    return np.sum(np.log(diagonal))


def _compute_quadratic_form(data: np.ndarray, mu: np.ndarray, 
                           del_matrix: np.ndarray, 
                           backend: BackendInterface) -> float:
    """
    Compute (y-mu)'Sigma^-1(y-mu) for all observations in a pattern.
    
    Uses the fact that Sigma^-1 = del'del where del is upper triangular.
    """
    # Center the data
    centered_data = data - mu[np.newaxis, :]  # Broadcasting
    
    # Convert to backend format
    gpu_centered = backend.asarray(centered_data)
    gpu_del = backend.asarray(del_matrix)
    
    total_quad_form = 0.0
    
    # For each observation in this pattern
    for i in range(data.shape[0]):
        # Solve del * x = (y_i - mu) where del is upper triangular
        # This gives us x such that del'del * x = del'(y_i - mu)
        centered_obs = centered_data[i, :]
        
        # Solve upper triangular system: del * x = centered_obs
        x = backend.solve_triangular(del_matrix, centered_obs, upper=True)
        
        # Quadratic form is x'x = (y-mu)'Sigma^-1(y-mu)
        quad_form = np.sum(x * x)
        total_quad_form += quad_form
    
    return total_quad_form


def compute_log_likelihood_and_gradient(params: np.ndarray, 
                                       sorted_data: np.ndarray,
                                       freq: np.ndarray, 
                                       presence_absence: np.ndarray,
                                       backend: BackendInterface) -> Tuple[float, np.ndarray]:
    """
    Compute negative log-likelihood and gradient using R's exact approach.
    
    R's nlm uses very specific finite difference parameters that work well
    for likelihood functions. We need to match this exactly.
    """
    # Compute function value
    f0 = compute_log_likelihood(params, sorted_data, freq, presence_absence, backend)
    
    n_params = len(params)
    gradient = np.zeros(n_params)
    
    # R's finite difference parameters (from nlm source code)
    # R uses: h = eps * max(|x|, 1) where eps is carefully chosen
    eps = 1.49011612e-08  # R's .Machine$double.eps^(1/3) ≈ sqrt(machine epsilon)
    
    for i in range(n_params):
        # R's step size calculation
        x_i = params[i] 
        h = eps * max(abs(x_i), 1.0)
        
        # Ensure step is not too small
        if h < 1e-12:
            h = 1e-12
            
        # Forward difference (R's default for nlm)
        params_h = params.copy()
        params_h[i] = x_i + h
        
        try:
            f_h = compute_log_likelihood(params_h, sorted_data, freq, presence_absence, backend)
            gradient[i] = (f_h - f0) / h
        except:
            # If forward fails, try backward
            params_h[i] = x_i - h
            try:
                f_h = compute_log_likelihood(params_h, sorted_data, freq, presence_absence, backend)
                gradient[i] = (f0 - f_h) / h
            except:
                gradient[i] = 0.0
    
    return f0, gradient


def _compute_quadratic_form_with_gradients(data: np.ndarray, mu: np.ndarray, 
                                         del_matrix: np.ndarray, observed_indices: np.ndarray,
                                         backend: BackendInterface) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute quadratic form and gradients with respect to mean and precision parameters.
    
    Returns
    -------
    quad_form : float
        Quadratic form value
    mu_gradient : np.ndarray
        Gradient with respect to mean parameters
    del_gradient : np.ndarray
        Gradient with respect to precision matrix elements
    """
    n_obs, n_vars = data.shape
    
    # Center the data
    centered_data = data - mu[np.newaxis, :]
    
    total_quad_form = 0.0
    mu_gradient = np.zeros(n_vars)
    del_gradient = np.zeros_like(del_matrix)
    
    # Process each observation
    for i in range(n_obs):
        centered_obs = centered_data[i, :]
        
        # Solve del * x = centered_obs
        x = backend.solve_triangular(del_matrix, centered_obs, upper=True)
        
        # Quadratic form contribution
        quad_form_i = np.sum(x * x)
        total_quad_form += quad_form_i
        
        # Gradient w.r.t. mean: ∂Q/∂μ = -2 * del^T * del * (y - μ) = -2 * del^T * x
        del_T_x = backend.solve_triangular(del_matrix.T, x, upper=False)
        mu_gradient -= 2 * del_T_x
        
        # Gradient w.r.t. del: ∂Q/∂del_ij = 2 * x_j * (y - μ)_i for upper triangular
        for j in range(n_vars):
            for i_del in range(j + 1):  # Upper triangular
                if i_del < j:  # Off-diagonal
                    del_gradient[i_del, j] += 2 * x[j] * centered_obs[i_del]
                else:  # Diagonal
                    del_gradient[i_del, j] += 2 * x[j] * centered_obs[i_del]
    
    return total_quad_form, mu_gradient, del_gradient


def _accumulate_del_gradient(full_gradient: np.ndarray, del_grad_contrib: np.ndarray,
                           observed_indices: np.ndarray, n_vars: int, 
                           pattern_freq: int, log_det_contrib: float):
    """
    Transform del gradient contributions to parameter gradient via chain rule.
    
    The parameterization uses:
    - log(del_ii) for diagonal elements
    - del_ij directly for off-diagonal elements
    
    So we need: ∂ℓ/∂θ = ∂ℓ/∂del * ∂del/∂θ
    """
    # Gradient w.r.t. log(diagonal elements): ∂ℓ/∂log(del_ii) = ∂ℓ/∂del_ii * del_ii
    for idx, i in enumerate(observed_indices):
        diag_grad_contrib = del_grad_contrib[idx, idx] * np.exp(full_gradient[n_vars + i])
        # Also add log-determinant contribution: ∂log|det|/∂log(del_ii) = 1
        diag_grad_contrib -= 2 * pattern_freq  # From log-determinant term
        full_gradient[n_vars + i] += diag_grad_contrib
    
    # Gradient w.r.t. off-diagonal elements: direct mapping
    param_idx = 2 * n_vars
    for j in range(1, n_vars):
        if j in observed_indices:
            for i in range(j):
                if i in observed_indices:
                    # Find positions in reduced matrix
                    i_reduced = np.where(observed_indices == i)[0][0]
                    j_reduced = np.where(observed_indices == j)[0][0]
                    full_gradient[param_idx] += del_grad_contrib[i_reduced, j_reduced]
                param_idx += 1
        else:
            param_idx += j  # Skip parameters for unobserved variables


def create_likelihood_function(sorted_data: np.ndarray, freq: np.ndarray, 
                              presence_absence: np.ndarray,
                              backend: BackendInterface):
    """
    Create a likelihood function for optimization (gradient-free methods).
    
    This mirrors the R getclf() function - returns a function that can be
    passed to scipy.optimize.minimize for methods that don't use gradients.
    
    Parameters
    ----------
    sorted_data : np.ndarray
        Data sorted by missingness pattern
    freq : np.ndarray
        Frequency of each missingness pattern
    presence_absence : np.ndarray
        Binary indicators for observed variables in each pattern
    backend : BackendInterface
        Computational backend to use
        
    Returns
    -------
    callable
        Function that takes parameter vector and returns negative log-likelihood
    """
    def likelihood_func(params):
        """Likelihood function for optimization."""
        try:
            return compute_log_likelihood(params, sorted_data, freq, 
                                        presence_absence, backend)
        except Exception as e:
            # Return large value if computation fails
            print(f"Likelihood computation failed: {e}")
            return 1e10
    
    return likelihood_func


def create_likelihood_function_with_gradient(sorted_data: np.ndarray, freq: np.ndarray, 
                                           presence_absence: np.ndarray,
                                           backend: BackendInterface):
    """
    Create a likelihood function that returns both value and gradient.
    
    This enables Newton-CG and other gradient-based optimizers.
    """
    def likelihood_func_with_grad(params):
        """Likelihood function with gradient for optimization."""
        try:
            return compute_log_likelihood_and_gradient(
                params, sorted_data, freq, presence_absence, backend
            )
        except Exception as e:
            print(f"Likelihood computation failed: {e}")
            # Return large value and zero gradient if computation fails
            return 1e10, np.zeros_like(params)
    
    return likelihood_func_with_grad