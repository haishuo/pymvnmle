"""
Core objective function for PyMVNMLE v2.0
Missing data MLE computation - mathematically correct R port

DESIGN PRINCIPLE: Mathematical correctness above all else
This module implements R's mvnmle algorithm with EXACT mathematical fidelity:
1. mysort data preprocessing (critical for R compatibility)
2. Givens rotations for numerical stability (from R's evallf.c)
3. Exact parameter ordering and reconstruction
4. Pattern-wise likelihood computation matching R exactly

DO NOT "simplify" any mathematical algorithms - they exist for numerical reasons.
Remove only bloat (diagnostics, verbose output), preserve ALL mathematics.
"""

import numpy as np
from typing import Union, Tuple, List
from dataclasses import dataclass


@dataclass
class PatternData:
    """R-compatible pattern data structure."""
    observed_indices: np.ndarray  # Which variables are observed (R's O_k)
    n_k: int                     # Number of cases with this pattern (R's freq)
    data_k: np.ndarray          # Data subset for this pattern (n_k × |O_k|)
    pattern_start: int          # Start index in sorted data
    pattern_end: int            # End index in sorted data


class MVNMLEObjective:
    """
    Objective function implementing R's mvnmle algorithm exactly.
    
    Critical R components preserved:
    1. mysort preprocessing - sorts observations by missingness pattern
    2. Givens rotations - numerical stabilization from R's evallf.c
    3. Exact parameter structure - R's column-major ordering
    4. Pattern-wise computation - efficiency and numerical stability
    
    This works with both NumPy arrays (CPU finite differences) and 
    PyTorch tensors (GPU autodiff) while preserving R mathematical fidelity.
    """
    
    def __init__(self, data: np.ndarray, backend=None):
        """
        Initialize with R's exact preprocessing algorithm.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Data matrix with missing values as np.nan
        backend : BackendInterface, optional
            Computational backend (unused - backends handle their own tensors)
        """
        # Store original data
        self.original_data = np.asarray(data, dtype=np.float64)
        self.n_obs, self.n_vars = self.original_data.shape
        
        # CRITICAL: Apply R's mysort preprocessing
        self.sorted_data, self.pattern_frequencies, self.presence_absence = self._mysort_data(
            self.original_data
        )
        
        # Extract pattern information in R-compatible format
        self.patterns = self._extract_pattern_data()
        self.n_patterns = len(self.patterns)
        
        # Parameter structure (R's exact layout)
        # θ = [μ₁, ..., μₚ, log(δ₁₁), ..., log(δₚₚ), δ₁₂, δ₁₃, δ₂₃, ...]
        self.n_params = self.n_vars + (self.n_vars * (self.n_vars + 1)) // 2
    
    def _mysort_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sort data by missingness patterns - EXACT port of R's mysort.
        
        This is CRITICAL for R compatibility and computational efficiency.
        R groups observations with identical missingness patterns together.
        
        Returns
        -------
        sorted_data : np.ndarray
            Data matrix with rows reordered by missingness pattern
        freq : np.ndarray  
            Number of observations in each pattern block
        presence_absence : np.ndarray
            Binary matrix (n_patterns × n_vars): 1=observed, 0=missing
        """
        n_obs, n_vars = data.shape
        
        # Step 1: Create binary representation (R: binrep <- ifelse(is.na(x), 0, 1))
        is_observed = (~np.isnan(data)).astype(int)
        
        # Step 2: Convert to decimal codes for sorting (R's exact method)
        # R: powers <- as.integer(2^((nvars-1):0))
        # R: decrep <- binrep %*% powers
        powers = 2 ** np.arange(n_vars - 1, -1, -1)
        pattern_codes = is_observed @ powers
        
        # Step 3: Sort by pattern codes (R: sorted <- x[order(decrep), ])
        sort_indices = np.argsort(pattern_codes)
        sorted_data = data[sort_indices]
        sorted_patterns = is_observed[sort_indices]
        sorted_codes = pattern_codes[sort_indices]
        
        # Step 4: Extract unique patterns and frequencies (R: freq = table(decrep))
        unique_codes, freq = np.unique(sorted_codes, return_counts=True)
        
        # Step 5: Build presence_absence matrix (pattern × variable indicator)
        presence_absence = []
        current_code = -1
        for i, code in enumerate(sorted_codes):
            if code != current_code:
                presence_absence.append(sorted_patterns[i])
                current_code = code
        
        presence_absence = np.array(presence_absence)
        
        return sorted_data, freq, presence_absence
    
    def _extract_pattern_data(self) -> List[PatternData]:
        """
        Extract pattern data structures from sorted data (R-compatible).
        
        Creates PatternData objects for each unique missingness pattern,
        maintaining R's data organization for efficient computation.
        """
        patterns = []
        data_start = 0
        
        for pattern_idx in range(len(self.pattern_frequencies)):
            n_k = self.pattern_frequencies[pattern_idx]
            data_end = data_start + n_k
            
            # Get observed variable indices for this pattern
            pattern_vec = self.presence_absence[pattern_idx]
            observed_indices = np.where(pattern_vec == 1)[0]
            
            # Skip patterns with no observed variables
            if len(observed_indices) == 0:
                data_start = data_end
                continue
            
            # Extract data subset (only observed variables)
            data_k = self.sorted_data[data_start:data_end][:, observed_indices]
            
            patterns.append(PatternData(
                observed_indices=observed_indices,
                n_k=n_k,
                data_k=data_k,
                pattern_start=data_start,
                pattern_end=data_end
            ))
            
            data_start = data_end
        
        return patterns
    
    def get_initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter estimates using R-compatible method.
        
        Uses pairwise complete observations to compute starting covariance,
        exactly matching R's getstartvals() function and V1.5 behavior.
        
        Returns
        -------
        np.ndarray
            Parameter vector in R's format:
            [μ₁, ..., μₚ, log(δ₁₁), ..., log(δₚₚ), δ₁₂, δ₁₃, δ₂₃, ...]
        """
        # Step 1: Compute sample means (unchanged)
        mu_start = np.nanmean(self.original_data, axis=0)
        
        # Step 2: Compute pairwise complete sample covariance (CRITICAL FIX)
        cov_sample = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Find pairwise complete observations
                mask = ~(np.isnan(self.original_data[:, i]) | np.isnan(self.original_data[:, j]))
                n_complete = np.sum(mask)
                
                if n_complete > 1:
                    if i == j:
                        # Variance
                        cov_sample[i, i] = np.var(self.original_data[mask, i], ddof=1)
                    else:
                        # Covariance - THIS CAPTURES THE CORRELATIONS!
                        cov_ij = np.cov(self.original_data[mask, i], 
                                    self.original_data[mask, j], ddof=1)[0, 1]
                        cov_sample[i, j] = cov_ij
                        cov_sample[j, i] = cov_ij
                else:
                    # No complete pairs, use defaults
                    if i == j:
                        cov_sample[i, i] = 1.0
                    else:
                        cov_sample[i, j] = 0.0
                        cov_sample[j, i] = 0.0
        
        # Step 3: Regularize to ensure positive definiteness (R's approach)
        eigenvals, eigenvecs = np.linalg.eigh(cov_sample)
        
        # Find smallest positive eigenvalue
        pos_eigenvals = eigenvals[eigenvals > 0]
        if len(pos_eigenvals) > 0:
            min_pos = np.min(pos_eigenvals)
        else:
            min_pos = 1.0
        
        # Regularize: any eigenvalue < eps * min_pos becomes eps * min_pos
        eps = 1e-3  # R's default
        threshold = eps * min_pos
        regularized_eigenvals = np.maximum(eigenvals, threshold)
        
        # Reconstruct regularized covariance
        cov_regularized = eigenvecs @ np.diag(regularized_eigenvals) @ eigenvecs.T
        
        # Step 4: Get Cholesky factor (R uses upper triangular)
        try:
            L = np.linalg.cholesky(cov_regularized)
            chol_upper = L.T
        except np.linalg.LinAlgError:
            # Fallback: add more regularization
            cov_regularized += np.eye(self.n_vars) * 1e-6
            L = np.linalg.cholesky(cov_regularized)
            chol_upper = L.T
        
        # Step 5: Compute inverse Cholesky factor (Delta)
        try:
            Delta_start = np.linalg.solve(chol_upper, np.eye(self.n_vars))
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            Delta_start = np.linalg.pinv(chol_upper)
        
        # Ensure positive diagonal (R's sign adjustment)
        for i in range(self.n_vars):
            if Delta_start[i, i] < 0:
                Delta_start[i, :] *= -1
        
        # Step 6: Pack into parameter vector (R's exact ordering)
        theta = np.zeros(self.n_params)
        
        # Mean parameters
        theta[:self.n_vars] = mu_start
        
        # Log diagonal of Delta
        theta[self.n_vars:2*self.n_vars] = np.log(np.diag(Delta_start))
        
        # Off-diagonal elements of Delta (R's ordering: by column)
        param_idx = 2 * self.n_vars
        for j in range(1, self.n_vars):
            for i in range(j):
                theta[param_idx] = Delta_start[i, j]
                param_idx += 1
        
        return theta
    
    def _reconstruct_delta_matrix(self, theta: Union[np.ndarray, "torch.Tensor"], 
                                 array_lib=np) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Reconstruct Δ matrix from parameter vector (R's exact algorithm).
        
        Critical: uses R's exact parameter ordering and bounds.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector in R's format
        array_lib : module
            Array library (np for NumPy, torch for PyTorch)
            
        Returns
        -------
        Delta : array
            Upper triangular Δ matrix
        """
        # Create zero matrix
        if array_lib.__name__ == 'torch':
            Delta = array_lib.zeros((self.n_vars, self.n_vars), 
                                  dtype=theta.dtype, device=theta.device)
            diag_indices = array_lib.arange(self.n_vars)
        else:
            Delta = array_lib.zeros((self.n_vars, self.n_vars))
            diag_indices = array_lib.arange(self.n_vars)
        
        # Extract and set diagonal elements (from log parameters)
        log_diag = theta[self.n_vars:2*self.n_vars]
        # Apply R's bounds to prevent overflow
        if array_lib.__name__ == 'torch':
            log_diag_bounded = array_lib.clamp(log_diag, -10.0, 10.0)
        else:
            log_diag_bounded = array_lib.clip(log_diag, -10.0, 10.0)
        
        Delta[diag_indices, diag_indices] = array_lib.exp(log_diag_bounded)
        
        # Fill upper triangle (R's exact column-major ordering)
        idx = 2 * self.n_vars
        for j in range(self.n_vars):      # Column
            for i in range(j):            # Row within column
                value = theta[idx]
                # Apply R's bounds for off-diagonals
                if array_lib.__name__ == 'torch':
                    value_bounded = array_lib.clamp(value, -100.0, 100.0)
                else:
                    value_bounded = array_lib.clip(value, -100.0, 100.0)
                
                Delta[i, j] = value_bounded
                idx += 1
        
        return Delta
    
    def _apply_givens_rotations(self, matrix: Union[np.ndarray, "torch.Tensor"], 
                               array_lib=np) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Apply Givens rotations for numerical stability (R's evallf.c algorithm).
        
        This is CRITICAL for numerical stability and R compatibility.
        Implements the exact algorithm from R's C code.
        
        Parameters
        ----------
        matrix : array
            Matrix to stabilize
        array_lib : module
            Array library (np or torch)
            
        Returns
        -------
        array
            Numerically stabilized matrix
        """
        result = matrix.clone() if hasattr(matrix, 'clone') else matrix.copy()
        
        # R's algorithm: bottom-up, left-to-right elimination
        for i in range(self.n_vars-1, -1, -1):    # Start from bottom row  
            for j in range(i):                     # Left to diagonal
                
                # Get elements to eliminate
                a = result[i, j]
                b = result[i, j+1] if j+1 < self.n_vars else 0.0
                
                # Skip if already small (R's exact threshold)
                if array_lib.__name__ == 'torch':
                    if torch.abs(a) < 1e-6:
                        result[i, j] = 0.0
                        continue
                else:
                    if abs(a) < 1e-6:
                        result[i, j] = 0.0
                        continue
                
                # Compute rotation magnitude
                if array_lib.__name__ == 'torch':
                    r = torch.sqrt(a*a + b*b)
                    if r < 1e-6:
                        continue
                else:
                    r = np.sqrt(a*a + b*b)  
                    if r < 1e-6:
                        continue
                
                # Rotation parameters (R's exact formulas)
                c = a / r  # cos(θ)
                s = b / r  # sin(θ)
                
                # Apply rotation to entire matrix (R's exact procedure)
                for k in range(self.n_vars):
                    old_kj = result[k, j]
                    old_kj1 = result[k, j+1] if j+1 < self.n_vars else 0.0
                    
                    # R's rotation formulas
                    result[k, j] = s * old_kj - c * old_kj1
                    if j+1 < self.n_vars:
                        result[k, j+1] = c * old_kj + s * old_kj1
                
                # Zero out the target element
                result[i, j] = 0.0
        
        # Ensure positive diagonal (R's sign adjustment)
        for i in range(self.n_vars):
            if array_lib.__name__ == 'torch':
                if result[i, i] < 0:
                    for k in range(i+1):
                        result[k, i] *= -1
            else:
                if result[i, i] < 0:
                    for k in range(i+1):
                        result[k, i] *= -1
        
        return result
    
    def extract_parameters(self, theta: Union[np.ndarray, "torch.Tensor"]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract μ, Σ, and log-likelihood from parameter vector.
        
        Uses R's exact parameter reconstruction algorithm.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector in R's format
            
        Returns
        -------
        mu : np.ndarray
            Mean vector estimate
        sigma : np.ndarray
            Covariance matrix estimate  
        loglik : float
            Log-likelihood value
        """
        # Handle tensor conversion for output
        if hasattr(theta, 'detach'):  # PyTorch tensor
            theta_np = theta.detach().cpu().numpy()
        else:
            theta_np = np.asarray(theta)
        
        # Extract mean parameters
        mu = theta_np[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta_np, np)
        
        # Convert to covariance matrix: Σ = (Δ⁻¹)ᵀ Δ⁻¹
        try:
            # Use triangular solve for numerical stability (R's approach)
            I = np.eye(self.n_vars)
            Delta_inv = np.linalg.solve(Delta, I)  # More stable than inv(Delta)
            sigma = Delta_inv.T @ Delta_inv
            
            # Ensure exact symmetry (R does this)
            sigma = (sigma + sigma.T) / 2.0
            
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            sigma = np.eye(self.n_vars)
        
        # Compute log-likelihood
        loglik = -self(theta_np) / 2.0  # Objective is negative log-likelihood
        
        return mu, sigma, loglik
    
    def __call__(self, theta: Union[np.ndarray, "torch.Tensor"]) -> Union[float, "torch.Tensor"]:
        """
        Compute negative log-likelihood using R's exact algorithm.
        
        Implements pattern-wise likelihood computation with Givens rotations
        for numerical stability, exactly matching R's evallf.c.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector in R's format
            
        Returns
        -------
        float or tensor
            Negative log-likelihood value
        """
        # Determine array library
        is_torch = hasattr(theta, 'requires_grad')
        if is_torch:
            import torch
            array_lib = torch
        else:
            array_lib = np
        
        # Extract mean parameters
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta, array_lib)
        
        # Apply Givens rotations for numerical stability (R's evallf.c)
        Delta_stabilized = self._apply_givens_rotations(Delta, array_lib)
        
        # Compute negative log-likelihood using pattern-wise formula
        if is_torch:
            neg_loglik = torch.tensor(0.0, dtype=theta.dtype, device=theta.device)
        else:
            neg_loglik = 0.0
        
        for pattern in self.patterns:
            if pattern.n_k == 0 or len(pattern.observed_indices) == 0:
                continue
            
            # Extract relevant submatrices (R's approach)
            obs_indices = pattern.observed_indices
            n_obs_vars = len(obs_indices)
            mu_k = mu[obs_indices]
            
            # CRITICAL: Implement R's row shuffling algorithm exactly
            # Create reordered Delta with observed rows first, missing rows last
            if is_torch:
                subdel = torch.zeros((self.n_vars, self.n_vars), dtype=theta.dtype, device=theta.device)
            else:
                subdel = np.zeros((self.n_vars, self.n_vars))
            
            # Put observed variable rows first
            pcount = 0
            for i in range(self.n_vars):
                if i in obs_indices:
                    subdel[pcount, :] = Delta_stabilized[i, :]
                    pcount += 1
            
            # Put missing variable rows last
            acount = 0
            for i in range(self.n_vars):
                if i not in obs_indices:
                    subdel[self.n_vars - acount - 1, :] = Delta_stabilized[i, :]
                    acount += 1
            
            # Apply Givens rotations to the reordered matrix
            subdel = self._apply_givens_rotations(subdel, array_lib)
            
            # Extract just the observed part (top-left submatrix)
            if is_torch:
                Delta_k = subdel[:n_obs_vars, :n_obs_vars]
            else:
                Delta_k = subdel[:n_obs_vars, :n_obs_vars]
            
            try:
                # Use triangular solve for stability (R's method)
                if is_torch:
                    I_k = torch.eye(n_obs_vars, dtype=theta.dtype, device=theta.device)
                    Delta_k_inv = torch.linalg.solve(Delta_k, I_k)
                else:
                    I_k = np.eye(n_obs_vars)
                    Delta_k_inv = np.linalg.solve(Delta_k, I_k)
                
                Sigma_k = Delta_k_inv.T @ Delta_k_inv
                
                # Log-determinant term
                if is_torch:
                    sign, logdet_Sigma_k = torch.linalg.slogdet(Sigma_k)
                else:
                    sign, logdet_Sigma_k = np.linalg.slogdet(Sigma_k)
                
                # Data fitting term (R's exact formula)
                if is_torch:
                    data_k = torch.tensor(pattern.data_k, dtype=theta.dtype, device=theta.device)
                else:
                    data_k = pattern.data_k
                
                # Center data by pattern mean
                centered = data_k - mu_k
                
                # Compute quadratic form: sum_i (y_i - μ)' Σ^{-1} (y_i - μ)
                if is_torch:
                    Sigma_k_inv = torch.linalg.solve(Sigma_k, I_k)
                    quad_form = torch.sum(torch.sum(centered @ Sigma_k_inv * centered, dim=1))
                else:
                    Sigma_k_inv = np.linalg.solve(Sigma_k, I_k)
                    quad_form = np.sum(np.sum(centered @ Sigma_k_inv * centered, axis=1))
                
                # Pattern contribution to negative log-likelihood
                # -2 * loglik = n_k * log|Σ_k| + quadratic form
                pattern_contrib = pattern.n_k * logdet_Sigma_k + quad_form
                neg_loglik = neg_loglik + pattern_contrib
                
            except (np.linalg.LinAlgError, RuntimeError) as e:
                # Handle numerical issues
                if is_torch:
                    return torch.tensor(1e20, dtype=theta.dtype, device=theta.device)
                else:
                    return 1e20
        
        return neg_loglik