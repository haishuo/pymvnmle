#!/usr/bin/env python3
"""Export problematic data for R testing."""

import numpy as np
import pandas as pd

# Generate the exact problematic dataset
np.random.seed(42)
n_obs, n_vars = 2000, 15

A = np.random.randn(n_vars, n_vars)
sigma_true = A @ A.T + np.eye(n_vars)
mu_true = np.random.randn(n_vars) * 3

data = np.random.multivariate_normal(mu_true, sigma_true, n_obs)

# Create 3 patterns like gpu_showcase
n_per_pattern = n_obs // 10
data[:n_per_pattern, -1] = np.nan
data[n_per_pattern:2*n_per_pattern, 0] = np.nan

# Save for R
pd.DataFrame(data).to_csv('problematic_data.csv', index=False, na_rep='NA')
print("Data saved to problematic_data.csv")
print(f"Shape: {data.shape}")
print(f"Missing: {np.sum(np.isnan(data))}/{data.size}")