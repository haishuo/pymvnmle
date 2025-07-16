# PyMVNMLE Mathematical Bible
## The Complete Mathematical Specification and Implementation Guide

> "Those who do not remember the past are condemned to repeat it." - George Santayana

This document exists because we've learned (twice) that maximum likelihood estimation for multivariate normal data with missing values is filled with mathematical subtleties that can make or break an implementation. Every algorithm documented here exists for a reason, often discovered through painful debugging.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Mathematical Foundation](#mathematical-foundation)
3. [The Inverse Cholesky Parameterization](#the-inverse-cholesky-parameterization)
4. [Data Preprocessing: The mysort Algorithm](#data-preprocessing-the-mysort-algorithm)
5. [The Critical Row Shuffling Algorithm](#the-critical-row-shuffling-algorithm)
6. [Givens Rotations for Numerical Stability](#givens-rotations-for-numerical-stability)
7. [Pattern-Wise Likelihood Computation](#pattern-wise-likelihood-computation)
8. [Starting Values: Pairwise Complete Covariance](#starting-values-pairwise-complete-covariance)
9. [Gradient Computation: Finite Differences](#gradient-computation-finite-differences)
10. [Common Pitfalls and Lessons Learned](#common-pitfalls-and-lessons-learned)

---

## 1. The Big Picture

PyMVNMLE implements maximum likelihood estimation for multivariate normal data with arbitrary missing data patterns. The goal is to estimate:

- **μ** (mu): The p-dimensional mean vector
- **Σ** (Sigma): The p×p covariance matrix

From data **Y** where some entries are missing (NaN).

### The Likelihood We're Maximizing

For complete data, the log-likelihood is:
```
ℓ(μ, Σ) = -n/2 log|2πΣ| - 1/2 Σᵢ (yᵢ - μ)ᵀ Σ⁻¹ (yᵢ - μ)
```

With missing data, we work with the **observed data likelihood**, which involves marginalizing over missing values.

### The Key Insight

Group observations by their **missingness pattern** and compute the likelihood contribution for each pattern using only the observed variables.

---

## 2. Mathematical Foundation

### 2.1 Missingness Patterns

Each observation has a pattern indicating which variables are observed:
```
Pattern = [1, 0, 1, 1, 0]  # Variables 1,3,4 observed; 2,5 missing
```

### 2.2 Pattern-Wise Likelihood

For pattern k with observed indices Oₖ:
```
ℓₖ(μ, Σ) = -nₖ/2 log|2πΣₖ| - 1/2 Σᵢ (yᵢₖ - μₖ)ᵀ Σₖ⁻¹ (yᵢₖ - μₖ)
```

Where:
- nₖ = number of observations with pattern k
- μₖ = μ[Oₖ] (subset of mean for observed variables)
- Σₖ = Σ[Oₖ, Oₖ] (submatrix of covariance for observed variables)
- yᵢₖ = observed values for observation i with pattern k

### 2.3 The Objective Function

We minimize:
```
f(θ) = -2 × log-likelihood = Σₖ nₖ log|Σₖ| + Σₖ Σᵢ (yᵢₖ - μₖ)ᵀ Σₖ⁻¹ (yᵢₖ - μₖ)
```

**Critical**: R's convention is to minimize -2×log-likelihood, not -log-likelihood!

---

## 3. The Inverse Cholesky Parameterization

### 3.1 Why Not Optimize Σ Directly?

Σ must be positive definite. Unconstrained optimization could produce non-PD matrices.

### 3.2 The Solution: Δ = L⁻¹

We parameterize using the **inverse** of the Cholesky factor:
```
Σ = LᵀL  →  Δ = L⁻¹  →  Σ = (Δ⁻¹)ᵀ(Δ⁻¹)
```

### 3.3 Parameter Vector Structure

```
θ = [μ₁, ..., μₚ,                    # Mean parameters (p)
     log(δ₁₁), ..., log(δₚₚ),        # Log diagonal of Δ (p)
     δ₁₂, δ₁₃, δ₂₃, δ₁₄, ...]       # Upper triangle of Δ (p(p-1)/2)
```

Total parameters: p + p(p+1)/2

### 3.4 Why Log for Diagonal?

- Ensures δᵢᵢ > 0 (diagonal elements must be positive)
- Unconstrained optimization
- Bounded to prevent overflow: -10 ≤ log(δᵢᵢ) ≤ 10

### 3.5 R's Column-Major Ordering

Off-diagonals are ordered by **column then row**:
```python
idx = 2 * p
for j in range(1, p):      # Column
    for i in range(j):     # Row within column
        theta[idx] = Delta[i, j]
        idx += 1
```

**This ordering is CRITICAL for R compatibility!**

---

## 4. Data Preprocessing: The mysort Algorithm

### 4.1 Purpose

Group observations with identical missingness patterns together for efficient computation.

### 4.2 Algorithm

```python
def mysort_data(data):
    # Convert missingness to binary (1=observed, 0=missing)
    is_observed = (~np.isnan(data)).astype(int)
    
    # Convert patterns to decimal for sorting
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_codes = is_observed @ powers
    
    # Sort by pattern code
    sort_indices = np.argsort(pattern_codes)
    sorted_data = data[sort_indices]
    
    # Count frequencies of each pattern
    unique_codes, freq = np.unique(pattern_codes[sort_indices], 
                                   return_counts=True)
```

### 4.3 Why This Matters

- Observations with the same pattern can be processed together
- Reduces redundant computations
- Critical for matching R's implementation

---

## 5. The Critical Row Shuffling Algorithm

### 5.1 The Problem

For pattern k with observed variables Oₖ, we need Σₖ = Σ[Oₖ, Oₖ]. But with the inverse Cholesky parameterization, extracting a submatrix of Σ from a submatrix of Δ is **not straightforward**.

### 5.2 R's Solution: Row Shuffling

Instead of extracting Δₖ = Δ[Oₖ, Oₖ], R does something more complex:

```python
# CRITICAL: R's row shuffling algorithm
subdel = np.zeros((n_vars, n_vars))

# Put observed variable rows FIRST
pcount = 0
for i in range(n_vars):
    if i in observed_indices:
        subdel[pcount, :] = Delta[i, :]
        pcount += 1

# Put missing variable rows LAST
acount = 0
for i in range(n_vars):
    if i not in observed_indices:
        subdel[n_vars - acount - 1, :] = Delta[i, :]
        acount += 1
```

### 5.3 Why This Works

The reordering preserves the mathematical relationships needed for the inverse Cholesky structure while allowing proper extraction of the observed submatrix after Givens rotations.

### 5.4 The Lesson

**This was the bug that broke V2.0!** Simply extracting Δ[Oₖ, Oₖ] gives wrong results. The row shuffling is **mandatory**.

---

## 6. Givens Rotations for Numerical Stability

### 6.1 Purpose

Transform the reordered Δ matrix to ensure numerical stability and proper structure.

### 6.2 The Algorithm

```python
def apply_givens_rotations(matrix, n_vars):
    result = matrix.copy()
    
    # Bottom-up, left-to-right
    for i in range(n_vars-1, -1, -1):  # Start from bottom row
        for j in range(i):              # Left to diagonal
            a = result[i, j]
            b = result[i, j+1] if j+1 < n_vars else 0.0
            
            # Skip if already small
            if abs(a) < 0.000001:
                result[i, j] = 0.0
                continue
            
            # Compute rotation
            r = np.sqrt(a*a + b*b)
            c = a / r
            d = b / r
            
            # Apply rotation to entire matrix
            # ... (details in code)
```

### 6.3 Critical Details

- Process from **bottom to top** (not top to bottom!)
- Within each row, go **left to right**
- Threshold: 0.000001 (R's exact value)
- Flip signs to ensure positive diagonal

---

## 7. Pattern-Wise Likelihood Computation

### 7.1 For Each Pattern k

1. **Row shuffle** Δ_stabilized for pattern k
2. **Apply Givens** to the shuffled matrix
3. **Extract** top-left nₖ×nₖ submatrix → Δₖ
4. **Compute** Σₖ = (Δₖ⁻¹)ᵀ(Δₖ⁻¹)
5. **Calculate** log|Σₖ| and (y - μₖ)ᵀΣₖ⁻¹(y - μₖ)

### 7.2 The Complete Formula

```python
obj_value = 0.0
for pattern in patterns:
    # ... (row shuffling and Givens)
    
    # Log-determinant contribution
    log_det_delta_k = np.sum(np.log(np.diag(Delta_k)))
    obj_value -= 2 * n_k * log_det_delta_k
    
    # Quadratic form contribution
    for i in range(n_k):
        centered = y_i - mu_k
        prod = Delta_k.T @ centered
        obj_value += np.dot(prod, prod)
```

---

## 8. Starting Values: Pairwise Complete Covariance

### 8.1 The Challenge

Need good starting values for optimization. Simple approaches (like diagonal covariance) miss correlations.

### 8.2 R's Solution: Pairwise Complete

```python
# Compute covariance using pairwise complete observations
cov_sample = np.zeros((n_vars, n_vars))

for i in range(n_vars):
    for j in range(i, n_vars):
        # Find observations where BOTH i and j are observed
        mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
        n_complete = np.sum(mask)
        
        if n_complete > 1:
            if i == j:
                cov_sample[i, i] = np.var(data[mask, i], ddof=1)
            else:
                cov_ij = np.cov(data[mask, i], data[mask, j], ddof=1)[0, 1]
                cov_sample[i, j] = cov_sample[j, i] = cov_ij
```

### 8.3 Why This Matters

For sparse patterns (few observed variables), capturing correlations in starting values is **critical** for finding the global optimum.

### 8.4 The Lesson

**This was another V2.0 bug!** Using diagonal starting values led to poor local optima for datasets with strong correlations.

---

## 9. Gradient Computation: Finite Differences

### 9.1 The Discovery

After 40+ years, we discovered that **NO** statistical software has implemented analytical gradients for this problem. Everyone uses finite differences!

### 9.2 R's Finite Difference Parameters

```python
eps = 1.49011612e-08  # R's .Machine$double.eps^(1/3)

for i in range(n_params):
    h = eps * max(abs(theta[i]), 1.0)
    if h < 1e-12:
        h = 1e-12
    
    # Forward difference
    theta_plus = theta.copy()
    theta_plus[i] = theta[i] + h
    grad[i] = (f(theta_plus) - f(theta)) / h
```

### 9.3 Convergence Implications

- Gradient norms at "convergence" are ~1e-4, not machine precision
- This is **expected** with finite differences
- Use gtol=1e-4 for BFGS (not 1e-6!)

---

## 10. Common Pitfalls and Lessons Learned

### 10.1 The Row Shuffling Bug

**Symptom**: Works for simple patterns, fails for sparse patterns
**Cause**: Using Δ[Oₖ, Oₖ] instead of row shuffling
**Fix**: Implement R's exact row shuffling algorithm

### 10.2 The Starting Values Bug

**Symptom**: Poor convergence, especially for correlated data
**Cause**: Using diagonal starting covariance
**Fix**: Compute pairwise complete covariance

### 10.3 The Gradient Tolerance Bug

**Symptom**: Optimization "fails" to converge
**Cause**: Expecting machine precision with finite differences
**Fix**: Use gtol=1e-4, not 1e-6

### 10.4 The Sign Convention Bug

**Symptom**: Log-likelihood off by factor of 2
**Cause**: Confusion about -loglik vs -2×loglik
**Fix**: R minimizes -2×loglik, divide by 2 to get loglik

### 10.5 The Parameter Ordering Bug

**Symptom**: Wrong results despite correct algorithms
**Cause**: Not using R's column-major ordering for off-diagonals
**Fix**: Fill parameters by column, then row within column

---

## Final Words

Every algorithm in this document exists because R's implementation requires it. The temptation to "simplify" is strong, but resist it. These mathematical details are what make the difference between a working implementation and hours of debugging.

Remember:
- The row shuffling is **not optional**
- Pairwise complete covariance is **essential**
- Finite differences are **the standard**
- R's conventions must be followed **exactly**

When in doubt, refer back to this document. It will save you from repeating our painful debugging experiences.