"""
Utility functions for latent class model.
"""

import numpy as np
from typing import Tuple


def log_sum_exp(log_values: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Compute log(sum(exp(log_values))) in a numerically stable way.
    
    Parameters
    ----------
    log_values : np.ndarray
        Array of log values
    axis : int, optional
        Axis along which to compute the sum
        
    Returns
    -------
    np.ndarray
        log(sum(exp(log_values)))
    """
    max_val = np.max(log_values, axis=axis, keepdims=True)
    return max_val.squeeze() + np.log(np.sum(np.exp(log_values - max_val), axis=axis))


def compute_gamma_stable(log_pi: np.ndarray, 
                         log_theta_products: np.ndarray) -> np.ndarray:
    """
    Compute posterior probabilities γ_ik in a numerically stable way.
    
    From the PDF (page 2):
    Define a_ik = log π_k + Σ_r log(θ_rkX_i^(r))
    Then γ_ik = exp(a_ik - M_i) / Σ_j exp(a_ij - M_i)
    where M_i = max_j a_ij
    
    Parameters
    ----------
    log_pi : np.ndarray, shape (K,)
        Log of mixture weights
    log_theta_products : np.ndarray, shape (n, K)
        For each sample i and class k: Σ_r log(θ_rkX_i^(r))
        
    Returns
    -------
    gamma : np.ndarray, shape (n, K)
        Posterior probabilities γ_ik = P(H_i = k | X_i, Θ)
    """
    # a_ik = log π_k + Σ_r log(θ_rkX_i^(r))
    # Shape: (n, K)
    a = log_pi[np.newaxis, :] + log_theta_products
    
    # M_i = max_j a_ij for each sample i
    # Shape: (n, 1)
    M = np.max(a, axis=1, keepdims=True)
    
    # exp(a_ik - M_i)
    exp_shifted = np.exp(a - M)
    
    # γ_ik = exp(a_ik - M_i) / Σ_j exp(a_ij - M_i)
    gamma = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    return gamma


def check_convergence(old_params: dict, 
                     new_params: dict, 
                     tol: float = 1e-6) -> bool:
    """
    Check if EM algorithm has converged.
    
    Parameters
    ----------
    old_params : dict
        Previous iteration parameters
    new_params : dict
        Current iteration parameters
    tol : float
        Convergence tolerance
        
    Returns
    -------
    bool
        True if converged
    """
    # Check convergence on mixture weights
    pi_diff = np.max(np.abs(new_params['pi'] - old_params['pi']))
    
    # Check convergence on categorical probabilities
    theta_diff = np.max(np.abs(new_params['theta'] - old_params['theta']))
    
    return (pi_diff < tol) and (theta_diff < tol)


def enforce_ordering(pi: np.ndarray, 
                    theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce ordering constraint π_1 ≥ π_2 ≥ ... ≥ π_K to handle label switching.
    
    From PDF (page 2): "To this end, we impose an ordering constraint on the 
    mixture weights: π_1 ≥ ... ≥ π_K"
    
    Parameters
    ----------
    pi : np.ndarray, shape (K,)
        Mixture weights
    theta : np.ndarray, shape (K, m, C_max)
        Categorical probabilities
        
    Returns
    -------
    pi_sorted : np.ndarray
        Sorted mixture weights
    theta_sorted : np.ndarray
        Correspondingly permuted categorical probabilities
    """
    # Get sorting indices (descending order)
    sort_idx = np.argsort(pi)[::-1]
    
    # Sort parameters
    pi_sorted = pi[sort_idx]
    theta_sorted = theta[sort_idx]
    
    return pi_sorted, theta_sorted


def validate_categorical_data(X: np.ndarray, categories: list) -> None:
    """
    Validate that data is categorical and within valid ranges.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, m)
        Data matrix where each column is a categorical variable
    categories : list of int
        Number of categories for each variable
        
    Raises
    ------
    ValueError
        If data contains invalid values
    """
    n, m = X.shape
    
    if len(categories) != m:
        raise ValueError(f"Length of categories ({len(categories)}) must match "
                        f"number of variables ({m})")
    
    for r in range(m):
        unique_vals = np.unique(X[:, r])
        if not np.all(np.isin(unique_vals, range(categories[r]))):
            raise ValueError(f"Variable {r} contains values outside range "
                           f"[0, {categories[r]-1}]")
        
        # Check for missing categories (optional warning)
        if len(unique_vals) < categories[r]:
            print(f"Warning: Variable {r} does not use all {categories[r]} categories")