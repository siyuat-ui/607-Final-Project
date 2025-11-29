"""
Latent class model for multivariate categorical data with EM algorithm.
"""

import numpy as np
from typing import Optional, Tuple, List
from joblib import Parallel, delayed
from src.utils import compute_gamma_stable, enforce_ordering, check_convergence


class LatentClassModel:
    """
    Latent class model for multivariate categorical data.
    
    Model specification (from PDF page 1):
    P(X = x) = Σ_k π_k ∏_r P(X^(r) = x^(r) | H = k)
    
    where:
    - π_k are mixture weights
    - θ_rkc = P(X^(r) = c | H = k) are categorical probabilities
    """
    
    def __init__(self, 
                 K: int,
                 categories: List[int],
                 random_state: Optional[int] = None):
        """
        Initialize the latent class model.
        
        Parameters
        ----------
        K : int
            Number of latent classes
        categories : list of int
            Number of categories for each variable
        random_state : int, optional
            Random seed for reproducibility
        """
        self.K = K
        self.m = len(categories)
        self.categories = categories
        self.C_max = max(categories)
        self.random_state = random_state
        
        # Parameters to be estimated
        self.pi = None  # Mixture weights, shape (K,)
        self.theta = None  # Categorical probabilities, shape (K, m, C_max)
        
        # Training history
        self.log_likelihood_history = []
        self.converged = False
        self.n_iterations = 0
        self.best_log_likelihood = -np.inf
        
    def _initialize_parameters(self, seed: Optional[int] = None) -> None:
        """
        Initialize parameters randomly.
        
        From PDF: π_k are mixture weights, θ_rkc are categorical probabilities
        
        Parameters
        ----------
        seed : int, optional
            Random seed for this specific initialization
        """
        rng = np.random.default_rng(seed)
        
        # Initialize mixture weights uniformly with small random perturbation
        self.pi = np.ones(self.K) / self.K + rng.normal(0, 0.01, self.K)
        self.pi = np.abs(self.pi)
        self.pi /= self.pi.sum()
        
        # Sort to satisfy ordering constraint
        self.pi = np.sort(self.pi)[::-1]
        
        # Initialize categorical probabilities
        self.theta = np.zeros((self.K, self.m, self.C_max))
        
        for k in range(self.K):
            for r in range(self.m):
                # Random initialization with Dirichlet
                alpha = np.ones(self.categories[r])
                probs = rng.dirichlet(alpha)
                self.theta[k, r, :self.categories[r]] = probs
    
    def _compute_log_theta_products(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Σ_r log(θ_rkX_i^(r)) for all samples and classes.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
            
        Returns
        -------
        log_theta_products : np.ndarray, shape (n, K)
            For each sample i and class k: Σ_r log(θ_rkX_i^(r))
        """
        n = X.shape[0]
        log_theta_products = np.zeros((n, self.K))
        
        for k in range(self.K):
            for r in range(self.m):
                # Extract the log probabilities for the observed categories
                # X[i, r] gives the category for sample i, variable r
                log_theta_products[:, k] += np.log(self.theta[k, r, X[:, r]] + 1e-10)
        
        return log_theta_products
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Compute posterior probabilities γ_ik.
        
        From PDF (page 2):
        γ_ik = P(H_i = k | X_i, Θ) = π_k ∏_r θ_rkX_i^(r) / Σ_j π_j ∏_r θ_rjX_i^(r)
        
        Using numerically stable computation via log-space.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
            
        Returns
        -------
        gamma : np.ndarray, shape (n, K)
            Posterior probabilities
        """
        log_pi = np.log(self.pi + 1e-10)
        log_theta_products = self._compute_log_theta_products(X)
        
        gamma = compute_gamma_stable(log_pi, log_theta_products)
        
        return gamma
    
    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """
        M-step: Update parameters to maximize expected complete log-likelihood.
        
        From PDF (page 2):
        π_k^new = (1/n) Σ_i γ_ik
        θ_rkc^new = Σ_i γ_ik 1(X_i^(r) = c) / Σ_i γ_ik
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
        gamma : np.ndarray, shape (n, K)
            Posterior probabilities from E-step
        """
        n = X.shape[0]
        
        # Update mixture weights
        # π_k^new = (1/n) Σ_i γ_ik
        self.pi = np.mean(gamma, axis=0)
        
        # Update categorical probabilities
        # θ_rkc^new = Σ_i γ_ik 1(X_i^(r) = c) / Σ_i γ_ik
        for k in range(self.K):
            for r in range(self.m):
                for c in range(self.categories[r]):
                    # Indicator: 1(X_i^(r) = c)
                    indicator = (X[:, r] == c).astype(float)
                    
                    # Numerator: Σ_i γ_ik 1(X_i^(r) = c)
                    numerator = np.sum(gamma[:, k] * indicator)
                    
                    # Denominator: Σ_i γ_ik
                    denominator = np.sum(gamma[:, k])
                    
                    self.theta[k, r, c] = numerator / (denominator + 1e-10)
        
        # Enforce ordering constraint to handle label switching
        self.pi, self.theta = enforce_ordering(self.pi, self.theta)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the observed data log-likelihood.
        
        From PDF (page 1):
        ℓ(Θ | {X_i}) = Σ_i log(Σ_k π_k ∏_r θ_rkX_i^(r))
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
            
        Returns
        -------
        log_likelihood : float
            Log-likelihood value
        """
        n = X.shape[0]
        log_pi = np.log(self.pi + 1e-10)
        log_theta_products = self._compute_log_theta_products(X)
        
        # a_ik = log π_k + Σ_r log(θ_rkX_i^(r))
        a = log_pi[np.newaxis, :] + log_theta_products
        
        # log(Σ_k exp(a_ik)) for each i, using log-sum-exp trick
        M = np.max(a, axis=1, keepdims=True)
        log_sum = M.squeeze() + np.log(np.sum(np.exp(a - M), axis=1))
        
        # Sum over all samples
        log_likelihood = np.sum(log_sum)
        
        return log_likelihood
    
    def _fit_single_run(self, 
                       X: np.ndarray,
                       max_iter: int,
                       tol: float,
                       seed: Optional[int]) -> Tuple[dict, float, int, bool]:
        """
        Fit the model with a single random initialization.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        seed : int, optional
            Random seed for initialization
            
        Returns
        -------
        params : dict
            Estimated parameters
        log_lik : float
            Final log-likelihood
        n_iter : int
            Number of iterations
        converged : bool
            Whether the algorithm converged
        """
        # Initialize parameters with specific seed
        self._initialize_parameters(seed=seed)
        
        log_lik_history = []
        converged = False
        
        for iteration in range(max_iter):
            # Store old parameters for convergence check
            old_params = {'pi': self.pi.copy(), 'theta': self.theta.copy()}
            
            # E-step
            gamma = self._e_step(X)
            
            # M-step
            self._m_step(X, gamma)
            
            # Compute log-likelihood
            log_lik = self._compute_log_likelihood(X)
            log_lik_history.append(log_lik)
            
            # Check convergence
            new_params = {'pi': self.pi.copy(), 'theta': self.theta.copy()}
            if check_convergence(old_params, new_params, tol):
                converged = True
                break
        
        # Return the final state
        params = {
            'pi': self.pi.copy(),
            'theta': self.theta.copy()
        }
        
        final_log_lik = log_lik_history[-1] if log_lik_history else -np.inf
        n_iter = len(log_lik_history)
        
        return params, final_log_lik, n_iter, converged
    
    def fit(self, 
            X: np.ndarray,
            max_iter: int = 200,
            tol: float = 1e-6,
            n_init: int = 10,
            n_jobs: int = -1,
            verbose: bool = True) -> 'LatentClassModel':
        """
        Fit the latent class model using EM algorithm with multiple random initializations.
        
        Uses parallel processing to speed up multiple initializations.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix where each column is a categorical variable
        max_iter : int, default=200
            Maximum number of EM iterations per initialization
        tol : float, default=1e-6
            Convergence tolerance
        n_init : int, default=10
            Number of random initializations. The best result is kept.
        n_jobs : int, default=-1
            Number of parallel jobs. -1 means using all processors.
        verbose : bool, default=True
            Whether to print progress
            
        Returns
        -------
        self : LatentClassModel
            Fitted model with best parameters from all initializations
        """
        n, m = X.shape
        
        if m != self.m:
            raise ValueError(f"Data has {m} variables but model expects {self.m}")
        
        if verbose:
            print(f"Fitting latent class model with K={self.K} classes...")
            print(f"Running EM algorithm with {n_init} random initializations (parallelized)...")
        
        # Generate random seeds for each initialization
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=n_init)
        
        # Run parallel EM with different initializations
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_single_run)(X, max_iter, tol, seed)
            for seed in seeds
        )
        
        # Extract results
        all_params = [r[0] for r in results]
        all_log_liks = [r[1] for r in results]
        all_n_iters = [r[2] for r in results]
        all_converged = [r[3] for r in results]
        
        # Find best solution
        best_idx = np.argmax(all_log_liks)
        best_params = all_params[best_idx]
        best_log_lik = all_log_liks[best_idx]
        best_n_iter = all_n_iters[best_idx]
        best_converged = all_converged[best_idx]
        
        # Set the best parameters
        self.pi = best_params['pi']
        self.theta = best_params['theta']
        self.best_log_likelihood = best_log_lik
        self.n_iterations = best_n_iter
        self.converged = best_converged
        self.log_likelihood_history = [best_log_lik]  # Store only final value
        
        if verbose:
            print(f"\nBest solution:")
            print(f"  Log-likelihood: {best_log_lik:.4f}")
            print(f"  Iterations: {best_n_iter}")
            print(f"  Converged: {best_converged}")
            print(f"  Converged runs: {sum(all_converged)}/{n_init}")
            print(f"  Range of log-likelihoods: [{min(all_log_liks):.4f}, {max(all_log_liks):.4f}]")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for new data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
            
        Returns
        -------
        gamma : np.ndarray, shape (n, K)
            Posterior probabilities P(H = k | X)
        """
        if self.pi is None or self.theta is None:
            raise ValueError("Model has not been fitted yet")
        
        return self._e_step(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict latent class assignments for new data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
            
        Returns
        -------
        labels : np.ndarray, shape (n,)
            Predicted class labels
        """
        gamma = self.predict_proba(X)
        return np.argmax(gamma, axis=1)
    
    def get_parameters(self) -> dict:
        """
        Get the estimated model parameters.
        
        Returns
        -------
        dict
            Dictionary containing 'pi' and 'theta'
        """
        if self.pi is None or self.theta is None:
            raise ValueError("Model has not been fitted yet")
        
        return {
            'pi': self.pi.copy(),
            'theta': self.theta.copy(),
            'K': self.K,
            'categories': self.categories,
            'log_likelihood': self.best_log_likelihood,
            'n_iterations': self.n_iterations,
            'converged': self.converged
        }