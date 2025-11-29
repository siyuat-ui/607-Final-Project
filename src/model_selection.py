"""
Model selection for latent class models using BIC.
Optimized parallelization strategy.
"""

import numpy as np
import multiprocessing
from typing import List, Tuple, Optional, Dict
from joblib import Parallel, delayed
from src.latent_class_modeling import LatentClassModel


def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).
    
    From the notes:
    BIC = -2 * log_likelihood + n_params * log(n_samples)
    
    Lower BIC indicates better model fit with penalty for complexity.
    
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model
    n_params : int
        Number of free parameters in the model
    n_samples : int
        Number of samples in the dataset
        
    Returns
    -------
    bic : float
        BIC value
    """
    return -2 * log_likelihood + n_params * np.log(n_samples)


def count_parameters(K: int, categories: List[int]) -> int:
    """
    Count the number of free parameters in a latent class model.
    
    From the PDF:
    - Mixture weights: K - 1 (since they sum to 1)
    - Categorical probabilities: Σ_r K(C_r - 1) (probabilities sum to 1 for each class and variable)
    
    Total: (K - 1) + Σ_r K(C_r - 1)
    
    Parameters
    ----------
    K : int
        Number of latent classes
    categories : list of int
        Number of categories for each variable
        
    Returns
    -------
    n_params : int
        Number of free parameters
    """
    m = len(categories)
    
    # Mixture weights: K - 1
    n_pi = K - 1
    
    # Categorical probabilities: Σ_r K(C_r - 1)
    n_theta = sum(K * (C_r - 1) for C_r in categories)
    
    return n_pi + n_theta


def fit_single_K(X: np.ndarray,
                 K: int,
                 categories: List[int],
                 max_iter: int,
                 tol: float,
                 n_init: int,
                 n_jobs_init: int,
                 random_state: Optional[int],
                 verbose: bool) -> Tuple[int, float, float, dict]:
    """
    Fit a latent class model for a single value of K and compute BIC.
    
    This is a helper function for parallel BIC computation across different K values.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    K : int
        Number of latent classes
    categories : list of int
        Number of categories for each variable
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of random initializations
    n_jobs_init : int
        Number of parallel jobs for initializations within this K
    random_state : int, optional
        Random seed
    verbose : bool
        Verbosity
        
    Returns
    -------
    K : int
        Number of classes
    bic : float
        BIC value
    log_lik : float
        Log-likelihood
    model_params : dict
        Fitted model parameters
    """
    n_samples = X.shape[0]
    
    # Fit model
    model = LatentClassModel(K=K, categories=categories, random_state=random_state)
    model.fit(X, max_iter=max_iter, tol=tol, n_init=n_init, n_jobs=n_jobs_init, verbose=False)
    
    # Get log-likelihood
    log_lik = model.best_log_likelihood
    
    # Compute BIC
    n_params = count_parameters(K, categories)
    bic = compute_bic(log_lik, n_params, n_samples)
    
    if verbose:
        print(f"K={K}: BIC={bic:.2f}, Log-likelihood={log_lik:.2f}, "
              f"Parameters={n_params}, Converged={model.converged}")
    
    return K, bic, log_lik, model.get_parameters()


class BICModelSelector:
    """
    Select optimal number of latent classes using BIC.
    Optimized parallelization strategy.
    """
    
    def __init__(self,
                 K_range: List[int],
                 categories: List[int],
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 n_init: int = 10,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize BIC model selector.
        
        Parameters
        ----------
        K_range : list of int
            Range of K values to try (e.g., [1, 2, 3, 4, 5])
        categories : list of int
            Number of categories for each variable
        max_iter : int, default=200
            Maximum EM iterations
        tol : float, default=1e-6
            Convergence tolerance
        n_init : int, default=10
            Number of random initializations per K
        n_jobs : int, default=-1
            Number of parallel jobs. -1 uses all processors.
        random_state : int, optional
            Random seed for reproducibility
        """
        self.K_range = sorted(K_range)
        self.categories = categories
        self.m = len(categories)
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.random_state = random_state
        
        # Results storage
        self.bic_values = None
        self.log_likelihoods = None
        self.best_K = None
        self.best_model_params = None
        self.all_model_params = None
        
    def _determine_parallel_strategy(self) -> Tuple[int, int]:
        """
        Determine optimal parallelization strategy.
        
        Strategy:
        - If K_range is small (≤4), parallelize initializations within each K
        - If K_range is large (>4), parallelize across K values
        - If both are moderate, use nested parallelization
        
        Returns
        -------
        n_jobs_K : int
            Number of parallel jobs for K values
        n_jobs_init : int
            Number of parallel jobs for initializations within each K
        """
        n_K = len(self.K_range)
        n_cores = self.n_jobs
        
        if n_K <= 3:
            # Few K values: parallelize initializations
            return 1, n_cores
        elif n_K >= n_cores:
            # Many K values: parallelize across K
            return n_cores, 1
        else:
            # Moderate case: distribute cores
            # Try to balance parallelization
            cores_per_K = max(1, n_cores // n_K)
            n_jobs_K = min(n_K, n_cores)
            n_jobs_init = max(1, cores_per_K)
            return n_jobs_K, n_jobs_init
    
    def fit(self, X: np.ndarray, 
            verbose: bool = True, 
            parallel_strategy: str = 'auto') -> 'BICModelSelector':
        """
        Fit models for all K values and select best via BIC.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, m)
            Data matrix
        verbose : bool, default=True
            Whether to print progress
        parallel_strategy : str, default='auto'
            Parallelization strategy:
            - 'auto': Automatically determine best strategy
            - 'across_K': Parallelize across K values (sequential initializations)
            - 'within_K': Sequential K values (parallel initializations)
            
        Returns
        -------
        self : BICModelSelector
            Fitted selector
        """
        n_samples, m = X.shape
        
        if m != self.m:
            raise ValueError(f"Data has {m} variables but model expects {self.m}")
        
        if verbose:
            print(f"BIC Model Selection: Testing K in {self.K_range}")
            print(f"Each K fitted with {self.n_init} random initializations")
            print(f"Total available cores: {self.n_jobs}")
            print("=" * 60)
        
        # Determine parallelization strategy
        if parallel_strategy == 'auto':
            n_jobs_K, n_jobs_init = self._determine_parallel_strategy()
            if verbose:
                print(f"Auto-selected strategy: {n_jobs_K} jobs for K, {n_jobs_init} jobs for inits")
        elif parallel_strategy == 'across_K':
            n_jobs_K = self.n_jobs
            n_jobs_init = 1
            if verbose:
                print(f"Strategy: Parallelize across K values")
        elif parallel_strategy == 'within_K':
            n_jobs_K = 1
            n_jobs_init = self.n_jobs
            if verbose:
                print(f"Strategy: Parallelize within each K")
        else:
            raise ValueError(f"Unknown parallel_strategy: {parallel_strategy}")
        
        # Fit models for each K value
        if n_jobs_K > 1:
            # Parallelize across K values
            if verbose:
                print(f"Running {len(self.K_range)} K values in parallel...")
            
            results = Parallel(n_jobs=n_jobs_K)(
                delayed(fit_single_K)(
                    X, K, self.categories, self.max_iter, self.tol,
                    self.n_init, n_jobs_init, self.random_state, verbose
                )
                for K in self.K_range
            )
        else:
            # Sequential across K values (but parallel initializations within each K)
            if verbose:
                print(f"Running K values sequentially with parallel initializations...")
            
            results = []
            for K in self.K_range:
                if verbose:
                    print(f"\nFitting K={K}...")
                result = fit_single_K(
                    X, K, self.categories, self.max_iter, self.tol,
                    self.n_init, n_jobs_init, self.random_state, verbose
                )
                results.append(result)
        
        # Extract results
        K_values = [r[0] for r in results]
        bic_values = [r[1] for r in results]
        log_liks = [r[2] for r in results]
        all_params = [r[3] for r in results]
        
        # Store results
        self.bic_values = dict(zip(K_values, bic_values))
        self.log_likelihoods = dict(zip(K_values, log_liks))
        self.all_model_params = dict(zip(K_values, all_params))
        
        # Select best K (minimum BIC)
        best_idx = np.argmin(bic_values)
        self.best_K = K_values[best_idx]
        self.best_model_params = all_params[best_idx]
        
        if verbose:
            print("=" * 60)
            print(f"\nBest K selected: {self.best_K} (BIC = {self.bic_values[self.best_K]:.2f})")
            print(f"Log-likelihood: {self.log_likelihoods[self.best_K]:.2f}")
        
        return self
    
    def get_best_model(self) -> Dict:
        """
        Get parameters of the best model (selected by BIC).
        
        Returns
        -------
        dict
            Best model parameters including K, pi, theta, etc.
        """
        if self.best_K is None:
            raise ValueError("Model selection not performed yet. Call fit() first.")
        
        return self.best_model_params
    
    def get_all_results(self) -> Dict:
        """
        Get BIC and log-likelihood for all K values tested.
        
        Returns
        -------
        dict
            Dictionary with 'K_range', 'bic_values', 'log_likelihoods'
        """
        if self.bic_values is None:
            raise ValueError("Model selection not performed yet. Call fit() first.")
        
        return {
            'K_range': self.K_range,
            'bic_values': [self.bic_values[K] for K in self.K_range],
            'log_likelihoods': [self.log_likelihoods[K] for K in self.K_range],
            'best_K': self.best_K
        }
    
    def get_model_params(self, K: int) -> Dict:
        """
        Get parameters for a specific K value.
        
        Parameters
        ----------
        K : int
            Number of classes
            
        Returns
        -------
        dict
            Model parameters for specified K
        """
        if self.all_model_params is None:
            raise ValueError("Model selection not performed yet. Call fit() first.")
        
        if K not in self.all_model_params:
            raise ValueError(f"K={K} not in tested range {self.K_range}")
        
        return self.all_model_params[K]


def select_best_K(X: np.ndarray,
                  K_range: List[int],
                  categories: List[int],
                  max_iter: int = 200,
                  tol: float = 1e-6,
                  n_init: int = 10,
                  n_jobs: int = -1,
                  random_state: Optional[int] = None,
                  verbose: bool = True) -> Tuple[int, Dict]:
    """
    Convenience function to select best K using BIC.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    K_range : list of int
        Range of K values to test
    categories : list of int
        Number of categories for each variable
    max_iter : int, default=200
        Maximum EM iterations
    tol : float, default=1e-6
        Convergence tolerance
    n_init : int, default=10
        Number of random initializations
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, optional
        Random seed
    verbose : bool, default=True
        Verbosity
        
    Returns
    -------
    best_K : int
        Selected number of classes
    results : dict
        All BIC results
    """
    selector = BICModelSelector(
        K_range=K_range,
        categories=categories,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    selector.fit(X, verbose=verbose)
    
    return selector.best_K, selector.get_all_results()