"""
Data Generating Processes (DGPs) for synthetic categorical data.
Vectorized implementation for faster data generation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List


class LatentClassDGP:
    """
    Generate synthetic data from a latent class model.
    
    The true model is:
    P(X = x) = Σ_k π_k ∏_r P(X^(r) = x^(r) | H = k)
    
    Uses vectorized operations for efficient data generation.
    """
    
    def __init__(self, 
                 K: int,
                 categories: List[int],
                 pi: Optional[np.ndarray] = None,
                 theta: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None):
        """
        Initialize the data generating process.
        
        Parameters
        ----------
        K : int
            Number of latent classes
        categories : list of int
            Number of categories for each variable
        pi : np.ndarray, shape (K,), optional
            Mixture weights. If None, will be randomly generated.
        theta : np.ndarray, shape (K, m, C_max), optional
            Categorical probabilities. If None, will be randomly generated.
        random_state : int, optional
            Random seed for reproducibility
        """
        self.K = K
        self.m = len(categories)
        self.categories = categories
        self.C_max = max(categories)
        self.rng = np.random.default_rng(random_state)
        
        # Generate or set mixture weights
        if pi is None:
            self.pi = self._generate_mixture_weights()
        else:
            if len(pi) != K:
                raise ValueError(f"pi must have length K={K}")
            if not np.isclose(np.sum(pi), 1.0):
                raise ValueError("pi must sum to 1")
            self.pi = pi
        
        # Generate or set categorical probabilities
        if theta is None:
            self.theta = self._generate_categorical_probs()
        else:
            if theta.shape[0] != K or theta.shape[1] != self.m:
                raise ValueError(f"theta must have shape (K={K}, m={self.m}, C_max)")
            self.theta = theta
    
    def _generate_mixture_weights(self) -> np.ndarray:
        """
        Generate random mixture weights that sum to 1 using Dirichlet distribution.
        
        Returns
        -------
        pi : np.ndarray, shape (K,)
            Mixture weights in descending order
        """
        # Generate from Dirichlet distribution for diversity
        pi = self.rng.dirichlet(np.ones(self.K)) + np.ones(self.K) * 0.2  # Small constant to avoid very small weights
        pi /= pi.sum()
        
        # Sort in descending order to satisfy ordering constraint
        pi = np.sort(pi)[::-1]
        
        return pi
    
    def _generate_categorical_probs(self) -> np.ndarray:
        """
        Generate random categorical probabilities for each class and variable.
        
        Returns
        -------
        theta : np.ndarray, shape (K, m, C_max)
            Categorical probabilities
        """
        theta = np.zeros((self.K, self.m, self.C_max))
        
        for k in range(self.K):
            for r in range(self.m):
                # Generate probabilities for this variable's categories
                # Using Dirichlet to ensure they sum to 1
                alpha = np.ones(self.categories[r]) * 2.0  # Concentration parameter
                probs = self.rng.dirichlet(alpha)
                
                # Place in the theta array
                theta[k, r, :self.categories[r]] = probs
        
        return theta
    
    def generate_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data from the latent class model.
        
        VECTORIZED implementation for faster generation.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
            
        Returns
        -------
        X : np.ndarray, shape (n, m)
            Generated categorical data
        H : np.ndarray, shape (n,)
            True latent class assignments
        """
        # Sample latent classes (already vectorized)
        H = self.rng.choice(self.K, size=n, p=self.pi)
        
        # Initialize data matrix
        X = np.zeros((n, self.m), dtype=int)
        
        # VECTORIZED generation: for each variable, generate all samples at once
        for r in range(self.m):
            C_r = self.categories[r]
            
            # Get probabilities for all samples for this variable
            # theta[H, r, :C_r] extracts probabilities for each sample's class
            # Shape: (n, C_r)
            probs_r = self.theta[H, r, :C_r]
            
            # Vectorized categorical sampling using inverse transform sampling
            # Compute cumulative probabilities
            cumsum = np.cumsum(probs_r, axis=1)
            
            # Generate uniform random numbers
            u = self.rng.random(n)[:, None]  # Shape: (n, 1)
            
            # Find first index where cumsum >= u for each sample
            # This is equivalent to categorical sampling
            X[:, r] = np.argmax(u < cumsum, axis=1)
        
        return X, H
    
    def save_data(self, 
                  n: int, 
                  filepath: str,
                  save_latent_classes: bool = False,
                  variable_names: Optional[List[str]] = None) -> None:
        """
        Generate and save synthetic data to CSV.
        
        Parameters
        ----------
        n : int
            Number of samples
        filepath : str
            Output file path
        save_latent_classes : bool, default=False
            If True, include true latent class assignments in output
        variable_names : list of str, optional
            Names for variables. If None, uses Var_0, Var_1, ...
        """
        X, H = self.generate_data(n)
        
        if variable_names is None:
            variable_names = [f"Var_{r}" for r in range(self.m)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=variable_names)
        
        if save_latent_classes:
            df['TrueClass'] = H
        
        df.to_csv(filepath, index=False)
        print(f"Generated {n} samples and saved to: {filepath}")
    
    def get_true_parameters(self) -> dict:
        """
        Get the true parameters used for data generation.
        
        Returns
        -------
        dict
            Dictionary containing 'pi' and 'theta'
        """
        return {
            'pi': self.pi.copy(),
            'theta': self.theta.copy()
        }


def generate_simple_scenario(n: int,
                            filepath: str,
                            K: int = 3,
                            m: int = 20,
                            C: int = 3,
                            random_state: Optional[int] = None) -> dict:
    """
    Generate a simple scenario with equal categories across all variables.
    
    Parameters
    ----------
    n : int
        Number of samples
    filepath : str
        Output file path
    K : int, default=3
        Number of latent classes
    m : int, default=20
        Number of variables
    C : int, default=3
        Number of categories for each variable
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        True parameters
    """
    categories = [C] * m
    dgp = LatentClassDGP(K=K, categories=categories, random_state=random_state)
    dgp.save_data(n, filepath, save_latent_classes=False)
    
    return dgp.get_true_parameters()


def generate_diverse_scenario(n: int,
                              filepath: str,
                              random_state: Optional[int] = None) -> dict:
    """
    Generate a more complex scenario with varying numbers of categories.
    
    Parameters
    ----------
    n : int
        Number of samples
    filepath : str
        Output file path
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        True parameters
    """
    # Example: 4 classes, 6 variables with different numbers of categories
    K = 4
    categories = [2, 2, 3, 3, 4, 5]  # Binary, ternary, and higher
    
    dgp = LatentClassDGP(K=K, categories=categories, random_state=random_state)
    dgp.save_data(n, filepath, save_latent_classes=False)
    
    return dgp.get_true_parameters()