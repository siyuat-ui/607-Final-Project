"""
Visualization functions for latent class model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_bic_curve(bic_results: Dict,
                   true_K: Optional[int] = None,
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Plot BIC and log-likelihood curves for model selection.
    
    Parameters
    ----------
    bic_results : dict
        Results from BICModelSelector.get_all_results()
        Must contain 'K_range', 'bic_values', 'log_likelihoods', 'best_K'
    true_K : int, optional
        True number of classes (if known, e.g., for synthetic data)
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(14, 5)
        Figure size
    """
    K_range = bic_results['K_range']
    bic_values = bic_results['bic_values']
    log_liks = bic_results['log_likelihoods']
    best_K = bic_results['best_K']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # BIC curve
    ax = axes[0]
    ax.plot(K_range, bic_values, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='BIC')
    
    # Mark selected K
    best_idx = K_range.index(best_K)
    ax.scatter([best_K], [bic_values[best_idx]], 
               color='green', s=300, zorder=5, marker='*', 
               edgecolors='black', linewidths=2, label=f'Selected K={best_K}')
    ax.axvline(x=best_K, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # Mark true K if provided
    if true_K is not None and true_K in K_range:
        ax.axvline(x=true_K, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'True K={true_K}')
    
    ax.set_xlabel('Number of Classes (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIC', fontsize=12, fontweight='bold')
    ax.set_title('BIC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_range)
    
    # Log-likelihood curve
    ax = axes[1]
    ax.plot(K_range, log_liks, marker='o', linewidth=2, markersize=8, 
            color='coral', label='Log-Likelihood')
    
    # Mark selected K
    ax.scatter([best_K], [log_liks[best_idx]], 
               color='green', s=300, zorder=5, marker='*', 
               edgecolors='black', linewidths=2, label=f'Selected K={best_K}')
    ax.axvline(x=best_K, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # Mark true K if provided
    if true_K is not None and true_K in K_range:
        ax.axvline(x=true_K, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'True K={true_K}')
    
    ax.set_xlabel('Number of Classes (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log-Likelihood', fontsize=12, fontweight='bold')
    ax.set_title('Log-Likelihood Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_mixture_weights(pi: np.ndarray,
                        true_pi: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot estimated (and optionally true) mixture weights.
    
    Parameters
    ----------
    pi : np.ndarray
        Estimated mixture weights
    true_pi : np.ndarray, optional
        True mixture weights (for comparison)
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(10, 6)
        Figure size
    """
    K = len(pi)
    x = np.arange(K)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if true_pi is not None:
        # Plot both estimated and true
        ax.bar(x - width/2, pi, width, label='Estimated', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, true_pi, width, label='True', color='coral', alpha=0.8)
    else:
        # Plot only estimated
        ax.bar(x, pi, width, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Latent Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mixture Weight (π)', fontsize=12, fontweight='bold')
    ax.set_title('Mixture Weights by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {k}' for k in range(K)])
    ax.grid(True, alpha=0.3, axis='y')
    
    if true_pi is not None:
        ax.legend(fontsize=10)
    
    # Add value labels on bars
    for i, v in enumerate(pi):
        offset = -width/2 if true_pi is not None else 0
        ax.text(i + offset, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    if true_pi is not None:
        for i, v in enumerate(true_pi):
            ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_categorical_probabilities(theta: np.ndarray,
                                   categories: List[int],
                                   variable_names: Optional[List[str]] = None,
                                   max_variables: int = 10,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    Plot categorical probabilities for each class and variable.
    
    Parameters
    ----------
    theta : np.ndarray, shape (K, m, C_max)
        Categorical probabilities
    categories : list of int
        Number of categories for each variable
    variable_names : list of str, optional
        Names of variables
    max_variables : int, default=10
        Maximum number of variables to plot (to avoid overcrowding)
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(16, 10)
        Figure size
    """
    K, m, C_max = theta.shape
    
    if variable_names is None:
        variable_names = [f'Var {r}' for r in range(m)]
    
    # Limit number of variables to plot
    m_plot = min(m, max_variables)
    if m > max_variables:
        print(f"Note: Plotting first {max_variables} variables out of {m}")
    
    # Create subplots
    n_cols = min(4, m_plot)
    n_rows = int(np.ceil(m_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if m_plot > 1 else [axes]
    
    for r in range(m_plot):
        ax = axes[r]
        C_r = categories[r]
        
        # Prepare data for this variable
        x = np.arange(C_r)
        width = 0.8 / K
        
        # Plot bars for each class
        for k in range(K):
            offset = (k - K/2 + 0.5) * width
            probs = theta[k, r, :C_r]
            ax.bar(x + offset, probs, width, label=f'Class {k}', alpha=0.8)
        
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f'{variable_names[r]}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        if r == 0:  # Add legend to first subplot
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for idx in range(m_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Categorical Probabilities by Class and Variable', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_class_assignments(true_labels: Optional[np.ndarray],
                          predicted_labels: np.ndarray,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot class assignment comparison (if true labels are available).
    
    Parameters
    ----------
    true_labels : np.ndarray, optional
        True latent class assignments
    predicted_labels : np.ndarray
        Predicted latent class assignments
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(12, 5)
        Figure size
    """
    if true_labels is None:
        # Only plot predicted distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        
        unique, counts = np.unique(predicted_labels, return_counts=True)
        ax.bar(unique, counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Predicted Classes', fontsize=14, fontweight='bold')
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Plot confusion matrix and distributions
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Confusion matrix
        ax = axes[0]
        K = max(max(true_labels), max(predicted_labels)) + 1
        confusion = np.zeros((K, K), dtype=int)
        
        for t, p in zip(true_labels, predicted_labels):
            confusion[t, p] += 1
        
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[f'Pred {k}' for k in range(K)],
                   yticklabels=[f'True {k}' for k in range(K)])
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        
        # Class distributions
        ax = axes[1]
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        unique_pred, counts_pred = np.unique(predicted_labels, return_counts=True)
        
        x = np.arange(K)
        width = 0.35
        
        ax.bar(x - width/2, [counts_true[counts_true >= 0][k] if k in unique_true else 0 for k in range(K)], 
               width, label='True', color='coral', alpha=0.8)
        ax.bar(x + width/2, [counts_pred[counts_pred >= 0][k] if k in unique_pred else 0 for k in range(K)], 
               width, label='Predicted', color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distributions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Calculate accuracy
        accuracy = np.mean(true_labels == predicted_labels)
        plt.suptitle(f'Class Assignments (Accuracy: {accuracy:.2%})', 
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_posterior_uncertainty(gamma: np.ndarray,
                               predicted_labels: np.ndarray,
                               n_samples: int = 500,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Plot posterior probability distributions to assess classification uncertainty.
    
    Parameters
    ----------
    gamma : np.ndarray, shape (n, K)
        Posterior probabilities
    predicted_labels : np.ndarray, shape (n,)
        Predicted class labels
    n_samples : int, default=500
        Number of samples to plot (to avoid overcrowding)
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(14, 5)
        Figure size
    """
    n, K = gamma.shape
    
    # Subsample if needed
    if n > n_samples:
        idx = np.random.choice(n, size=n_samples, replace=False)
        gamma_plot = gamma[idx]
        labels_plot = predicted_labels[idx]
    else:
        gamma_plot = gamma
        labels_plot = predicted_labels
    
    # Sort by predicted class
    sort_idx = np.argsort(labels_plot)
    gamma_sorted = gamma_plot[sort_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Stacked bar chart
    ax = axes[0]
    bottom = np.zeros(len(gamma_sorted))
    
    for k in range(K):
        ax.bar(range(len(gamma_sorted)), gamma_sorted[:, k], bottom=bottom,
               label=f'Class {k}', alpha=0.8)
        bottom += gamma_sorted[:, k]
    
    ax.set_xlabel('Sample (sorted by predicted class)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posterior Probability', fontsize=12, fontweight='bold')
    ax.set_title('Posterior Probabilities (Stacked)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim([0, 1])
    
    # Entropy histogram
    ax = axes[1]
    entropy = -np.sum(gamma * np.log(gamma + 1e-10), axis=1)
    
    ax.hist(entropy, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(entropy), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(entropy):.3f}')
    ax.set_xlabel('Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Classification Uncertainty (Entropy)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def create_results_summary_table(model_params: Dict,
                                 variable_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of model results.
    
    Parameters
    ----------
    model_params : dict
        Model parameters from get_parameters()
    variable_names : list of str, optional
        Variable names
    save_path : str, optional
        Path to save CSV
        
    Returns
    -------
    pd.DataFrame
        Summary table
    """
    K = model_params['K']
    pi = model_params['pi']
    theta = model_params['theta']
    categories = model_params['categories']
    m = len(categories)
    
    if variable_names is None:
        variable_names = [f'Var_{r}' for r in range(m)]
    
    # Create summary
    summary_data = []
    
    # Mixture weights
    for k in range(K):
        summary_data.append({
            'Class': k,
            'Mixture_Weight': pi[k],
            'Percentage': f'{pi[k]*100:.2f}%'
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\nModel Summary:")
    print("=" * 60)
    print(f"Number of classes (K): {K}")
    print(f"Number of variables (m): {m}")
    print(f"Log-likelihood: {model_params['log_likelihood']:.4f}")
    print(f"Converged: {model_params['converged']}")
    print(f"Iterations: {model_params['n_iterations']}")
    print("\nMixture Weights:")
    print(df_summary.to_string(index=False))
    print("=" * 60)
    
    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"\nSummary saved to: {save_path}")
    
    return df_summary