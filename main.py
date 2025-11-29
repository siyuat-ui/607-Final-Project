"""
Main script for latent class model analysis.

This script provides an end-to-end pipeline:
1. Load data from CSV (or generate synthetic data)
2. Automatically select optimal K using BIC
3. Fit the best model
4. Generate visualizations and save results
"""

import numpy as np
import argparse
from pathlib import Path
import json

from src.data_loader import load_csv_data
from src.dgp import generate_simple_scenario
from src.model_selection import BICModelSelector
from src.latent_class_modeling import LatentClassModel
from src.visualization import (
    plot_bic_curve,
    plot_mixture_weights,
    plot_categorical_probabilities,
    plot_class_assignments,
    plot_posterior_uncertainty,
    create_results_summary_table
)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'results', 'results/figures']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_analysis(data_path: str,
                 K_range: list = None,
                 output_prefix: str = "analysis",
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 n_init: int = 10,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 has_header: bool = True,
                 zero_indexed: bool = True,
                 generate_all_plots: bool = True):
    """
    Run complete latent class analysis pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to CSV data file
    K_range : list, optional
        Range of K values to test. If None, uses [1, 2, 3, 4, 5, 6]
    output_prefix : str, default="analysis"
        Prefix for output files
    max_iter : int, default=200
        Maximum EM iterations
    tol : float, default=1e-6
        Convergence tolerance
    n_init : int, default=10
        Number of random initializations
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed
    has_header : bool, default=True
        Whether CSV has header
    zero_indexed : bool, default=True
        Whether data is 0-indexed
    generate_all_plots : bool, default=True
        Whether to generate all visualization plots
    """
    print("\n" + "="*70)
    print("LATENT CLASS MODEL ANALYSIS PIPELINE")
    print("="*70)
    
    # Create output directories
    create_directories()
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[STEP 1] Loading data...")
    X, categories, variable_names = load_csv_data(
        data_path, 
        has_header=has_header,
        zero_indexed=zero_indexed
    )
    
    n_samples, n_variables = X.shape
    print(f"  - Samples: {n_samples}")
    print(f"  - Variables: {n_variables}")
    print(f"  - Categories: {categories}")
    
    # ========================================================================
    # STEP 2: Model Selection via BIC
    # ========================================================================
    print("\n[STEP 2] Model selection via BIC...")
    
    if K_range is None:
        K_range = list(range(1, min(7, n_samples // 10 + 1)))  # Default range
    
    print(f"  - Testing K values: {K_range}")
    
    selector = BICModelSelector(
        K_range=K_range,
        categories=categories,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    selector.fit(X, verbose=True, parallel_strategy='auto')
    
    best_K = selector.best_K
    bic_results = selector.get_all_results()
    
    print(f"\n  ✓ Optimal K selected: {best_K}")
    
    # ========================================================================
    # STEP 3: Fit Best Model
    # ========================================================================
    print(f"\n[STEP 3] Fitting final model with K={best_K}...")
    
    best_model_params = selector.get_best_model()
    
    # Create model object for prediction
    final_model = LatentClassModel(K=best_K, categories=categories, random_state=random_state)
    final_model.pi = best_model_params['pi']
    final_model.theta = best_model_params['theta']
    final_model.best_log_likelihood = best_model_params['log_likelihood']
    final_model.n_iterations = best_model_params['n_iterations']
    final_model.converged = best_model_params['converged']
    
    # Predictions
    gamma = final_model.predict_proba(X)
    predicted_labels = final_model.predict(X)
    
    print(f"  ✓ Model fitted successfully")
    print(f"  - Log-likelihood: {best_model_params['log_likelihood']:.4f}")
    print(f"  - Converged: {best_model_params['converged']}")
    
    # ========================================================================
    # STEP 4: Save Results
    # ========================================================================
    print("\n[STEP 4] Saving results...")
    
    # Save model parameters
    results_dict = {
        'best_K': int(best_K),
        'mixture_weights': best_model_params['pi'].tolist(),
        'log_likelihood': float(best_model_params['log_likelihood']),
        'converged': bool(best_model_params['converged']),
        'n_iterations': int(best_model_params['n_iterations']),
        'bic_values': {int(k): float(v) for k, v in zip(bic_results['K_range'], bic_results['bic_values'])},
        'categories': [int(c) for c in categories],
        'variable_names': variable_names,
        'n_samples': int(n_samples),
        'n_variables': int(n_variables)
    }
    
    results_path = f"results/{output_prefix}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  ✓ Results saved to: {results_path}")
    
    # Save theta (categorical probabilities) separately as numpy array
    theta_path = f"results/{output_prefix}_theta.npy"
    np.save(theta_path, best_model_params['theta'])
    print(f"  ✓ Categorical probabilities saved to: {theta_path}")
    
    # Save predictions
    predictions_path = f"results/{output_prefix}_predictions.csv"
    import pandas as pd
    df_pred = pd.DataFrame({
        'predicted_class': predicted_labels,
        **{f'prob_class_{k}': gamma[:, k] for k in range(best_K)}
    })
    df_pred.to_csv(predictions_path, index=False)
    print(f"  ✓ Predictions saved to: {predictions_path}")
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    if generate_all_plots:
        print("\n[STEP 5] Generating visualizations...")
        
        # BIC curve
        plot_bic_curve(
            bic_results,
            save_path=f"results/figures/{output_prefix}_bic_curve.png"
        )
        print("  ✓ BIC curve saved")
        
        # Mixture weights
        plot_mixture_weights(
            best_model_params['pi'],
            save_path=f"results/figures/{output_prefix}_mixture_weights.png"
        )
        print("  ✓ Mixture weights plot saved")
        
        # Categorical probabilities
        plot_categorical_probabilities(
            best_model_params['theta'],
            categories=categories,
            variable_names=variable_names,
            max_variables=12,
            save_path=f"results/figures/{output_prefix}_categorical_probs.png"
        )
        print("  ✓ Categorical probabilities plot saved")
        
        # Class assignments
        plot_class_assignments(
            true_labels=None,
            predicted_labels=predicted_labels,
            save_path=f"results/figures/{output_prefix}_class_assignments.png"
        )
        print("  ✓ Class assignments plot saved")
        
        # Posterior uncertainty
        plot_posterior_uncertainty(
            gamma=gamma,
            predicted_labels=predicted_labels,
            save_path=f"results/figures/{output_prefix}_posterior_uncertainty.png"
        )
        print("  ✓ Posterior uncertainty plot saved")
    
    # ========================================================================
    # STEP 6: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Optimal number of classes: {best_K}")
    print(f"Mixture weights: {best_model_params['pi']}")
    print(f"Log-likelihood: {best_model_params['log_likelihood']:.4f}")
    print(f"\nAll results saved to: results/{output_prefix}_*")
    print("="*70 + "\n")
    
    return selector, final_model


def generate_and_analyze_synthetic_data(n_samples: int = 1000,
                                       K_true: int = 3,
                                       n_variables: int = 20,
                                       n_categories: int = 3,
                                       output_prefix: str = "synthetic",
                                       random_state: int = 42):
    """
    Generate synthetic data and run analysis (useful for testing).
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    K_true : int, default=3
        True number of latent classes
    n_variables : int, default=20
        Number of variables
    n_categories : int, default=3
        Number of categories per variable
    output_prefix : str, default="synthetic"
        Prefix for output files
    random_state : int, default=42
        Random seed
    """
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATA")
    print("="*70)
    
    create_directories()
    
    # Generate data
    data_path = f"data/{output_prefix}_data.csv"
    true_params = generate_simple_scenario(
        n=n_samples,
        filepath=data_path,
        K=K_true,
        m=n_variables,
        C=n_categories,
        random_state=random_state
    )
    
    print(f"  ✓ Synthetic data generated with K={K_true}")
    print(f"  ✓ True mixture weights: {true_params['pi']}")
    
    # Run analysis
    K_range = list(range(1, K_true + 4))  # Test around true K
    
    selector, model = run_analysis(
        data_path=data_path,
        K_range=K_range,
        output_prefix=output_prefix,
        random_state=random_state
    )
    
# Additional validation for synthetic data
    print("\n" + "="*70)
    print("VALIDATION (Synthetic Data)")
    print("="*70)
    
    best_model_params = selector.get_best_model()
    
    # Check if correct K was selected
    if selector.best_K == K_true:
        print("✓ Correct number of classes selected!")
    else:
        print(f"✗ Selected K={selector.best_K}, True K={K_true}")
    
    # Only compare parameters if K matches
    if selector.best_K == K_true:
        # Compare mixture weights
        estimated_pi = best_model_params['pi']
        pi_mae = np.mean(np.abs(estimated_pi - true_params['pi']))
        pi_rmse = np.sqrt(np.mean((estimated_pi - true_params['pi'])**2))
        
        print(f"\nMixture Weights:")
        print(f"  True π:      {np.array2string(true_params['pi'], precision=4, suppress_small=True)}")
        print(f"  Estimated π: {np.array2string(estimated_pi, precision=4, suppress_small=True)}")
        print(f"  Mean Absolute Error (MAE):  {pi_mae:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {pi_rmse:.6f}")
        
        # Compare categorical probabilities (theta)
        estimated_theta = best_model_params['theta']
        true_theta = true_params['theta']
        
        theta_mae = np.mean(np.abs(estimated_theta - true_theta))
        theta_rmse = np.sqrt(np.mean((estimated_theta - true_theta)**2))
        
        print(f"\nCategorical Probabilities (θ_rkc):")
        print(f"  Mean Absolute Error (MAE):  {theta_mae:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {theta_rmse:.6f}")
        
        # Detailed breakdown by class
        print(f"\nPer-Class θ Errors:")
        for k in range(K_true):
            class_mae = np.mean(np.abs(estimated_theta[k] - true_theta[k]))
            print(f"  Class {k}: MAE = {class_mae:.6f}")
    else:
        print(f"\nParameter Comparison:")
        print(f"  ✗ Cannot compare parameters: K mismatch (selected={selector.best_K}, true={K_true})")
        print(f"  True K has {K_true} classes, selected model has {selector.best_K} classes")
    
    # Save true parameters for reference
    true_params_path = f"results/{output_prefix}_true_params.npz"
    np.savez(
        true_params_path,
        pi=true_params['pi'],
        theta=true_params['theta']
    )
    print(f"\n  ✓ True parameters saved to: {true_params_path}")
    
    print("="*70 + "\n")
    
    return selector, model, true_params


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Latent Class Model Analysis Pipeline"
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        help='Path to CSV data file (if not provided, generates synthetic data)'
    )
    parser.add_argument(
        '--k-range',
        type=int,
        nargs='+',
        default=None,
        help='Range of K values to test (e.g., --k-range 1 2 3 4 5)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='analysis',
        help='Prefix for output files (default: analysis)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=200,
        help='Maximum EM iterations (default: 200)'
    )
    parser.add_argument(
        '--n-init',
        type=int,
        default=10,
        help='Number of random initializations (default: 10)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs, -1 for all CPUs (default: -1)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--generate-synthetic',
        action='store_true',
        help='Generate and analyze synthetic data'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data (default: 1000)'
    )
    parser.add_argument(
        '--k-true',
        type=int,
        default=3,
        help='True K for synthetic data (default: 3)'
    )
    
    args = parser.parse_args()
    
    if args.generate_synthetic:
        # Generate and analyze synthetic data
        generate_and_analyze_synthetic_data(
            n_samples=args.n_samples,
            K_true=args.k_true,
            output_prefix=args.output_prefix,
            random_state=args.random_state
        )
    elif args.data:
        # Analyze provided data
        run_analysis(
            data_path=args.data,
            K_range=args.k_range,
            output_prefix=args.output_prefix,
            max_iter=args.max_iter,
            n_init=args.n_init,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            generate_all_plots=not args.no_plots
        )
    else:
        print("Error: Please provide --data <path> or use --generate-synthetic")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()