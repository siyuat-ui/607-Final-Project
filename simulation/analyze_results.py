"""
Analyze and visualize simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import json


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_simulation_config() -> dict:
    """Load simulation configuration."""
    config_path = "simulation/results/simulation_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def plot_bic_success_rates(save_figures: bool = True):
    """
    Plot BIC selection success rates vs sample size.
    
    Creates one plot per K_true showing success rate vs sample size.
    """
    print("\n" + "="*70)
    print("Plotting BIC Success Rates")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    # Create figure with subplots
    n_plots = len(K_values)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, K_true in enumerate(K_values):
        ax = axes[idx]
        
        # Load results
        results_path = f"simulation/results/bic_selection/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        # Plot success rate
        ax.plot(df['n'], df['success_rate'], marker='o', linewidth=2.5, 
                markersize=8, label='Success Rate', color='steelblue')
        ax.plot(df['n'], df['over_rate'], marker='s', linewidth=2, 
                markersize=6, label='Over-selection', color='coral', linestyle='--')
        ax.plot(df['n'], df['under_rate'], marker='^', linewidth=2, 
                markersize=6, label='Under-selection', color='lightgreen', linestyle='--')
        
        # Add horizontal line at 100%
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'BIC Selection Performance (K_true = {K_true})', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # Print summary
        print(f"\nK_true = {K_true}:")
        print(f"  Sample sizes: {df['n'].tolist()}")
        print(f"  Success rates: {df['success_rate'].tolist()}")
        print(f"  Final success rate (n={df['n'].iloc[-1]}): {df['success_rate'].iloc[-1]:.3f}")
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/bic_success_rates_by_K.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_bic_success_rates_combined(save_figures: bool = True):
    """
    Plot BIC selection success rates for all K values on one plot.
    """
    print("\n" + "="*70)
    print("Plotting Combined BIC Success Rates")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("husl", len(K_values))
    
    for idx, K_true in enumerate(K_values):
        # Load results
        results_path = f"simulation/results/bic_selection/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        # Plot success rate
        ax.plot(df['n'], df['success_rate'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
    
    # Add horizontal line at 100%
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, 
               label='Perfect Selection')
    
    ax.set_xlabel('Sample Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
    ax.set_title('BIC Selection Success Rate vs Sample Size', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, title='True K', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/bic_success_rates_combined.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_parameter_estimation_errors(save_figures: bool = True):
    """
    Plot parameter estimation errors (MAE and RMSE) vs sample size.
    
    Creates separate plots for π and θ.
    """
    print("\n" + "="*70)
    print("Plotting Parameter Estimation Errors")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    colors = sns.color_palette("husl", len(K_values))
    
    # ========================================================================
    # Plot for π (mixture weights)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE
    ax = axes[0]
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['pi_mae_mean'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['pi_mae_mean'] - df['pi_mae_std'],
                        df['pi_mae_mean'] + df['pi_mae_std'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Mixture Weights (π) - Mean Absolute Error', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='True K')
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1]
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['pi_rmse_mean'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['pi_rmse_mean'] - df['pi_rmse_std'],
                        df['pi_rmse_mean'] + df['pi_rmse_std'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Mixture Weights (π) - Root Mean Squared Error', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='True K')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/pi_estimation_errors.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # ========================================================================
    # Plot for θ (categorical probabilities)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE
    ax = axes[0]
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['theta_mae_mean'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['theta_mae_mean'] - df['theta_mae_std'],
                        df['theta_mae_mean'] + df['theta_mae_std'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Categorical Probabilities (θ) - Mean Absolute Error', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='True K')
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1]
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['theta_rmse_mean'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['theta_rmse_mean'] - df['theta_rmse_std'],
                        df['theta_rmse_mean'] + df['theta_rmse_std'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Categorical Probabilities (θ) - Root Mean Squared Error', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='True K')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/theta_estimation_errors.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_computation_time_analysis(save_figures: bool = True):
    """
    Plot computation time analysis.
    """
    print("\n" + "="*70)
    print("Plotting Computation Time Analysis")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    colors = sns.color_palette("husl", len(K_values))
    
    # ========================================================================
    # BIC Selection Time
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/bic_selection/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['mean_time'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['mean_time'] - df['std_time'],
                        df['mean_time'] + df['std_time'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Computation Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('BIC Selection Computation Time vs Sample Size', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, title='True K', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/bic_computation_time.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # ========================================================================
    # Parameter Estimation Time
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, K_true in enumerate(K_values):
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        ax.plot(df['n'], df['mean_time'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['mean_time'] - df['std_time'],
                        df['mean_time'] + df['std_time'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Computation Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Parameter Estimation Computation Time vs Sample Size', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, title='True K', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/estimation_computation_time.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def create_summary_tables(save_tables: bool = True):
    """
    Create comprehensive summary tables of simulation results.
    """
    print("\n" + "="*70)
    print("Creating Summary Tables")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    # ========================================================================
    # BIC Selection Summary
    # ========================================================================
    print("\nBIC Selection Summary:")
    
    bic_summary = []
    for K_true in K_values:
        results_path = f"simulation/results/bic_selection/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        for _, row in df.iterrows():
            bic_summary.append({
                'K_true': K_true,
                'n': row['n'],
                'success_rate': row['success_rate'],
                'over_rate': row['over_rate'],
                'under_rate': row['under_rate'],
                'mean_selected_K': row['mean_selected_K'],
                'mean_time_sec': row['mean_time']
            })
    
    df_bic = pd.DataFrame(bic_summary)
    print(df_bic.to_string(index=False))
    
    if save_tables:
        table_path = "simulation/results/bic_summary_table.csv"
        df_bic.to_csv(table_path, index=False)
        print(f"\nBIC summary table saved to: {table_path}")
    
    # ========================================================================
    # Parameter Estimation Summary
    # ========================================================================
    print("\n" + "="*70)
    print("\nParameter Estimation Summary:")
    
    est_summary = []
    for K_true in K_values:
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        for _, row in df.iterrows():
            est_summary.append({
                'K_true': K_true,
                'n': row['n'],
                'pi_mae': row['pi_mae_mean'],
                'pi_rmse': row['pi_rmse_mean'],
                'theta_mae': row['theta_mae_mean'],
                'theta_rmse': row['theta_rmse_mean'],
                'convergence_rate': row['convergence_rate'],
                'mean_time_sec': row['mean_time']
            })
    
    df_est = pd.DataFrame(est_summary)
    print(df_est.to_string(index=False))
    
    if save_tables:
        table_path = "simulation/results/estimation_summary_table.csv"
        df_est.to_csv(table_path, index=False)
        print(f"\nEstimation summary table saved to: {table_path}")

def plot_confusion_matrices_by_sample_size(save_figures: bool = True):
    """
    Plot confusion matrices showing how classification improves with sample size.
    
    Creates one figure per K_true with subplots for each sample size (Option C).
    """
    print("\n" + "="*70)
    print("Plotting Confusion Matrices by Sample Size")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    sample_sizes = config['sample_sizes']
    
    for K_true in K_values:
        # Load results for this K
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        # Create subplots - one per sample size
        n_sizes = len(sample_sizes)
        n_cols = min(5, n_sizes)  # Max 5 columns
        n_rows = int(np.ceil(n_sizes / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = np.array(axes).flatten() if n_sizes > 1 else [axes]
        
        for idx, n in enumerate(sample_sizes):
            ax = axes[idx]
            
            # Get confusion matrix for this sample size
            row = df[df['n'] == n].iloc[0]
            
            # Reconstruct normalized confusion matrix
            conf_matrix = np.zeros((K_true, K_true))
            for i in range(K_true):
                for j in range(K_true):
                    conf_matrix[i, j] = row[f'conf_norm_{i}_{j}']
            
            # Plot heatmap
            sns.heatmap(conf_matrix, annot=True, fmt='.3f', cmap='Blues', 
                       vmin=0, vmax=1, cbar=True, ax=ax,
                       xticklabels=[f'{i}' for i in range(K_true)],
                       yticklabels=[f'{i}' for i in range(K_true)])
            
            ax.set_title(f'n = {n}\nAccuracy: {row["accuracy_mean"]:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=10)
            ax.set_ylabel('True Class', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_sizes, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Confusion Matrices for K_true = {K_true} (Row-Normalized)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_figures:
            save_path = f"simulation/results/figures/confusion_matrices_K{K_true}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Confusion matrices for K={K_true} saved to: {save_path}")
        
        plt.show()


def plot_classification_accuracy(save_figures: bool = True):
    """
    Plot classification accuracy vs sample size for all K values.
    """
    print("\n" + "="*70)
    print("Plotting Classification Accuracy vs Sample Size")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    colors = sns.color_palette("husl", len(K_values))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, K_true in enumerate(K_values):
        # Load results
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        # Plot accuracy with error bars
        ax.plot(df['n'], df['accuracy_mean'], marker='o', linewidth=2.5, 
                markersize=8, label=f'K = {K_true}', color=colors[idx])
        ax.fill_between(df['n'], 
                        df['accuracy_mean'] - df['accuracy_std'],
                        df['accuracy_mean'] + df['accuracy_std'],
                        alpha=0.2, color=colors[idx])
    
    ax.set_xlabel('Sample Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Classification Accuracy vs Sample Size', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, title='True K', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add horizontal line at perfect accuracy
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Perfect')
    
    plt.tight_layout()
    
    if save_figures:
        save_path = "simulation/results/figures/classification_accuracy_vs_n.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Classification accuracy plot saved to: {save_path}")
    
    plt.show()


def plot_per_class_accuracy(save_figures: bool = True):
    """
    Plot per-class accuracy (diagonal of confusion matrix) vs sample size.
    """
    print("\n" + "="*70)
    print("Plotting Per-Class Accuracy")
    print("="*70)
    
    # Load configuration
    config = load_simulation_config()
    K_values = config['K_values']
    
    for K_true in K_values:
        # Load results
        results_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df = pd.read_csv(results_path)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = sns.color_palette("husl", K_true)
        
        # Plot per-class accuracy (diagonal of confusion matrix)
        for class_idx in range(K_true):
            per_class_acc = df[f'conf_norm_{class_idx}_{class_idx}']
            ax.plot(df['n'], per_class_acc, marker='o', linewidth=2.5, 
                   markersize=8, label=f'Class {class_idx}', color=colors[class_idx])
        
        ax.set_xlabel('Sample Size (n)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Per-Class Accuracy vs Sample Size (K_true = {K_true})', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_figures:
            save_path = f"simulation/results/figures/per_class_accuracy_K{K_true}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Per-class accuracy for K={K_true} saved to: {save_path}")
        
        plt.show()


def generate_all_plots_and_tables():
    """Generate all plots and tables."""
    print("\n" + "="*70)
    print("GENERATING ALL VISUALIZATIONS AND TABLES")
    print("="*70)
    
    # Create figures directory
    Path("simulation/results/figures").mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_bic_success_rates(save_figures=True)
    plot_bic_success_rates_combined(save_figures=True)
    plot_parameter_estimation_errors(save_figures=True)
    plot_computation_time_analysis(save_figures=True)
    
    # NEW: Confusion matrix plots
    plot_confusion_matrices_by_sample_size(save_figures=True)
    plot_classification_accuracy(save_figures=True)
    plot_per_class_accuracy(save_figures=True)
    
    # Generate tables
    create_summary_tables(save_tables=True)
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("Figures saved in: simulation/results/figures/")
    print("Tables saved in: simulation/results/")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze and visualize simulation results"
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        choices=['bic', 'estimation', 'time', 'confusion', 'all'],
        default='all',
        help='Which plots to generate (default: all)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save figures'
    )
    
    args = parser.parse_args()
    
    save_figs = not args.no_save
    
    if args.plot == 'all':
        generate_all_plots_and_tables()
    else:
        if args.plot == 'bic':
            plot_bic_success_rates(save_figures=save_figs)
            plot_bic_success_rates_combined(save_figures=save_figs)
        elif args.plot == 'estimation':
            plot_parameter_estimation_errors(save_figures=save_figs)
        elif args.plot == 'time':
            plot_computation_time_analysis(save_figures=save_figs)
        elif args.plot == 'confusion':
            plot_confusion_matrices_by_sample_size(save_figures=save_figs)
            plot_classification_accuracy(save_figures=save_figs)
            plot_per_class_accuracy(save_figures=save_figs)
        
        create_summary_tables(save_tables=save_figs)


if __name__ == "__main__":
    main()