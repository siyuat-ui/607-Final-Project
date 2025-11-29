"""
Main script to run simulation studies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import argparse
import json

from simulation_utils import (
    run_single_bic_simulation,
    run_single_estimation_simulation,
    aggregate_bic_results,
    aggregate_estimation_results
)


# ============================================================================
# Simulation Configuration
# ============================================================================

SIMULATION_CONFIG = {
    'M': 100,                                           # Simulations per configuration
    'm': 20,                                            # Number of variables
    'C': 2,                                             # Categories per variable
    'K_values': [2, 3, 4, 5],                          # True K values to test
    'sample_sizes': [500, 1000, 1500, 2000, 3000, 5000, 6000, 8000],
    'K_range_test': list(range(1, 8)),                 # K range for BIC selection
    'max_iter': 1000,                                   # EM max iterations
    'tol': 1e-6,                                        # EM convergence tolerance
    'n_init': 5,                                        # Number of random initializations
    'base_seed': 42                                     # Base random seed
}


def create_directories():
    """Create necessary directories for simulation results."""
    directories = [
        'simulation/results',
        'simulation/results/bic_selection',
        'simulation/results/parameter_estimation',
        'simulation/results/figures',
        'simulation/results/raw_data'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_config(config: dict, filepath: str):
    """Save simulation configuration to JSON."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {filepath}")


def run_bic_study(config: dict, n_jobs: int = -1):
    """
    Run BIC selection study.
    
    Parameters
    ----------
    config : dict
        Simulation configuration
    n_jobs : int, default=-1
        Number of parallel jobs
    """
    print("\n" + "="*70)
    print("STUDY 1: BIC SELECTION ACCURACY")
    print("="*70)
    
    M = config['M']
    m = config['m']
    C = config['C']
    K_values = config['K_values']
    sample_sizes = config['sample_sizes']
    K_range_test = config['K_range_test']
    max_iter = config['max_iter']
    tol = config['tol']
    n_init = config['n_init']
    base_seed = config['base_seed']
    
    total_configs = len(K_values) * len(sample_sizes)
    print(f"Total configurations: {total_configs}")
    print(f"Simulations per configuration: {M}")
    print(f"Total simulation runs: {total_configs * M}")
    print(f"Parallelization: n_jobs={n_jobs}")
    
    overall_start = time.time()
    
    # Results storage
    all_results = []
    
    # Loop over each configuration
    for K_true in K_values:
        print(f"\n{'='*70}")
        print(f"Running simulations for K_true = {K_true}")
        print(f"{'='*70}")
        
        config_results = []
        
        for n in tqdm(sample_sizes, desc=f"K={K_true}"):
            # Run M simulations in parallel for this (K_true, n) configuration
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_bic_simulation)(
                    sim_id=sim_id,
                    n=n,
                    K_true=K_true,
                    m=m,
                    C=C,
                    K_range=K_range_test,
                    max_iter=max_iter,
                    tol=tol,
                    n_init=n_init,
                    base_seed=base_seed + K_true * 10000 + n
                )
                for sim_id in range(M)
            )
            
            # Save raw results
            df_raw = pd.DataFrame(results)
            raw_path = f"simulation/results/raw_data/bic_K{K_true}_n{n}_raw.csv"
            df_raw.to_csv(raw_path, index=False)
            
            # Aggregate results
            agg = aggregate_bic_results(results)
            config_results.append(agg)
            all_results.append(agg)
            
            # Print progress
            print(f"  n={n:5d}: Success rate = {agg['success_rate']:.3f}, "
                  f"Time = {agg['mean_time']:.2f}s ± {agg['std_time']:.2f}s")
        
        # Save aggregated results for this K
        df_config = pd.DataFrame(config_results)
        config_path = f"simulation/results/bic_selection/K{K_true}_results.csv"
        df_config.to_csv(config_path, index=False)
        print(f"\nResults for K={K_true} saved to: {config_path}")
    
    # Save all results
    df_all = pd.DataFrame(all_results)
    all_path = "simulation/results/bic_selection/all_results.csv"
    df_all.to_csv(all_path, index=False)
    
    overall_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"BIC Study completed in {overall_time/60:.2f} minutes")
    print(f"Results saved to: simulation/results/bic_selection/")
    print(f"{'='*70}")


def run_estimation_study(config: dict, n_jobs: int = -1):
    """
    Run parameter estimation study.
    
    Parameters
    ----------
    config : dict
        Simulation configuration
    n_jobs : int, default=-1
        Number of parallel jobs
    """
    print("\n" + "="*70)
    print("STUDY 2: PARAMETER ESTIMATION ACCURACY")
    print("="*70)
    
    M = config['M']
    m = config['m']
    C = config['C']
    K_values = config['K_values']
    sample_sizes = config['sample_sizes']
    max_iter = config['max_iter']
    tol = config['tol']
    n_init = config['n_init']
    base_seed = config['base_seed']
    
    total_configs = len(K_values) * len(sample_sizes)
    print(f"Total configurations: {total_configs}")
    print(f"Simulations per configuration: {M}")
    print(f"Total simulation runs: {total_configs * M}")
    print(f"Parallelization: n_jobs={n_jobs}")
    
    overall_start = time.time()
    
    # Results storage
    all_results = []
    
    # Loop over each configuration
    for K_true in K_values:
        print(f"\n{'='*70}")
        print(f"Running simulations for K_true = {K_true}")
        print(f"{'='*70}")
        
        config_results = []
        
        for n in tqdm(sample_sizes, desc=f"K={K_true}"):
            # Run M simulations in parallel for this (K_true, n) configuration
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_estimation_simulation)(
                    sim_id=sim_id,
                    n=n,
                    K_true=K_true,
                    m=m,
                    C=C,
                    max_iter=max_iter,
                    tol=tol,
                    n_init=n_init,
                    base_seed=base_seed + K_true * 10000 + n + 100000
                )
                for sim_id in range(M)
            )
            
            # Save raw results
            df_raw = pd.DataFrame(results)
            raw_path = f"simulation/results/raw_data/estimation_K{K_true}_n{n}_raw.csv"
            df_raw.to_csv(raw_path, index=False)
            
            # Aggregate results
            agg = aggregate_estimation_results(results)
            config_results.append(agg)
            all_results.append(agg)
            
            # Print progress
            print(f"  n={n:5d}: π MAE = {agg['pi_mae_mean']:.4f}, "
                  f"θ MAE = {agg['theta_mae_mean']:.4f}, "
                  f"Time = {agg['mean_time']:.2f}s ± {agg['std_time']:.2f}s")
        
        # Save aggregated results for this K
        df_config = pd.DataFrame(config_results)
        config_path = f"simulation/results/parameter_estimation/K{K_true}_results.csv"
        df_config.to_csv(config_path, index=False)
        print(f"\nResults for K={K_true} saved to: {config_path}")
    
    # Save all results
    df_all = pd.DataFrame(all_results)
    all_path = "simulation/results/parameter_estimation/all_results.csv"
    df_all.to_csv(all_path, index=False)
    
    overall_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"Estimation Study completed in {overall_time/60:.2f} minutes")
    print(f"Results saved to: simulation/results/parameter_estimation/")
    print(f"{'='*70}")


def main():
    """Main entry point for simulation studies."""
    parser = argparse.ArgumentParser(
        description="Run simulation studies for latent class models"
    )
    
    parser.add_argument(
        '--study',
        type=str,
        choices=['bic', 'estimation', 'both'],
        default='both',
        help='Which study to run (default: both)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (default: -1, use all cores)'
    )
    parser.add_argument(
        '--M',
        type=int,
        default=None,
        help='Override number of simulations per configuration'
    )
    parser.add_argument(
        '--sample-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Override sample sizes to test'
    )
    parser.add_argument(
        '--K-values',
        type=int,
        nargs='+',
        default=None,
        help='Override K values to test'
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Load and possibly override configuration
    config = SIMULATION_CONFIG.copy()
    
    if args.M is not None:
        config['M'] = args.M
        print(f"Override: M = {args.M}")
    
    if args.sample_sizes is not None:
        config['sample_sizes'] = args.sample_sizes
        print(f"Override: sample_sizes = {args.sample_sizes}")
    
    if args.K_values is not None:
        config['K_values'] = args.K_values
        print(f"Override: K_values = {args.K_values}")
    
    # Save configuration
    save_config(config, "simulation/results/simulation_config.json")
    
    # Run studies
    start_time = time.time()
    
    if args.study in ['bic', 'both']:
        run_bic_study(config, n_jobs=args.n_jobs)
    
    if args.study in ['estimation', 'both']:
        run_estimation_study(config, n_jobs=args.n_jobs)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETED")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved in: simulation/results/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()