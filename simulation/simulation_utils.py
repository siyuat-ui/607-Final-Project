"""
Utility functions for simulation studies.
Optimized with better parallelization and batch processing.
"""

import numpy as np
import time
from typing import Tuple, Dict, List
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dgp import LatentClassDGP
from src.latent_class_modeling import LatentClassModel
from src.model_selection import BICModelSelector


def generate_dataset(n: int,
                     K: int,
                     m: int,
                     C: int,
                     random_state: int) -> Tuple[np.ndarray, Dict]:
    """
    Generate a single synthetic dataset.
    
    Parameters
    ----------
    n : int
        Number of samples
    K : int
        True number of classes
    m : int
        Number of variables
    C : int
        Number of categories per variable
    random_state : int
        Random seed
        
    Returns
    -------
    X : np.ndarray, shape (n, m)
        Generated data
    true_params : dict
        True parameters (pi, theta)
    """
    categories = [C] * m
    dgp = LatentClassDGP(K=K, categories=categories, random_state=random_state)
    X, H = dgp.generate_data(n)
    true_params = dgp.get_true_parameters()
    
    return X, true_params


def run_bic_selection(X: np.ndarray,
                     categories: List[int],
                     K_range: List[int],
                     max_iter: int,
                     tol: float,
                     n_init: int,
                     random_state: int) -> Tuple[int, float, Dict]:
    """
    Run BIC model selection on a dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    categories : list of int
        Number of categories per variable
    K_range : list of int
        Range of K values to test
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of random initializations
    random_state : int
        Random seed
        
    Returns
    -------
    selected_K : int
        Selected number of classes
    computation_time : float
        Time in seconds
    bic_values : dict
        BIC values for all K
    """
    start_time = time.time()
    
    selector = BICModelSelector(
        K_range=K_range,
        categories=categories,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        n_jobs=1,  # No nested parallelization
        random_state=random_state
    )
    
    selector.fit(X, verbose=False, parallel_strategy='within_K')
    
    computation_time = time.time() - start_time
    
    results = selector.get_all_results()
    bic_values = {k: v for k, v in zip(results['K_range'], results['bic_values'])}
    
    return selector.best_K, computation_time, bic_values


def fit_model_with_true_K(X: np.ndarray,
                          K_true: int,
                          categories: List[int],
                          max_iter: int,
                          tol: float,
                          n_init: int,
                          random_state: int) -> Tuple[Dict, float]:
    """
    Fit model with true K (for parameter estimation study).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    K_true : int
        True number of classes
    categories : list of int
        Number of categories per variable
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of random initializations
    random_state : int
        Random seed
        
    Returns
    -------
    params : dict
        Estimated parameters
    computation_time : float
        Time in seconds
    """
    start_time = time.time()
    
    model = LatentClassModel(K=K_true, categories=categories, random_state=random_state)
    model.fit(X, max_iter=max_iter, tol=tol, n_init=n_init, n_jobs=1, verbose=False)
    
    computation_time = time.time() - start_time
    
    params = model.get_parameters()
    
    return params, computation_time


def compute_pi_errors(true_pi: np.ndarray,
                     estimated_pi: np.ndarray) -> Dict[str, float]:
    """
    Compute errors for mixture weights.
    
    Parameters
    ----------
    true_pi : np.ndarray
        True mixture weights
    estimated_pi : np.ndarray
        Estimated mixture weights
        
    Returns
    -------
    errors : dict
        MAE and RMSE
    """
    mae = np.mean(np.abs(true_pi - estimated_pi))
    rmse = np.sqrt(np.mean((true_pi - estimated_pi)**2))
    
    return {'mae': mae, 'rmse': rmse}


def compute_theta_errors(true_theta: np.ndarray,
                        estimated_theta: np.ndarray) -> Dict[str, float]:
    """
    Compute errors for categorical probabilities.
    
    Parameters
    ----------
    true_theta : np.ndarray, shape (K, m, C_max)
        True categorical probabilities
    estimated_theta : np.ndarray, shape (K, m, C_max)
        Estimated categorical probabilities
        
    Returns
    -------
    errors : dict
        MAE and RMSE
    """
    mae = np.mean(np.abs(true_theta - estimated_theta))
    rmse = np.sqrt(np.mean((true_theta - estimated_theta)**2))
    
    return {'mae': mae, 'rmse': rmse}


def run_single_bic_simulation(sim_id: int,
                              n: int,
                              K_true: int,
                              m: int,
                              C: int,
                              K_range: List[int],
                              max_iter: int,
                              tol: float,
                              n_init: int,
                              base_seed: int) -> Dict:
    """
    Run a single BIC selection simulation.
    
    Parameters
    ----------
    sim_id : int
        Simulation index
    n : int
        Sample size
    K_true : int
        True number of classes
    m : int
        Number of variables
    C : int
        Number of categories
    K_range : list of int
        Range of K to test
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of initializations
    base_seed : int
        Base random seed
        
    Returns
    -------
    result : dict
        Simulation results
    """
    seed = base_seed + sim_id
    
    # Generate data
    X, true_params = generate_dataset(n, K_true, m, C, random_state=seed)
    
    # Run BIC selection
    selected_K, comp_time, bic_values = run_bic_selection(
        X, [C] * m, K_range, max_iter, tol, n_init, random_state=seed
    )
    
    result = {
        'sim_id': sim_id,
        'n': n,
        'K_true': K_true,
        'selected_K': selected_K,
        'correct': int(selected_K == K_true),
        'over_selected': int(selected_K > K_true),
        'under_selected': int(selected_K < K_true),
        'computation_time': comp_time
    }
    
    # Add BIC values for each K
    for k in K_range:
        result[f'bic_K{k}'] = bic_values.get(k, np.nan)
    
    return result


def run_single_estimation_simulation(sim_id: int,
                                     n: int,
                                     K_true: int,
                                     m: int,
                                     C: int,
                                     max_iter: int,
                                     tol: float,
                                     n_init: int,
                                     base_seed: int) -> Dict:
    """
    Run a single parameter estimation simulation.
    
    Parameters
    ----------
    sim_id : int
        Simulation index
    n : int
        Sample size
    K_true : int
        True number of classes
    m : int
        Number of variables
    C : int
        Number of categories
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of initializations
    base_seed : int
        Base random seed
        
    Returns
    -------
    result : dict
        Simulation results
    """
    seed = base_seed + sim_id
    
    # Generate data
    X, true_params = generate_dataset(n, K_true, m, C, random_state=seed)
    
    # Fit model with true K
    estimated_params, comp_time = fit_model_with_true_K(
        X, K_true, [C] * m, max_iter, tol, n_init, random_state=seed
    )
    
    # Compute errors
    pi_errors = compute_pi_errors(true_params['pi'], estimated_params['pi'])
    theta_errors = compute_theta_errors(true_params['theta'], estimated_params['theta'])
    
    result = {
        'sim_id': sim_id,
        'n': n,
        'K_true': K_true,
        'pi_mae': pi_errors['mae'],
        'pi_rmse': pi_errors['rmse'],
        'theta_mae': theta_errors['mae'],
        'theta_rmse': theta_errors['rmse'],
        'converged': int(estimated_params['converged']),
        'n_iterations': estimated_params['n_iterations'],
        'computation_time': comp_time
    }
    
    return result


def run_batch_bic_simulations(sim_ids: List[int],
                              n: int,
                              K_true: int,
                              m: int,
                              C: int,
                              K_range: List[int],
                              max_iter: int,
                              tol: float,
                              n_init: int,
                              base_seed: int) -> List[Dict]:
    """
    Run a batch of BIC simulations to reduce overhead.
    
    Parameters
    ----------
    sim_ids : list of int
        List of simulation indices to run in this batch
    (other parameters same as run_single_bic_simulation)
        
    Returns
    -------
    results : list of dict
        List of simulation results
    """
    results = []
    for sim_id in sim_ids:
        result = run_single_bic_simulation(
            sim_id, n, K_true, m, C, K_range, 
            max_iter, tol, n_init, base_seed
        )
        results.append(result)
    return results


def run_batch_estimation_simulations(sim_ids: List[int],
                                     n: int,
                                     K_true: int,
                                     m: int,
                                     C: int,
                                     max_iter: int,
                                     tol: float,
                                     n_init: int,
                                     base_seed: int) -> List[Dict]:
    """
    Run a batch of parameter estimation simulations to reduce overhead.
    
    Parameters
    ----------
    sim_ids : list of int
        List of simulation indices to run in this batch
    (other parameters same as run_single_estimation_simulation)
        
    Returns
    -------
    results : list of dict
        List of simulation results
    """
    results = []
    for sim_id in sim_ids:
        result = run_single_estimation_simulation(
            sim_id, n, K_true, m, C, 
            max_iter, tol, n_init, base_seed
        )
        results.append(result)
    return results


def aggregate_bic_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple BIC selection simulations.
    
    Parameters
    ----------
    results_list : list of dict
        List of individual simulation results
        
    Returns
    -------
    aggregated : dict
        Aggregated statistics
    """
    n = results_list[0]['n']
    K_true = results_list[0]['K_true']
    M = len(results_list)
    
    success_rate = np.mean([r['correct'] for r in results_list])
    over_rate = np.mean([r['over_selected'] for r in results_list])
    under_rate = np.mean([r['under_selected'] for r in results_list])
    
    selected_Ks = [r['selected_K'] for r in results_list]
    mean_selected_K = np.mean(selected_Ks)
    std_selected_K = np.std(selected_Ks)
    
    mean_time = np.mean([r['computation_time'] for r in results_list])
    std_time = np.std([r['computation_time'] for r in results_list])
    
    aggregated = {
        'n': n,
        'K_true': K_true,
        'M': M,
        'success_rate': success_rate,
        'over_rate': over_rate,
        'under_rate': under_rate,
        'mean_selected_K': mean_selected_K,
        'std_selected_K': std_selected_K,
        'mean_time': mean_time,
        'std_time': std_time
    }
    
    return aggregated


def aggregate_estimation_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple parameter estimation simulations.
    
    Parameters
    ----------
    results_list : list of dict
        List of individual simulation results
        
    Returns
    -------
    aggregated : dict
        Aggregated statistics
    """
    n = results_list[0]['n']
    K_true = results_list[0]['K_true']
    M = len(results_list)
    
    pi_maes = [r['pi_mae'] for r in results_list]
    pi_rmses = [r['pi_rmse'] for r in results_list]
    theta_maes = [r['theta_mae'] for r in results_list]
    theta_rmses = [r['theta_rmse'] for r in results_list]
    
    convergence_rate = np.mean([r['converged'] for r in results_list])
    mean_iterations = np.mean([r['n_iterations'] for r in results_list])
    
    mean_time = np.mean([r['computation_time'] for r in results_list])
    std_time = np.std([r['computation_time'] for r in results_list])
    
    aggregated = {
        'n': n,
        'K_true': K_true,
        'M': M,
        'pi_mae_mean': np.mean(pi_maes),
        'pi_mae_std': np.std(pi_maes),
        'pi_rmse_mean': np.mean(pi_rmses),
        'pi_rmse_std': np.std(pi_rmses),
        'theta_mae_mean': np.mean(theta_maes),
        'theta_mae_std': np.std(theta_maes),
        'theta_rmse_mean': np.mean(theta_rmses),
        'theta_rmse_std': np.std(theta_rmses),
        'convergence_rate': convergence_rate,
        'mean_iterations': mean_iterations,
        'mean_time': mean_time,
        'std_time': std_time
    }
    
    return aggregated

def run_single_estimation_simulation_with_labels(sim_id: int,
                                                 n: int,
                                                 K_true: int,
                                                 m: int,
                                                 C: int,
                                                 max_iter: int,
                                                 tol: float,
                                                 n_init: int,
                                                 base_seed: int) -> Dict:
    """
    Run a single parameter estimation simulation with true label tracking.
    
    Parameters
    ----------
    sim_id : int
        Simulation index
    n : int
        Sample size
    K_true : int
        True number of classes
    m : int
        Number of variables
    C : int
        Number of categories
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_init : int
        Number of initializations
    base_seed : int
        Base random seed
        
    Returns
    -------
    result : dict
        Simulation results including confusion matrix
    """
    seed = base_seed + sim_id
    
    # Generate data (keep true labels H)
    categories = [C] * m
    dgp = LatentClassDGP(K=K_true, categories=categories, random_state=seed)
    X, H = dgp.generate_data(n)
    true_params = dgp.get_true_parameters()
    
    # Fit model with true K
    model = LatentClassModel(K=K_true, categories=categories, random_state=seed)
    model.fit(X, max_iter=max_iter, tol=tol, n_init=n_init, n_jobs=1, verbose=False)
    
    # Get computation time
    estimated_params = model.get_parameters()
    
    # Predict labels
    predicted_labels = model.predict(X)
    
    # Compute confusion matrix
    confusion = np.zeros((K_true, K_true), dtype=int)
    for true_label, pred_label in zip(H, predicted_labels):
        confusion[true_label, pred_label] += 1
    
    # Compute accuracy
    accuracy = np.sum(H == predicted_labels) / n
    
    # Compute errors
    pi_errors = compute_pi_errors(true_params['pi'], estimated_params['pi'])
    theta_errors = compute_theta_errors(true_params['theta'], estimated_params['theta'])
    
    result = {
        'sim_id': sim_id,
        'n': n,
        'K_true': K_true,
        'pi_mae': pi_errors['mae'],
        'pi_rmse': pi_errors['rmse'],
        'theta_mae': theta_errors['mae'],
        'theta_rmse': theta_errors['rmse'],
        'accuracy': accuracy,
        'converged': int(estimated_params['converged']),
        'n_iterations': estimated_params['n_iterations'],
        'computation_time': 0  # Will be computed in the wrapper
    }
    
    # Add confusion matrix as separate keys
    for i in range(K_true):
        for j in range(K_true):
            result[f'conf_{i}_{j}'] = confusion[i, j]
    
    return result


def aggregate_estimation_results_with_confusion(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple parameter estimation simulations including confusion matrix.
    
    Parameters
    ----------
    results_list : list of dict
        List of individual simulation results
        
    Returns
    -------
    aggregated : dict
        Aggregated statistics including mean confusion matrix
    """
    n = results_list[0]['n']
    K_true = results_list[0]['K_true']
    M = len(results_list)
    
    pi_maes = [r['pi_mae'] for r in results_list]
    pi_rmses = [r['pi_rmse'] for r in results_list]
    theta_maes = [r['theta_mae'] for r in results_list]
    theta_rmses = [r['theta_rmse'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    
    convergence_rate = np.mean([r['converged'] for r in results_list])
    mean_iterations = np.mean([r['n_iterations'] for r in results_list])
    
    mean_time = np.mean([r['computation_time'] for r in results_list])
    std_time = np.std([r['computation_time'] for r in results_list])
    
    # Aggregate confusion matrix
    confusion_sum = np.zeros((K_true, K_true))
    for result in results_list:
        for i in range(K_true):
            for j in range(K_true):
                confusion_sum[i, j] += result[f'conf_{i}_{j}']
    
    # Mean confusion matrix (normalized by M simulations)
    confusion_mean = confusion_sum / M
    
    # Normalize row-wise to get conditional probabilities
    confusion_normalized = confusion_mean / confusion_mean.sum(axis=1, keepdims=True)
    
    aggregated = {
        'n': n,
        'K_true': K_true,
        'M': M,
        'pi_mae_mean': np.mean(pi_maes),
        'pi_mae_std': np.std(pi_maes),
        'pi_rmse_mean': np.mean(pi_rmses),
        'pi_rmse_std': np.std(pi_rmses),
        'theta_mae_mean': np.mean(theta_maes),
        'theta_mae_std': np.std(theta_maes),
        'theta_rmse_mean': np.mean(theta_rmses),
        'theta_rmse_std': np.std(theta_rmses),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'convergence_rate': convergence_rate,
        'mean_iterations': mean_iterations,
        'mean_time': mean_time,
        'std_time': std_time
    }
    
    # Add normalized confusion matrix entries
    for i in range(K_true):
        for j in range(K_true):
            aggregated[f'conf_norm_{i}_{j}'] = confusion_normalized[i, j]
    
    return aggregated