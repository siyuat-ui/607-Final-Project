# Latent Class Model Analysis Pipeline

End-to-end software for latent class analysis of multivariate categorical data with automatic BIC-based model selection.

**Author:** Siyuan Tang | **Course:** STATS 607, University of Michigan, Ann Arbor | **Date:** December 2025

---

## Overview

This package implements a **latent class model for multivariate categorical data** with:
- Automatic model selection via BIC
- Vectorized EM algorithm
- Parallel processing across initializations and $K$ values ($K$ = # latent classes)
- Comprehensive simulation validation
- Makefile commands

**Methodology:** See `docs/methodology.pdf`

For a self-contained example, see `demo/latent_class_demo.ipynb`. We recommend reading `docs/methodology.pdf` before diving into the notebook.

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/siyuat-ui/607-Final-Project.git
cd 607-Final-Project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Analyze Data and Fit the Model

```bash
# Analyze CSV file (suppose the user's dataset is `data/mydata.csv`)
python main.py --data data/mydata.csv --output-prefix my_analysis

# Generate synthetic data and fit the model
python main.py --generate-synthetic --n-samples 2000 --k-true 3
```

### Run Simulations

```bash
# Full simulation study
python simulation/run_simulations.py

# Generate visualizations
python simulation/analyze_results.py
```

Please note that it takes 40-50 minutes if you run the default simulation configuration.

### Makefile Commands

We also provide a Makefile to simplify common tasks. 

```bash
# Show help
make help

# Run full simulation study (with confirmation prompt)
make simulate

# Clean simulation results
make clean-sim

# Analyze user data
make analyze DATA=data/mydata.csv PREFIX=my_analysis

# Generate synthetic data
make synthetic N=2000 K=3

# Clean results matching prefix
make clean-results PREFIX=my_analysis
make clean-results PREFIX=analysis

# Clean everything
make clean-all
```

If you plan to use Makefile commands, we recommend that you first display all available commands with usage examples by running `make help`.

---

## Project Structure

```
607-Final-Project/
├── src/                            # Core algorithms
│   ├── latent_class_modeling.py    # Vectorized EM algorithm
│   ├── model_selection.py          # BIC-based K selection
│   ├── dgp.py                      # Data generation
│   ├── data_loader.py              # Data loading
│   ├── visualization.py            # Plotting
│   └── utils.py                    # Helper functions
├── simulation/                     # Validation studies
│   ├── simulation_utils.py         # Helper functions for simulation studies
│   ├── run_simulations.py          # Run simulations
│   ├── analyze_results.py          # Generate plots
│   └── results/                    # Outputs
├── main.py                         # Main pipeline
├── docs/                           # Documents
│   ├── presentation.pdf
│   └── methodology.pdf             
├── demo/
│   └── latent_class_demo.ipynb     # A self-contained example
├── README.md                       # This file
├── report-Tang.pdf                 # Project report
└── Makefile                        # Makefile commands
```

---

## Usage Examples

### Example 1: Basic Analysis
```python
from src.model_selection import BICModelSelector
from src.data_loader import load_csv_data

X, categories, var_names = load_csv_data('data/mydata.csv')
selector = BICModelSelector(K_range=[1,2,3,4,5], categories=categories)
selector.fit(X)
print(f"Best K: {selector.best_K}")
```

### Example 2: Fit Model
```python
from src.latent_class_modeling import LatentClassModel

model = LatentClassModel(K=3, categories=categories)
model.fit(X, n_init=10)
params = model.get_parameters()
```

---

## Simulation Configuration

Edit `simulation/run_simulations.py` if you need to:

```python
SIMULATION_CONFIG = {

'M': 50, # Simulations per configuration

'm': 20, # Number of variables

'C': 2, # Categories per variable

'K_values': [2, 3, 4, 5], # True K values to test

'sample_sizes': [500, 1000, 2000, 3000, 5000],

'K_range_margin': 2, # Test K_true ± margin

'K_min': 1, # Minimum K to test

'K_max': 10, # Maximum K to test

'max_iter': 200, # EM max iterations

'tol': 1e-6, # EM convergence tolerance

'n_init': 5, # Number of random initializations

'base_seed': 42 # Base random seed

}
```

---

## Key Features

- Automatic model fitting and selection for user-provided data

### Optimizations
- Fully vectorized EM
- Parallel initialization across cores

### Outputs
- BIC curves and success rates
- Parameter estimation errors
- Confusion matrices across sample sizes
- Classification accuracy trends
- Computation time analysis

---

## Documentation

- **Methodology:** `docs/methodology.pdf`
- **A Self-contained Example:** `demo/latent_class_demo.ipynb`
- **Project Report:** `docs/report-Tang.pdf`
- **In-class Presentation Slides:** `docs/presentation.pdf`