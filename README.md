# Latent Class Model Analysis Pipeline

End-to-end software for latent class analysis of multivariate categorical data with automatic BIC-based model selection.

**Author:** Siyuan Tang | **Course:** STATS 607, University of Michigan, Ann Arbor | **Date:** December 2025

---

## Overview

This package implements a **latent class model** for multivariate categorical data with:
- Automatic K selection via BIC (K = # latent classes)
- Vectorized EM algorithm
- Parallel processing across initializations and K values
- Comprehensive simulation validation

**Methodology:** See `docs/methodology.pdf`

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
│   ├── run_simulations.py          # Run simulations
│   ├── analyze_results.py          # Generate plots
│   └── results/                    # Outputs
├── main.py                         # Main pipeline
├── docs/                           # Documents
│   ├── presentation.pdf
│   ├── methodology.pdf             
├── README.md                       # This file
└── report-Tang.md                  # Project report
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

Edit `simulation/run_simulations.py`:

```python
SIMULATION_CONFIG = {
    'M': 50,                        # Simulations per config
    'K_values': [2, 3, 4, 5],       # True K values
    'sample_sizes': [500, 1000, 2000, 3000, 5000, 8000],
    'max_iter': 200,
    'n_init': 5
}
```

---

## Key Features

### Optimizations
- Fully vectorized EM
- Parallel initialization across cores

### Outputs
- BIC curves and success rates
- Parameter estimation errors
- Confusion matrices across sample sizes
- Classification accuracy trends
- Computation time analysis

## Documentation

- **Methodology:** `docs/methodology.pdf`
- **Project Report:** `docs/report-Tang.md`
