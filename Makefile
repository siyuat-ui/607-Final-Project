# Makefile for Latent Class Model Project
# ========================================

.PHONY: help simulate clean-sim analyze synthetic clean-results clean-all

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Help
# ============================================================================

help:
	@echo "=========================================="
	@echo "Latent Class Model Project - Makefile"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make simulate"
	@echo "      Run full simulation study (run_simulations.py + analyze_results.py)"
	@echo "      ⚠️  Takes approximately 40-50 minutes with default configuration"
	@echo ""
	@echo "  make clean-sim"
	@echo "      Delete simulation/results/ folder and all contents"
	@echo ""
	@echo "  make analyze DATA=path/to/data.csv PREFIX=my_prefix"
	@echo "      Analyze user CSV file"
	@echo "      Example: make analyze DATA=data/mydata.csv PREFIX=my_analysis"
	@echo "      Both DATA and PREFIX are REQUIRED"
	@echo ""
	@echo "  make synthetic N=2000 K=3"
	@echo "      Generate synthetic data and fit model"
	@echo "      Example: make synthetic N=2000 K=3"
	@echo "      Both N and K are REQUIRED"
	@echo ""
	@echo "  make clean-results PREFIX=my_prefix"
	@echo "      Delete all results matching the prefix in results/ folder"
	@echo "      Example: make clean-results PREFIX=my_analysis"
	@echo "      PREFIX is REQUIRED"
	@echo ""
	@echo "  make clean-all"
	@echo "      Delete both simulation/results/ and results/ folders"
	@echo ""
	@echo "  make help"
	@echo "      Show this help message"
	@echo ""
	@echo "=========================================="

# ============================================================================
# Simulation Study
# ============================================================================

simulate:
	@echo "=========================================="
	@echo "Running Full Simulation Study"
	@echo "=========================================="
	@echo ""
	@echo "⚠️  WARNING: This will take approximately 40-50 minutes"
	@echo "    with the default configuration (M=50, 5 sample sizes, 4 K values)"
	@echo ""
	@read -p "Do you want to continue? [y/N] " response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		echo ""; \
		echo "Step 1: Running simulations..."; \
		python simulation/run_simulations.py; \
		echo ""; \
		echo "Step 2: Analyzing results..."; \
		python simulation/analyze_results.py; \
		echo ""; \
		echo "✓ Simulation study complete!"; \
		echo "  Results saved in: simulation/results/"; \
	else \
		echo ""; \
		echo "Simulation cancelled."; \
	fi

# ============================================================================
# Clean Simulation Results
# ============================================================================

clean-sim:
	@echo "=========================================="
	@echo "Cleaning Simulation Results"
	@echo "=========================================="
	@if [ -d "simulation/results" ]; then \
		rm -rf simulation/results; \
		echo "✓ Deleted: simulation/results/"; \
	else \
		echo "ℹ  simulation/results/ does not exist"; \
	fi
	@echo ""

# ============================================================================
# Analyze User Data
# ============================================================================

analyze:
	@if [ -z "$(DATA)" ] || [ -z "$(PREFIX)" ]; then \
		echo ""; \
		echo "❌ ERROR: Both DATA and PREFIX are required"; \
		echo ""; \
		echo "Usage:"; \
		echo "  make analyze DATA=path/to/data.csv PREFIX=my_prefix"; \
		echo ""; \
		echo "Example:"; \
		echo "  make analyze DATA=data/mydata.csv PREFIX=my_analysis"; \
		echo ""; \
		exit 1; \
	fi
	@if [ ! -f "$(DATA)" ]; then \
		echo ""; \
		echo "❌ ERROR: File not found: $(DATA)"; \
		echo ""; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "Analyzing User Data"
	@echo "=========================================="
	@echo "Data file: $(DATA)"
	@echo "Output prefix: $(PREFIX)"
	@echo ""
	python main.py --data $(DATA) --output-prefix $(PREFIX)
	@echo ""
	@echo "✓ Analysis complete!"
	@echo "  Results saved in: results/$(PREFIX)_*"

# ============================================================================
# Generate Synthetic Data
# ============================================================================

synthetic:
	@if [ -z "$(N)" ] || [ -z "$(K)" ]; then \
		echo ""; \
		echo "❌ ERROR: Both N and K are required"; \
		echo ""; \
		echo "Usage:"; \
		echo "  make synthetic N=sample_size K=num_classes"; \
		echo ""; \
		echo "Example:"; \
		echo "  make synthetic N=2000 K=3"; \
		echo ""; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "Generating Synthetic Data"
	@echo "=========================================="
	@echo "Sample size: $(N)"
	@echo "Number of classes: $(K)"
	@echo ""
	python main.py --generate-synthetic --n-samples $(N) --k-true $(K)
	@echo ""
	@echo "✓ Synthetic data analysis complete!"
	@echo "  Results saved in: results/synthetic_*"

# ============================================================================
# Clean Results (User Data or Synthetic)
# ============================================================================

clean-results:
	@if [ -z "$(PREFIX)" ]; then \
		echo ""; \
		echo "❌ ERROR: PREFIX is required"; \
		echo ""; \
		echo "Usage:"; \
		echo "  make clean-results PREFIX=my_prefix"; \
		echo ""; \
		echo "Example:"; \
		echo "  make clean-results PREFIX=my_analysis"; \
		echo "  make clean-results PREFIX=synthetic"; \
		echo ""; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "Cleaning Results with Prefix: $(PREFIX)"
	@echo "=========================================="
	@if [ -d "results" ]; then \
		echo "Deleting files matching: results/$(PREFIX)_*"; \
		rm -f results/$(PREFIX)_*; \
		if [ -d "results/figures" ]; then \
			echo "Deleting files matching: results/figures/$(PREFIX)_*"; \
			rm -f results/figures/$(PREFIX)_*; \
		fi; \
		echo "✓ Deleted all files matching prefix: $(PREFIX)"; \
	else \
		echo "ℹ  results/ folder does not exist"; \
	fi
	@echo ""

# ============================================================================
# Clean All Results
# ============================================================================

clean-all:
	@echo "=========================================="
	@echo "Cleaning All Results"
	@echo "=========================================="
	@read -p "⚠️  This will delete simulation/results/ and results/. Continue? [y/N] " response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		echo ""; \
		if [ -d "simulation/results" ]; then \
			rm -rf simulation/results; \
			echo "✓ Deleted: simulation/results/"; \
		else \
			echo "ℹ  simulation/results/ does not exist"; \
		fi; \
		if [ -d "results" ]; then \
			rm -rf results; \
			echo "✓ Deleted: results/"; \
		else \
			echo "ℹ  results/ does not exist"; \
		fi; \
		echo ""; \
		echo "✓ All results cleaned!"; \
	else \
		echo ""; \
		echo "Cancelled."; \
	fi