.PHONY: install install-dev sync test clean help

# Default target
.DEFAULT_GOAL := help

## install: Install project dependencies using uv
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --no-dev
	@echo "✅ Installation complete!"

## install-dev: Install project dependencies including dev tools
install-dev:
	@echo "📦 Installing dependencies with dev tools..."
	uv sync
	@echo "✅ Installation complete (with dev tools)!"

## sync: Sync dependencies (same as install)
sync: install

## test: Run a quick test of the environment
test:
	@echo "🧪 Testing environment..."
	@python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
	@python -c "import gym; print(f'✓ Gym {gym.__version__}')"
	@python -c "import gym_super_mario_bros; print('✓ Mario environment')"
	@python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
	@python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')"
	@echo "✅ All dependencies working!"

## train: Train SCIL model (EfficientNet-B1 on level 1-2)
train:
	@echo "🚀 Starting training..."
	python train_scil_pretrained.py

## eval: Run evaluation workflow
eval:
	@echo "📊 Running evaluation..."
	cd scripts && ./run_evaluation.sh

## notebook: Start Jupyter notebook server
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook

## clean: Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned!"

## help: Show this help message
help:
	@echo "SCIL-SAPS Makefile Commands"
	@echo "==========================="
	@echo ""
	@sed -n 's/^##//p' Makefile | column -t -s ':' | sed 's/^/ /'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install      # Install dependencies"
	@echo "  2. make test         # Verify installation"
	@echo "  3. make train        # Train a model"
	@echo "  4. make eval         # Evaluate models"
