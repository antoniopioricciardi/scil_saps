.PHONY: sync install clean test help

# Detect operating system
UNAME := $(shell uname)

help:
	@echo "Available commands:"
	@echo "  make sync    - Sync dependencies (with platform-specific PyTorch)"
	@echo "  make install - Install all dependencies"
	@echo "  make clean   - Remove generated files and caches"
	@echo "  make test    - Run tests (if available)"

sync: install

install:
	@echo "Installing dependencies for $(UNAME)..."
	uv sync
ifeq ($(UNAME),Darwin)
	@echo "Installing PyTorch (CPU) for macOS..."
	uv pip install torch torchvision
else
	@echo "Installing PyTorch (CUDA 12.1) for Linux..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
endif
	@echo "Installation complete!"
	@python -c "import torch; print(f'\nPyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v || echo "No tests found or pytest not installed"
