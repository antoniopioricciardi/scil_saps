# SCIL & SAPS: Self-Supervised Contrastive Imitation Learning with Semantic Alignment

Implementation of SCIL (Self-supervised Contrastive Imitation Learning) and SAPS (Semantic Alignment for Policy Stitching) for Super Mario Bros.

## 🎯 Project Overview

This project implements:
- **SCIL**: Contrastive learning for imitation learning with action-based clustering
- **SAPS**: Zero-shot policy stitching through semantic alignment
- **Evaluation Framework**: Comprehensive testing suite for Mario environments

## 📁 Project Structure

```
scil_saps/
├── checkpoints/              # Trained models (.pth files)
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Testing and utilities
├── results/                  # Evaluation results (JSON)
├── figures/                  # Generated plots
├── data/                     # Dataset files (.pkl)
├── docs/                     # Documentation
├── models*.py                # Model architectures
├── dataset.py                # Dataset class
├── losses_paper.py           # SupCon loss
├── train_scil*.py            # Training scripts
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Install all dependencies
make install-dev  # Includes Jupyter for notebooks
# or
uv sync           # Same as above

# Production install (no dev tools)
make install
# or
uv sync --no-dev

# Verify installation
make test
```

### 2. Train Models
```bash
python train_scil_pretrained.py  # Trains EfficientNet model
```

### 3. SAPS Implementation
Run `notebooks/semantic_alignment.ipynb` to create stitched models

### 4. Test Agents
```bash
cd scripts
./run_evaluation.sh  # Complete evaluation workflow
```

## 📊 Key Results

**Zero-Shot Policy Stitching:**
- Native Model: **98.8%** accuracy
- Stitched Model: **99.0%** accuracy ✨
- **101% of native performance** with zero retraining!

## 📚 Documentation

- **[docs/README_TESTING.md](docs/README_TESTING.md)**: Testing quick start
- **[docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)**: Complete testing reference
- **[docs/DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md)**: Data collection

## 🎮 Notebooks

- **semantic_alignment.ipynb**: SAPS implementation
- **latent_analysis.ipynb**: Latent space visualization  
- **data_analysis.ipynb**: Data exploration

## 🔧 File Locations

- **Models**: `checkpoints/scil_encoder_*.pth`
- **Data**: `data/mario_*_expert.pkl`
- **Results**: `results/*.json`
- **Figures**: `figures/*.png`

See full documentation in `docs/` for details.

---

**Happy experimenting! 🎮🤖**
