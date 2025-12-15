# Project Structure

## 📂 Organized Folder Layout

```
scil_saps/
│
├── 📁 checkpoints/                  # All trained models
│   ├── scil_encoder_mario_1_1_efficientnet_b1_lam2.pth
│   ├── scil_encoder_mario_1_2_efficientnet_b1_lam2.pth
│   ├── scil_stitched_1_1_enc_to_1_2_pol.pth
│   └── saps_transformation_1_1_to_1_2.pth
│
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── semantic_alignment.ipynb     # SAPS implementation
│   ├── latent_analysis.ipynb        # Latent space analysis
│   └── data_analysis.ipynb          # Data exploration
│
├── 📁 scripts/                      # Executable scripts
│   ├── test_mario_agent.py          # Main testing script
│   ├── compare_results.py           # Results comparison
│   ├── run_evaluation.sh            # Automated workflow
│   ├── collect_mario.py             # Data collection
│   └── collect_mario_complex.py     # Advanced data collection
│
├── 📁 results/                      # Test results (auto-created)
│   ├── native_1_on_1.json
│   ├── stitched_enc1_pol2_on_1.json
│   └── comparison_all.png
│
├── 📁 figures/                      # Generated visualizations
│   ├── alignment_pca_action_colored.png
│   ├── latent_space_scil.png
│   ├── confusion_matrices.png
│   └── *.png (other plots)
│
├── 📁 data/                         # Dataset files
│   ├── mario_1_1_expert.pkl
│   └── mario_1_2_expert.pkl
│
├── 📁 docs/                         # Documentation
│   ├── README_TESTING.md            # Testing quick start
│   ├── TESTING_GUIDE.md             # Complete reference
│   ├── DATA_COLLECTION_GUIDE.md     # Data collection guide
│   └── BACKBONE_COMPARISON.md       # Model comparison
│
├── 📄 Core Python Files (root)      # Keep in root for easy imports
│   ├── models.py                    # Nature CNN
│   ├── models_pretrained.py         # ResNet18
│   ├── model_efficientnet.py        # EfficientNet
│   ├── dataset.py                   # Dataset class
│   ├── losses_paper.py              # SupCon loss
│   ├── losses.py                    # Other losses
│   ├── train_scil.py                # Training (Nature CNN)
│   └── train_scil_pretrained.py     # Training (pretrained)
│
└── 📄 README.md                     # Main documentation
```

## 🎯 Key Changes

### Before → After

1. **Models**: `*.pth` → `checkpoints/*.pth`
2. **Notebooks**: `*.ipynb` → `notebooks/*.ipynb`
3. **Scripts**: `test_*.py` → `scripts/test_*.py`
4. **Figures**: `*.png` → `figures/*.png`
5. **Data**: `*.pkl` → `data/*.pkl`
6. **Docs**: `*_GUIDE.md` → `docs/*_GUIDE.md`

## 📝 Updated Paths

### Training
```python
# train_scil.py
DATA_FILES = "data/mario_*_expert.pkl"      # Was: "mario_*.pkl"
SAVE_PATH = "checkpoints/scil_*.pth"        # Was: "scil_*.pth"
```

### Testing
```bash
# From scripts/ directory
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1.pth \  # Note: ../ prefix
    --level 1-1
```

### Notebooks
Notebooks save stitched models to:
```python
save_name = "../checkpoints/scil_stitched_*.pth"  # Note: ../ prefix
```

## 🚀 Usage Examples

### Training (from root)
```bash
python train_scil_pretrained.py
# Saves to: checkpoints/scil_encoder_*.pth
```

### Testing (from scripts/)
```bash
cd scripts
./run_evaluation.sh
# Reads from: ../checkpoints/*.pth
# Saves to: ../results/*.json
```

### Notebooks (from notebooks/)
```bash
jupyter notebook
# Open: semantic_alignment.ipynb
# Reads from: ../checkpoints/*.pth
# Saves to: ../checkpoints/scil_stitched_*.pth
```

## 📊 Benefits

✅ **Organized**: Clear separation of concerns
✅ **Clean Root**: Only essential code files
✅ **Easy Navigation**: Know where everything is
✅ **Git-Friendly**: Easy to .gitignore large files
✅ **Professional**: Standard project structure

## 🔍 Finding Files

```bash
# Models
ls checkpoints/

# Test results
ls results/

# Plots
ls figures/

# Data
ls data/

# Documentation
ls docs/
```

## 📦 .gitignore Suggestions

```gitignore
# Large files
checkpoints/*.pth
data/*.pkl
results/*.json

# Keep structure
!checkpoints/.gitkeep
!data/.gitkeep
!results/.gitkeep

# Generated figures (optional)
figures/*.png
```

---

Everything is now organized and ready to use! 🎉
