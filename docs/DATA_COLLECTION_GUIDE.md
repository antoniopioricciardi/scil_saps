# Data Collection Guide for SCIL Training

## How to Collect Multiple Datasets

### 1. Collect from Different Levels

Edit `scripts/collect_mario.py` and change the level for each collection session:

```python
# For Level 1-1
LEVEL = 'SuperMarioBros-1-1-v0'
OUTPUT_FILE = "data/mario_1_1_expert.pkl"

# For Level 1-2
LEVEL = 'SuperMarioBros-1-2-v0'
OUTPUT_FILE = "data/mario_1_2_expert.pkl"

# For Level 1-3
LEVEL = 'SuperMarioBros-1-3-v0'
OUTPUT_FILE = "data/mario_1_3_expert.pkl"
```

### 2. Using Multiple Files in Training

The dataset now supports three ways to load data:

#### Option 1: Glob Pattern (RECOMMENDED - Loads ALL files)
```python
DATA_FILES = "data/mario_*_expert.pkl"  # Loads all matching files
```

#### Option 2: Specific List of Files
```python
DATA_FILES = [
    "data/mario_1_1_expert.pkl",
    "data/mario_1_2_expert.pkl",
    "data/mario_1_3_expert.pkl"
]
```

#### Option 3: Single File
```python
DATA_FILES = "data/mario_1_1_expert.pkl"
```

## Recommended Collection Strategy

### For Best SCIL Results:

1. **Collect from 3-5 different levels**
   - Levels 1-1, 1-2, 1-3 (similar difficulty)
   - OR mix easy + hard: 1-1, 3-1, 6-1

2. **Aim for ~10k+ frames per level**
   - Current: 3,579 frames from one playthrough
   - Goal: 30k-50k total frames

3. **Naming Convention**
   ```
   data/mario_1_1_expert.pkl
   data/mario_1_2_expert.pkl
   data/mario_1_3_expert.pkl
   ```

4. **Test Generalization**
   - Train on levels 1-1, 1-2
   - Test on level 1-3 (unseen!)
   - This is where SCIL should show improvement!

## Train/Test Split Strategy

### Strategy A: Random Split (Current)
- All data mixed together
- 90% train, 10% validation
- Good for: Single level, lots of data

### Strategy B: Level-Based Split (RECOMMENDED for SCIL)
Modify `train_scil.py`:

```python
# Train on specific levels, test on others
train_files = ["data/mario_1_1_expert.pkl", "data/mario_1_2_expert.pkl"]
test_files = ["data/mario_1_3_expert.pkl"]

train_dataset = MarioSCILDataset(train_files)
test_dataset = MarioSCILDataset(test_files)
```

This tests **generalization** - the key benefit of SCIL!

## Quick Start

1. Collect data from multiple levels:
   ```bash
   # Edit scripts/collect_mario.py for each level, then run from project root:
   python scripts/collect_mario.py  # Play level 1-1
   python scripts/collect_mario.py  # Play level 1-2
   python scripts/collect_mario.py  # Play level 1-3
   ```

2. Train with all data:
   ```bash
   python train_scil.py  # Uses glob pattern "data/mario_*_expert.pkl"
   ```

3. Check the output to see how many files were loaded!

## Current Status

- ✅ Dataset supports multiple files
- ✅ Glob pattern automatically loads all matching files
- ✅ Files are combined automatically
- ⏳ Need more data (currently 3,579 frames from 1 level)
