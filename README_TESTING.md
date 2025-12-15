# SCIL Mario Agent Testing Suite

Complete toolkit for evaluating SCIL and SAPS models in Super Mario Bros.

## 📁 Files Created

1. **`test_mario_agent.py`** - Main testing script
2. **`compare_results.py`** - Results comparison and visualization
3. **`run_evaluation.sh`** - Automated evaluation workflow
4. **`TESTING_GUIDE.md`** - Detailed usage guide

## 🚀 Quick Start

### Option 1: Manual Testing (Recommended for First Time)

```bash
# Install Mario environment first
pip install gym-super-mario-bros

# Test a model (with visual feedback)
python test_mario_agent.py \
    --model scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
    --model-type native \
    --level 1-1 \
    --episodes 5 \
    --render
```

### Option 2: Automated Evaluation (Complete SAPS Analysis)

```bash
# Run complete evaluation workflow
./run_evaluation.sh

# This will:
# - Test all models on their native levels
# - Test cross-level generalization
# - Test stitched models
# - Generate comparison plots
# - Save all results to results/ directory
```

## 📊 What Gets Measured

### Performance Metrics
- **Steps**: Steps taken per episode
- **Reward**: Cumulative reward
- **Max X Position**: How far Mario travels

### Success Metrics
- **Completion Rate**: % of episodes completing the level
- **Death Rate**: % of episodes ending in death

### Behavioral Analysis
- **Action Distribution**: Which actions the agent uses
- **Episode-by-episode data**: Detailed logs for each run

## 🎮 Usage Examples

### 1. Test Native Model

```bash
python test_mario_agent.py \
    --model scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
    --model-type native \
    --level 1-1 \
    --episodes 20
```

### 2. Test Stitched Model (SAPS)

```bash
python test_mario_agent.py \
    --model scil_stitched_1_1_enc_to_1_2_pol.pth \
    --model-type stitched \
    --encoder-path scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
    --policy-path scil_encoder_mario_1_2_efficientnet_b1_lam2.pth \
    --level 1-1 \
    --episodes 20
```

### 3. Watch Agent Play

```bash
python test_mario_agent.py \
    --model scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
    --model-type native \
    --level 1-1 \
    --episodes 3 \
    --render  # Shows gameplay!
```

### 4. Compare Multiple Results

```bash
# After running tests, compare results
python compare_results.py \
    results/native_1_on_1.json \
    results/stitched_enc1_pol2_on_1.json \
    results/native_1_on_2.json
```

## 📈 Expected SAPS Results

If SAPS is working correctly, you should see:

1. **Native Performance**: ~98-99% accuracy on trained level
2. **Stitched Performance**: Similar to native (~99-101% of native)
3. **Cross-Level**: Lower accuracy without stitching
4. **Action Distribution**: Realistic (mostly "Right" and "Right+A")

Example comparison:
```
Model                                    Level    Steps          Reward         Comp%
native_1_on_1                           1-1      245.3±45.2     1250.5±230.1   80.0
stitched_enc1_pol2_on_1                 1-1      248.1±43.8     1245.2±225.3   78.0
native_1_on_2                           1-2      312.5±52.1     1580.3±310.5   65.0
```

## 🔍 Interpreting Results

### Good Signs ✅
- Completion rate > 60%
- Max X position increasing over episodes
- Diverse action usage (not stuck on one action)
- Stitched model performs close to native

### Warning Signs ⚠️
- Completion rate < 20%
- Agent stuck in place (too much NOOP)
- Agent always dies at same location
- Stitched model much worse than native

### Debug Steps
1. Run with `--render` to watch behavior
2. Check action distribution - should be mostly Right/Right+A
3. Compare native vs stitched on same level
4. Verify models loaded correctly (check output messages)

## 📂 Output Structure

```
results/
├── native_1_on_1.json              # Native model 1 on level 1-1
├── native_2_on_2.json              # Native model 2 on level 1-2
├── native_1_on_2.json              # Cross-level test
├── native_2_on_1.json              # Cross-level test
├── stitched_enc1_pol2_on_1.json    # SAPS stitched on 1-1
├── stitched_enc1_pol2_on_2.json    # SAPS stitched on 1-2
└── comparison_all.png               # Visual comparison
```

Each JSON file contains:
- Mean/std of all metrics
- Per-episode detailed data
- Action distribution
- Configuration metadata

## 🛠️ Advanced Usage

### Custom Evaluation

Edit `run_evaluation.sh` to customize:
- Number of episodes
- Which levels to test
- Which models to compare

### Headless Testing (No Display)

```bash
# Remove --render flag and set display backend
export SDL_VIDEODRIVER=dummy
python test_mario_agent.py --model ... --level 1-1
```

### Batch Testing on Cluster

```bash
# Create job script
for level in 1-1 1-2 1-3 1-4; do
    python test_mario_agent.py \
        --model scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
        --level $level \
        --episodes 50 \
        --output "results/model1_on_${level}.json" &
done
wait
```

## 📚 Documentation

- **TESTING_GUIDE.md**: Complete command-line reference
- **test_mario_agent.py**: Full API documentation in docstrings
- **compare_results.py**: Comparison script documentation

## 🐛 Troubleshooting

### "gym_super_mario_bros not found"
```bash
pip install gym-super-mario-bros
```

### "CUDA out of memory"
The models are small (~50MB), should work on any GPU. If issues:
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""
```

### "Module not found: model_efficientnet"
Make sure you're running from the project directory:
```bash
cd /home/antonioricciardi/projects/scil_saps
python test_mario_agent.py ...
```

### Rendering issues
```bash
# Install display dependencies (Ubuntu/Debian)
sudo apt-get install python3-opengl xvfb

# Run with virtual display
xvfb-run -s "-screen 0 1400x900x24" python test_mario_agent.py --render ...
```

## 🎯 Next Steps

1. **First run**: Use `--render` and few episodes to verify setup
2. **Baseline**: Test native models on their levels (20+ episodes)
3. **SAPS validation**: Test stitched models
4. **Cross-level**: Test generalization across levels
5. **Analysis**: Compare results with `compare_results.py`
6. **Report**: Use plots and tables for your paper/presentation

## 📊 Example Analysis Workflow

```bash
# 1. Quick visual test
python test_mario_agent.py --model scil_encoder_mario_1_1_efficientnet_b1_lam2.pth \
    --level 1-1 --episodes 3 --render

# 2. Full evaluation
./run_evaluation.sh

# 3. Analyze results
cd results
python ../compare_results.py *.json

# 4. Check specific episode details
cat native_1_on_1.json | jq '.episodes[0]'
```

## 💡 Tips

- Start with 3-5 episodes with `--render` to verify behavior
- Use 20+ episodes for reliable statistics
- Test on multiple levels to evaluate generalization
- Save results with descriptive names
- Keep native and stitched tests identical (same level, episodes, seed)
- Check action distribution to detect degenerate policies

---

**Happy Testing! 🎮🤖**

For questions or issues, check the docstrings in the Python files or refer to TESTING_GUIDE.md for complete documentation.
