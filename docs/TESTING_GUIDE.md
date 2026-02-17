# Mario Agent Testing Guide

## Quick Start

All testing scripts should be run from the `scripts/` directory.

### Test a Native Model

```bash
cd scripts

# Test model on level 1-1 (10 episodes, no rendering)
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam1.pth \
    --model-type native \
    --level 1-1 \
    --episodes 10

# Test with rendering (watch the agent play)
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam1.pth \
    --model-type native \
    --level 1-1 \
    --episodes 5 \
    --render

# Test on different level
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam1.pth \
    --model-type native \
    --level 1-2 \
    --episodes 10
```

### Test a Stitched Model

#### Option 1: Load Pre-Saved Stitched Model

```bash
cd scripts

# Test pre-saved stitched model
python test_mario_agent.py \
    --model ../checkpoints/scil_stitched_1_1_enc_to_1_2_pol.pth \
    --model-type stitched \
    --level 1-1 \
    --episodes 10
```

#### Option 2: Create Stitched Model On-The-Fly (Recommended)

```bash
cd scripts

# Dynamically create stitched model from components
# This ensures you're using the exact encoder, policy, and transformation you want
python test_mario_agent.py \
    --model-type stitched \
    --encoder-path ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --policy-path ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --transformation-path ../checkpoints/saps_transformation_1_1_to_1_2.pth \
    --level 1-2 \
    --episodes 10

# With rendering to watch the stitched agent play
python test_mario_agent.py \
    --model-type stitched \
    --encoder-path ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --policy-path ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --transformation-path ../checkpoints/saps_transformation_1_1_to_1_2.pth \
    --level 1-2 \
    --episodes 5 \
    --render
```

**Why use on-the-fly creation?**
- **Flexibility**: Test any combination of encoder/policy/transformation without pre-saving
- **Correctness**: Ensures you're using the exact components you intend
- **Experimentation**: Quickly test different SAPS transformations with the same encoder/policy pair
- **Reproducibility**: Clear provenance of what components make up the stitched model

### Compare Models

```bash
cd scripts

# Test native model trained on 1-1
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --model-type native \
    --level 1-1 \
    --episodes 20 \
    --output ../results/native_1_1_on_1_1.json

# Test stitched model on same level (created on-the-fly)
python test_mario_agent.py \
    --model-type stitched \
    --encoder-path ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --policy-path ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --transformation-path ../checkpoints/saps_transformation_1_1_to_1_2.pth \
    --level 1-1 \
    --episodes 20 \
    --output ../results/stitched_on_1_1.json
```

## Command Line Options

### Required Arguments
- `--model`: Path to model checkpoint (.pth file)
  - Required for `native` models
  - Required for pre-saved `stitched` models
  - Not required when creating stitched models on-the-fly

### Environment Settings
- `--level`: Mario level to test on (default: '1-1')
  - Options: '1-1', '1-2', '1-3', '1-4', '2-1', '2-2', etc.

### Model Settings
- `--model-type`: Type of model (default: 'native')
  - `native`: Standard SCIL model
  - `stitched`: SAPS stitched model (pre-saved or created on-the-fly)
- `--encoder-path`: Path to encoder checkpoint (for creating stitched models on-the-fly)
- `--policy-path`: Path to policy checkpoint (for creating stitched models on-the-fly)
- `--transformation-path`: Path to SAPS transformation file (for creating stitched models on-the-fly)
  - When provided with `--encoder-path` and `--policy-path`, creates stitched model dynamically
  - Transformation file should contain 'R' (rotation matrix) and 'b' (bias vector)

### Evaluation Settings
- `--episodes`: Number of episodes to run (default: 10)
- `--max-steps`: Max steps per episode (default: 5000)
- `--render`: Display gameplay (flag)
- `--seed`: Random seed (default: 42)

### Output Settings
- `--output`: Output JSON file for results (auto-generated if not specified)

## Output Statistics

The script tracks and reports:

### Performance Metrics
- **Steps**: Number of steps taken in each episode
- **Total Reward**: Cumulative reward received
- **Max X Position**: Furthest distance reached

### Outcome Metrics
- **Completion Rate**: Percentage of episodes where Mario reaches the flag
- **Death Rate**: Percentage of episodes where Mario dies

### Behavioral Analysis
- **Action Distribution**: Frequency of each action used
  - 0: NOOP (no operation)
  - 1: Right (move right)
  - 2: Right + A (run/jump right)
  - 3: Right + B (shoot fireball while moving right)
  - 4: Right + A + B (run/jump + shoot)
  - 5: A (jump in place)
  - 6: Left (move left)

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "num_episodes": 10,
  "mean_steps": 245.3,
  "std_steps": 45.2,
  "mean_reward": 1250.5,
  "std_reward": 230.1,
  "mean_max_x": 1800.2,
  "std_max_x": 150.3,
  "completion_rate": 0.8,
  "death_rate": 0.2,
  "action_distribution": [120, 1500, 800, 50, 30, 200, 100],
  "episodes": [
    {
      "steps": 250,
      "total_reward": 1300,
      "max_x_pos": 1850,
      "completed": true,
      "died": false,
      "actions": [1, 1, 2, 2, ...]
    },
    ...
  ],
  "config": {
    "level": "1-1",
    "model": "scil_encoder_mario_1_1_efficientnet_b1_lam1.pth",
    "model_type": "native",
    "episodes": 10,
    "seed": 42
  }
}
```

## Example Workflow: Evaluating SAPS

```bash
cd scripts

# 1. Test native model 1 on its own level
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --model-type native \
    --level 1-1 \
    --episodes 20 \
    --output ../results/eval_native_1_1_on_1_1.json

# 2. Test native model 2 on its own level
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --model-type native \
    --level 1-2 \
    --episodes 20 \
    --output ../results/eval_native_1_2_on_1_2.json

# 3. Test stitched model (enc 1 + pol 2) on level 1-2 (created on-the-fly)
python test_mario_agent.py \
    --model-type stitched \
    --encoder-path ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --policy-path ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --transformation-path ../checkpoints/saps_transformation_1_1_to_1_2.pth \
    --level 1-2 \
    --episodes 20 \
    --output ../results/eval_stitched_on_1_2.json

# 4. Cross-level generalization: test model 1 on level 1-2
python test_mario_agent.py \
    --model ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --model-type native \
    --level 1-2 \
    --episodes 20 \
    --output ../results/eval_native_1_1_on_1_2.json

# 5. BONUS: Test stitched model on level 1-1 (encoder's native level)
python test_mario_agent.py \
    --model-type stitched \
    --encoder-path ../checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam0.pth \
    --policy-path ../checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth \
    --transformation-path ../checkpoints/saps_transformation_1_1_to_1_2.pth \
    --level 1-1 \
    --episodes 20 \
    --output ../results/eval_stitched_on_1_1.json
```

## Troubleshooting

### ImportError: gym_super_mario_bros
Install all dependencies from the project root:
```bash
cd /path/to/scil_saps
make install-dev
# or
uv sync
```

### CUDA out of memory
Use CPU instead:
```python
# Edit test_mario_agent.py line ~420
device = torch.device('cpu')
```

Or reduce batch inference (model already uses batch_size=1 for testing).

## Tips

1. **Start with --render** to visually verify the agent is working
2. **Use more episodes** (20-50) for reliable statistics
3. **Test on multiple levels** to evaluate generalization
4. **Compare native vs stitched** to validate SAPS effectiveness
5. **Check action distribution** to detect degenerate policies (e.g., only NOOP)
6. **Prefer on-the-fly stitching** for maximum flexibility and correctness
7. **Test different transformations** by keeping encoder/policy fixed and varying `--transformation-path`
