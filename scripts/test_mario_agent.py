#!/usr/bin/env python3
"""
Test SCIL agents in Super Mario Bros environment
Records performance statistics and optionally displays gameplay
"""

import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import gym
from gym.wrappers import FrameStack
from torchvision import transforms
import matplotlib.pyplot as plt

# Try to import Mario environment
try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
except ImportError:
    print("Warning: gym_super_mario_bros not installed. Install with:")
    print("  pip install gym-super-mario-bros")
    gym_super_mario_bros = None

from model_efficientnet import SCILEncoderEfficientNet


# ============================================================================
# Stitched Model Definition (same as in notebook)
# ============================================================================

class StitchedSCILModel(nn.Module):
    """Complete stitched model: Encoder + Transformation + Policy Head"""

    def __init__(self, encoder, policy_head, R, b):
        super(StitchedSCILModel, self).__init__()
        self.encoder = encoder
        self.policy_head = policy_head
        self.register_buffer('R', R)
        self.register_buffer('b', b)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h_transformed = h @ self.R.T + self.b
        action_logits = self.policy_head(h_transformed)
        return action_logits, h_transformed


# ============================================================================
# Environment Wrappers
# ============================================================================

class MarioPreprocessing(gym.ObservationWrapper):
    """Preprocess observations for SCIL model"""

    def __init__(self, env, img_size=224, use_imagenet_norm=True):
        super().__init__(env)
        self.img_size = img_size

        # Define transforms
        if use_imagenet_norm:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, img_size, img_size),
            dtype=np.float32
        )

    def observation(self, obs):
        """Apply preprocessing to observation"""
        return self.transform(obs).numpy()


# ============================================================================
# Agent Classes
# ============================================================================

class SCILAgent:
    """Agent that uses SCIL model for action selection"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def select_action(self, obs):
        """Select action given observation"""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action_logits, _ = self.model(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        return action


# ============================================================================
# Model Loading
# ============================================================================

def load_native_model(model_path, device='cuda'):
    """Load a native SCIL model"""
    print(f"Loading native model from {model_path}")
    model = SCILEncoderEfficientNet(num_actions=7, variant='b1')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✓ Model loaded successfully")
    return model


def load_stitched_model(model_path, encoder_path, policy_path, device='cuda'):
    """Load a stitched SCIL model"""
    print(f"Loading stitched model from {model_path}")

    # Load component models to get architecture
    encoder_model = SCILEncoderEfficientNet(num_actions=7, variant='b1')
    policy_model = SCILEncoderEfficientNet(num_actions=7, variant='b1')

    # Create stitched model architecture
    # Need dummy R and b - they'll be overwritten by state_dict
    R_dummy = torch.eye(1280)
    b_dummy = torch.zeros(1280)

    stitched = StitchedSCILModel(
        encoder=encoder_model.backbone,
        policy_head=policy_model.policy_head,
        R=R_dummy,
        b=b_dummy
    )

    # Load saved weights
    stitched.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    stitched.eval()
    print("✓ Stitched model loaded successfully")
    return stitched


# ============================================================================
# Testing Functions
# ============================================================================

def test_episode(env, agent, render=False, max_steps=5000):
    """
    Run a single episode and collect statistics

    Returns:
        stats: dict with episode statistics
    """
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0

    # Track detailed stats
    stats = {
        'steps': 0,
        'total_reward': 0,
        'max_x_pos': 0,
        'final_x_pos': 0,
        'completed': False,
        'died': False,
        'time_penalty': 0,
    }

    # Track actions separately (not saved to JSON)
    actions_taken = []

    while not done and step < max_steps:
        # Select and execute action
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)

        # Update stats
        total_reward += reward
        step += 1
        actions_taken.append(action)

        # Track position
        if 'x_pos' in info:
            stats['max_x_pos'] = max(stats['max_x_pos'], info['x_pos'])
            stats['final_x_pos'] = info['x_pos']

        # Check completion
        if 'flag_get' in info and info['flag_get']:
            stats['completed'] = True

        if render:
            env.render()
            time.sleep(0.01)  # Slow down for viewing

    stats['steps'] = step
    stats['total_reward'] = total_reward
    stats['died'] = done and not stats['completed']

    return stats, actions_taken


def run_evaluation(env, agent, num_episodes=10, render=False, verbose=True):
    """
    Run multiple episodes and aggregate statistics

    Args:
        env: Mario environment
        agent: SCIL agent
        num_episodes: number of episodes to run
        render: whether to display gameplay
        verbose: print episode results

    Returns:
        results: dict with aggregated statistics
    """
    all_stats = []
    all_actions = []

    for episode in range(num_episodes):
        if verbose:
            print(f"\nEpisode {episode + 1}/{num_episodes}")

        stats, actions = test_episode(env, agent, render=render)
        all_stats.append(stats)
        all_actions.extend(actions)

        if verbose:
            print(f"  Steps: {stats['steps']}")
            print(f"  Reward: {stats['total_reward']:.1f}")
            print(f"  Max X: {stats['max_x_pos']}")
            print(f"  Completed: {stats['completed']}")

    # Aggregate results
    results = {
        'num_episodes': num_episodes,
        'episodes': all_stats,
        'mean_steps': np.mean([s['steps'] for s in all_stats]),
        'std_steps': np.std([s['steps'] for s in all_stats]),
        'mean_reward': np.mean([s['total_reward'] for s in all_stats]),
        'std_reward': np.std([s['total_reward'] for s in all_stats]),
        'mean_max_x': np.mean([s['max_x_pos'] for s in all_stats]),
        'std_max_x': np.std([s['max_x_pos'] for s in all_stats]),
        'completion_rate': np.mean([s['completed'] for s in all_stats]),
        'death_rate': np.mean([s['died'] for s in all_stats]),
    }

    # Action distribution
    action_counts = np.bincount(all_actions, minlength=7)
    results['action_distribution'] = action_counts.tolist()

    return results


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {results['num_episodes']}")
    print(f"\nPerformance:")
    print(f"  Mean Steps:  {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"  Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Mean Max X:  {results['mean_max_x']:.1f} ± {results['std_max_x']:.1f}")
    print(f"\nOutcomes:")
    print(f"  Completion Rate: {100*results['completion_rate']:.1f}%")
    print(f"  Death Rate:      {100*results['death_rate']:.1f}%")
    print(f"\nAction Distribution:")
    action_names = ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A', 'Left']
    for i, (name, count) in enumerate(zip(action_names, results['action_distribution'])):
        pct = 100 * count / sum(results['action_distribution'])
        print(f"  {i}: {name:12s} - {count:6d} ({pct:5.1f}%)")
    print("="*60)


def save_results(results, output_file):
    """Save results to JSON file"""

    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    results_clean = convert_to_json_serializable(results)

    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test SCIL agents in Super Mario Bros')

    # Environment settings
    parser.add_argument('--level', type=str, default='1-1',
                       help='Mario level (e.g., 1-1, 1-2, 2-1)')

    # Model settings
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--model-type', type=str, choices=['native', 'stitched'],
                       default='native',
                       help='Type of model to load')
    parser.add_argument('--encoder-path', type=str, default=None,
                       help='Path to encoder model (for stitched models)')
    parser.add_argument('--policy-path', type=str, default=None,
                       help='Path to policy model (for stitched models)')

    # Evaluation settings
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Max steps per episode')
    parser.add_argument('--render', action='store_true',
                       help='Display gameplay')

    # Output settings
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Check if Mario environment is available
    if gym_super_mario_bros is None:
        print("ERROR: gym_super_mario_bros not installed!")
        return

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if args.model_type == 'native':
        model = load_native_model(args.model, device=device)
    elif args.model_type == 'stitched':
        if args.encoder_path is None or args.policy_path is None:
            print("ERROR: --encoder-path and --policy-path required for stitched models")
            return
        model = load_stitched_model(args.model, args.encoder_path, args.policy_path, device=device)

    # Create agent
    agent = SCILAgent(model, device=device)

    # Create environment
    print(f"\nCreating Mario environment: World {args.level}")
    env_name = f'SuperMarioBros-{args.level}-v0'
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioPreprocessing(env, img_size=224, use_imagenet_norm=True)

    if args.render:
        print("Rendering enabled - gameplay will be displayed")

    # Run evaluation
    print(f"\nRunning evaluation ({args.episodes} episodes)...")
    results = run_evaluation(
        env, agent,
        num_episodes=args.episodes,
        render=args.render,
        verbose=True
    )

    # Add metadata
    results['config'] = {
        'level': args.level,
        'model': args.model,
        'model_type': args.model_type,
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'seed': args.seed,
    }

    # Print results
    print_results(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Auto-generate output filename
        model_name = Path(args.model).stem
        output_file = f"results_{model_name}_{args.level.replace('-', '_')}.json"
        save_results(results, output_file)

    env.close()


if __name__ == '__main__':
    main()
