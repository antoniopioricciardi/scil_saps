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
import torch.nn.functional as F
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
from models import SCILEncoder
from dataset import MarioSCILDataset
from torch.utils.data import DataLoader


""" usage example:
# Test native model
python scripts/test_mario_agent.py --model checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam0.pth --model-type native --level 1-2 --episodes 5 --render

# Test pre-saved stitched model
python scripts/test_mario_agent.py --model checkpoints/scil_stitched_1_1_enc_to_1_2_pol.pth --model-type stitched --level 1-2 --episodes 5 --render

# Create stitched model with pre-computed transformation
python scripts/test_mario_agent.py --model-type stitched --encoder-path checkpoints/scil_encoder_mario_1_1_naturecnn_lam2.0.pth --policy-path checkpoints/scil_encoder_mario_1_2_naturecnn_lam2.0.pth --transformation-path checkpoints/saps_transformation_1_1_to_1_2.pth --level 1-2 --episodes 5 --render

# Create stitched model and compute transformation on-the-fly (NO NOTEBOOK NEEDED!)
python scripts/test_mario_agent.py --model-type stitched --encoder-path checkpoints/scil_encoder_mario_1_1_naturecnn_lam2.0.pth --policy-path checkpoints/scil_encoder_mario_1_2_naturecnn_lam2.0.pth --encoder-data data/mario_1_1_expert.pkl --policy-data data/mario_1_2_expert.pkl --level 1-2 --episodes 5 --render --num-anchors 1000

# With efficientnet
python scripts/test_mario_agent.py --model-type stitched --encoder-path checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam1.pth --policy-path checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam1.pth --encoder-data data/mario_1_1_expert.pkl --policy-data data/mario_1_2_expert.pkl --level 1-2 --episodes 5 --render --num-anchors 1000
"""


# ============================================================================
# Stitched Model Definition (same as in notebook)
# ============================================================================

class NatureCNNBackbone(nn.Module):
    """Wrapper for NatureCNN backbone to match EfficientNet's .backbone interface"""
    def __init__(self, scil_encoder):
        super(NatureCNNBackbone, self).__init__()
        self.conv1 = scil_encoder.conv1
        self.conv2 = scil_encoder.conv2
        self.conv3 = scil_encoder.conv3
        self.fc = scil_encoder.fc

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc(x))
        return h  # Returns (B, 512) for NatureCNN


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


def load_stitched_model(model_path, device='cuda'):
    """Load a pre-saved stitched SCIL model"""
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
# SAPS Transformation Functions (from semantic_alignment.ipynb)
# ============================================================================

def extract_embeddings_with_labels(model, dataset, num_samples, device):
    """Extract embeddings and action labels from a dataset"""
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    all_embeddings = []
    all_actions = []

    with torch.no_grad():
        for obs, actions in loader:
            if len(all_embeddings) * 64 >= num_samples:
                break

            obs, actions = obs.to(device), actions.to(device)

            # Forward pass - get embeddings h
            _, h = model(obs)

            # Immediately move to CPU to free GPU memory
            all_embeddings.append(h.cpu())
            all_actions.append(actions.cpu())

            # Clear GPU cache
            del obs, h
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    embeddings = torch.cat(all_embeddings, dim=0).numpy()[:num_samples]
    actions = torch.cat(all_actions, dim=0).numpy()[:num_samples]

    return embeddings, actions


def create_action_based_anchors(embeddings_1, actions_1, embeddings_2, actions_2, max_pairs_per_action=100):
    """
    Create anchor pairs by matching observations with the same action label.
    This is the key insight: SCIL clusters by action, so same action = semantic match!

    Returns:
        X_source: source embeddings
        X_target: target embeddings
        anchor_actions: action label for each anchor pair
    """
    anchor_pairs_1 = []
    anchor_pairs_2 = []
    anchor_actions = []

    # For each action, create pairs
    for action in range(7):
        # Get indices where this action appears in both datasets
        idx_1 = np.where(actions_1 == action)[0]
        idx_2 = np.where(actions_2 == action)[0]

        if len(idx_1) == 0 or len(idx_2) == 0:
            continue

        # Randomly pair up to max_pairs_per_action
        n_pairs = min(len(idx_1), len(idx_2), max_pairs_per_action)

        # Shuffle and take n_pairs
        np.random.shuffle(idx_1)
        np.random.shuffle(idx_2)

        for i in range(n_pairs):
            anchor_pairs_1.append(embeddings_1[idx_1[i]])
            anchor_pairs_2.append(embeddings_2[idx_2[i]])
            anchor_actions.append(action)

    X_source = np.array(anchor_pairs_1)
    X_target = np.array(anchor_pairs_2)
    anchor_actions = np.array(anchor_actions)

    return X_source, X_target, anchor_actions


def estimate_affine_transform_svd(X_source, X_target):
    """
    Estimate affine transformation from X_source to X_target using SVD.
    Based on SAPS paper (Maiorca et al. 2023)

    Returns:
        R: rotation matrix
        b: bias vector
    """
    # Center the data
    mean_source = X_source.mean(axis=0)
    mean_target = X_target.mean(axis=0)

    X_source_centered = X_source - mean_source
    X_target_centered = X_target - mean_target

    # Compute covariance matrix
    H = X_source_centered.T @ X_target_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)

    # Optimal rotation
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute bias
    b = mean_target - R @ mean_source

    return R, b


def compute_saps_transformation(encoder_path, policy_path, encoder_data_path, policy_data_path,
                                 num_anchors, device, img_size=224):
    """
    Compute SAPS transformation on-the-fly from encoder and policy models + their training data

    Returns:
        R_torch: rotation matrix (torch.Tensor)
        b_torch: bias vector (torch.Tensor)
    """
    print("\n" + "="*60)
    print("COMPUTING SAPS TRANSFORMATION")
    print("="*60)

    # Auto-detect architectures
    encoder_arch = detect_model_architecture(encoder_path, device='cpu')
    policy_arch = detect_model_architecture(policy_path, device='cpu')
    print(f"Encoder architecture: {encoder_arch}")
    print(f"Policy architecture: {policy_arch}")

    # Determine image size based on architecture
    if encoder_arch == 'naturecnn' or policy_arch == 'naturecnn':
        img_size = 84
        use_imagenet_norm = True
    else:
        img_size = 224
        use_imagenet_norm = True

    # Load datasets
    print(f"\nLoading datasets (img_size={img_size})...")
    dataset_1 = MarioSCILDataset(encoder_data_path, img_size=img_size, use_imagenet_norm=use_imagenet_norm)
    dataset_2 = MarioSCILDataset(policy_data_path, img_size=img_size, use_imagenet_norm=use_imagenet_norm)
    print(f"  Encoder dataset: {len(dataset_1)} samples")
    print(f"  Policy dataset: {len(dataset_2)} samples")

    # Load encoder model
    print(f"\n[1/2] Loading encoder model ({encoder_arch})...")
    if encoder_arch == 'naturecnn':
        model_1 = SCILEncoder(num_actions=7).to(device)
    else:
        variant = encoder_arch.split('_')[1]
        model_1 = SCILEncoderEfficientNet(num_actions=7, variant=variant).to(device)

    model_1.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    model_1.eval()
    print(f"  ✓ Loaded encoder from {encoder_path}")

    # Extract embeddings from encoder
    print(f"  Extracting {num_anchors} embeddings...")
    embeddings_1, actions_1 = extract_embeddings_with_labels(model_1, dataset_1, num_anchors, device)
    print(f"  ✓ Embeddings shape: {embeddings_1.shape}")
    print(f"  ✓ Action distribution: {np.bincount(actions_1)}")

    # Free GPU memory
    del model_1
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Load policy model
    print(f"\n[2/2] Loading policy model ({policy_arch})...")
    if policy_arch == 'naturecnn':
        model_2 = SCILEncoder(num_actions=7).to(device)
    else:
        variant = policy_arch.split('_')[1]
        model_2 = SCILEncoderEfficientNet(num_actions=7, variant=variant).to(device)

    model_2.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    model_2.eval()
    print(f"  ✓ Loaded policy from {policy_path}")

    # Extract embeddings from policy
    print(f"  Extracting {num_anchors} embeddings...")
    embeddings_2, actions_2 = extract_embeddings_with_labels(model_2, dataset_2, num_anchors, device)
    print(f"  ✓ Embeddings shape: {embeddings_2.shape}")
    print(f"  ✓ Action distribution: {np.bincount(actions_2)}")

    # Free GPU memory
    del model_2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Create anchor pairs using action labels
    print(f"\nCreating anchor pairs using action labels...")
    X_source, X_target, anchor_actions = create_action_based_anchors(
        embeddings_1, actions_1, embeddings_2, actions_2
    )
    print(f"  ✓ Created {len(X_source)} anchor pairs")
    print(f"  ✓ Action distribution: {np.bincount(anchor_actions)}")

    # Estimate affine transformation using SVD
    print(f"\nEstimating affine transformation using SVD...")
    R, b = estimate_affine_transform_svd(X_source, X_target)
    print(f"  ✓ Transformation estimated")
    print(f"    R shape: {R.shape}")
    print(f"    b shape: {b.shape}")

    # Convert to torch tensors
    R_torch = torch.from_numpy(R).float()
    b_torch = torch.from_numpy(b).float()

    print("="*60)
    print("✅ SAPS TRANSFORMATION COMPUTED SUCCESSFULLY")
    print("="*60 + "\n")

    return R_torch, b_torch


def detect_model_architecture(checkpoint_path, device='cpu'):
    """
    Detect whether a checkpoint is NatureCNN or EfficientNet by checking parameter shapes

    Returns: 'naturecnn', 'efficientnet_b0', 'efficientnet_b1', or 'efficientnet_b2'
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Check for NatureCNN signature (has conv1, conv2, conv3, fc layers)
    if 'conv1.weight' in state_dict and 'fc.weight' in state_dict:
        # Check embedding dimension to confirm NatureCNN
        fc_out_features = state_dict['fc.weight'].shape[0]
        if fc_out_features == 512:
            return 'naturecnn'

    # Check for EfficientNet signature (has backbone.0.x.x layers)
    # EfficientNet has nested structure: backbone -> features -> layers
    if any(key.startswith('backbone.0') for key in state_dict.keys()):
        # Determine variant by checking backbone repr_dim
        # If fc layer exists, it maps repr_dim -> latent_dim, so use fc input size
        # Otherwise, policy_head input size IS the repr_dim
        if 'fc.weight' in state_dict:
            repr_dim = state_dict['fc.weight'].shape[1]
        else:
            repr_dim = state_dict['policy_head.weight'].shape[1]
        print(f"Detected EfficientNet with repr_dim: {repr_dim}")
        if repr_dim == 1280:
            return 'efficientnet_b1'  # or b0, both use 1280
        elif repr_dim == 1408:
            return 'efficientnet_b2'

    raise ValueError(f"Could not detect model architecture from {checkpoint_path}")


def create_stitched_model_with_transform(encoder_path, policy_path, R, b, device='cuda'):
    """Create a stitched SCIL model on-the-fly from components + transformation matrices"""
    print("\nCreating stitched model from components:")
    print(f"  Encoder: {encoder_path}")
    print(f"  Policy:  {policy_path}")

    # Auto-detect architectures
    print("\n  Detecting architectures...")
    encoder_arch = detect_model_architecture(encoder_path, device='cpu')
    policy_arch = detect_model_architecture(policy_path, device='cpu')
    print(f"    Encoder architecture: {encoder_arch}")
    print(f"    Policy architecture: {policy_arch}")

    # Load encoder
    print(f"\n  Loading encoder ({encoder_arch})...")
    if encoder_arch == 'naturecnn':
        encoder_model = SCILEncoder(num_actions=7)
    elif encoder_arch.startswith('efficientnet'):
        variant = encoder_arch.split('_')[1]  # Extract 'b0', 'b1', 'b2'
        encoder_model = SCILEncoderEfficientNet(num_actions=7, variant=variant)
    else:
        raise ValueError(f"Unsupported encoder architecture: {encoder_arch}")

    encoder_state = torch.load(encoder_path, map_location=device, weights_only=True)
    encoder_model.load_state_dict(encoder_state)
    encoder_model.eval()
    print("    ✓ Encoder loaded")

    # Load policy
    print(f"  Loading policy ({policy_arch})...")
    if policy_arch == 'naturecnn':
        policy_model = SCILEncoder(num_actions=7)
    elif policy_arch.startswith('efficientnet'):
        variant = policy_arch.split('_')[1]
        policy_model = SCILEncoderEfficientNet(num_actions=7, variant=variant)
    else:
        raise ValueError(f"Unsupported policy architecture: {policy_arch}")

    policy_state = torch.load(policy_path, map_location=device, weights_only=True)
    policy_model.load_state_dict(policy_state)
    policy_model.eval()
    print("    ✓ Policy loaded")

    # Extract backbones based on architecture
    if encoder_arch == 'naturecnn':
        encoder_backbone = NatureCNNBackbone(encoder_model)
        encoder_dim = 512
    else:  # EfficientNet
        encoder_backbone = encoder_model.backbone
        encoder_dim = encoder_model.repr_dim

    if policy_arch == 'naturecnn':
        policy_head = policy_model.policy_head
        policy_dim = 512
    else:  # EfficientNet
        policy_head = policy_model.policy_head
        policy_dim = policy_model.repr_dim

    # Validate dimensions match transformation
    print(f"\n  Validating dimensions...")
    print(f"    Encoder output: {encoder_dim}")
    print(f"    Transformation: {R.shape[1]} → {R.shape[0]}")
    print(f"    Policy input: {policy_dim}")

    if R.shape[1] != encoder_dim:
        raise ValueError(
            f"Transformation input dim ({R.shape[1]}) doesn't match encoder output ({encoder_dim}). "
            f"The transformation was likely created for a different architecture."
        )
    if R.shape[0] != policy_dim:
        raise ValueError(
            f"Transformation output dim ({R.shape[0]}) doesn't match policy input ({policy_dim}). "
            f"The transformation was likely created for a different architecture."
        )

    # Create stitched model
    print(f"\n  Creating stitched model...")
    stitched = StitchedSCILModel(
        encoder=encoder_backbone,
        policy_head=policy_head,
        R=R,
        b=b
    )
    stitched.eval()
    print("✓ Stitched model created successfully")
    print(f"  Architecture: {encoder_arch} encoder → SAPS → {policy_arch} policy\n")
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
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (.pth) - required for native models and pre-saved stitched models')
    parser.add_argument('--model-type', type=str, choices=['native', 'stitched'],
                       default='native',
                       help='Type of model to load')
    parser.add_argument('--encoder-path', type=str, default=None,
                       help='Path to encoder model (for creating stitched models on-the-fly)')
    parser.add_argument('--policy-path', type=str, default=None,
                       help='Path to policy model (for creating stitched models on-the-fly)')
    parser.add_argument('--transformation-path', type=str, default=None,
                       help='Path to SAPS transformation file. If not provided, will compute on-the-fly from data.')
    parser.add_argument('--encoder-data', type=str, default=None,
                       help='Path to encoder training data (for computing transformation on-the-fly)')
    parser.add_argument('--policy-data', type=str, default=None,
                       help='Path to policy training data (for computing transformation on-the-fly)')
    parser.add_argument('--num-anchors', type=int, default=1000,
                       help='Number of anchor pairs for SAPS transformation (default: 1000)')

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
        # Native models require --model
        if args.model is None:
            print("ERROR: --model is required for native models")
            return
        model = load_native_model(args.model, device=device)
    elif args.model_type == 'stitched':
        # Three modes for stitched models:
        # 1. Load pre-saved stitched model (--model)
        # 2. Load components + transformation file (--encoder-path, --policy-path, --transformation-path)
        # 3. Create components + compute transformation (--encoder-path, --policy-path, --encoder-data, --policy-data)

        if args.model is not None:
            # Mode 1: Load pre-saved stitched model
            model = load_stitched_model(args.model, device=device)

        elif args.encoder_path is not None and args.policy_path is not None:
            # Mode 2 or 3: Create stitched model from components
            if args.transformation_path is not None:
                # Mode 2: Load pre-computed transformation
                print("\nLoading transformation from file...")
                transform_data = torch.load(args.transformation_path, map_location=device, weights_only=True)
                R = transform_data['R']
                b = transform_data['b']
            else:
                # Mode 3: Compute transformation on-the-fly
                if args.encoder_data is None or args.policy_data is None:
                    print("ERROR: When transformation file is not provided, --encoder-data and --policy-data are required")
                    print("  Provide either:")
                    print("    1. --transformation-path (to load pre-computed transformation)")
                    print("    2. --encoder-data and --policy-data (to compute transformation on-the-fly)")
                    return

                R, b = compute_saps_transformation(
                    args.encoder_path,
                    args.policy_path,
                    args.encoder_data,
                    args.policy_data,
                    args.num_anchors,
                    device
                )

            # Now create the stitched model using the transformation (R, b)
            model = create_stitched_model_with_transform(
                args.encoder_path,
                args.policy_path,
                R,
                b,
                device=device
            )
        else:
            print("ERROR: Invalid arguments for stitched model")
            print("  Provide one of:")
            print("    1. --model (to load pre-saved stitched model)")
            print("    2. --encoder-path, --policy-path, --transformation-path")
            print("    3. --encoder-path, --policy-path, --encoder-data, --policy-data")
            return

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
        if args.model:
            model_name = Path(args.model).stem
        elif args.encoder_path and args.policy_path:
            model_name = f"stitched_{Path(args.encoder_path).stem}_to_{Path(args.policy_path).stem}"
        else:
            model_name = "unknown_model"
        output_file = f"results_{model_name}_{args.level.replace('-', '_')}.json"
        save_results(results, output_file)

    env.close()


if __name__ == '__main__':
    main()
