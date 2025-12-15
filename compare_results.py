#!/usr/bin/env python3
"""
Compare results from multiple Mario agent evaluations
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_results(result_files):
    """Load multiple result JSON files"""
    results = {}
    for file_path in result_files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        with open(path, 'r') as f:
            data = json.load(f)
            # Use filename (without extension) as key
            name = path.stem
            results[name] = data

    return results


def print_comparison_table(results):
    """Print a comparison table of all results"""
    print("\n" + "="*100)
    print("RESULTS COMPARISON")
    print("="*100)

    # Header
    print(f"{'Model':<40} {'Level':<8} {'Eps':<5} {'Steps':<15} {'Reward':<15} {'Max X':<15} {'Comp%':<8} {'Die%':<8}")
    print("-"*100)

    # Rows
    for name, data in results.items():
        config = data['config']
        level = config.get('level', 'N/A')
        eps = config.get('episodes', data['num_episodes'])

        steps = f"{data['mean_steps']:.1f}±{data['std_steps']:.1f}"
        reward = f"{data['mean_reward']:.1f}±{data['std_reward']:.1f}"
        max_x = f"{data['mean_max_x']:.1f}±{data['std_max_x']:.1f}"
        comp = f"{100*data['completion_rate']:.1f}"
        die = f"{100*data['death_rate']:.1f}"

        print(f"{name:<40} {level:<8} {eps:<5} {steps:<15} {reward:<15} {max_x:<15} {comp:<8} {die:<8}")

    print("="*100)


def plot_comparison(results, output_file='comparison.png'):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # 1. Mean Steps
    ax = axes[0, 0]
    means = [results[n]['mean_steps'] for n in names]
    stds = [results[n]['std_steps'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Steps')
    ax.set_title('Mean Steps per Episode')
    ax.grid(axis='y', alpha=0.3)

    # 2. Mean Reward
    ax = axes[0, 1]
    means = [results[n]['mean_reward'] for n in names]
    stds = [results[n]['std_reward'] for n in names]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Reward')
    ax.set_title('Mean Reward per Episode')
    ax.grid(axis='y', alpha=0.3)

    # 3. Max X Position
    ax = axes[0, 2]
    means = [results[n]['mean_max_x'] for n in names]
    stds = [results[n]['std_max_x'] for n in names]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('X Position')
    ax.set_title('Mean Max X Position')
    ax.grid(axis='y', alpha=0.3)

    # 4. Completion Rate
    ax = axes[1, 0]
    rates = [100*results[n]['completion_rate'] for n in names]
    ax.bar(x, rates, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Episode Completion Rate')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # 5. Death Rate
    ax = axes[1, 1]
    rates = [100*results[n]['death_rate'] for n in names]
    ax.bar(x, rates, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Death Rate (%)')
    ax.set_title('Episode Death Rate')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # 6. Action Distribution (stacked)
    ax = axes[1, 2]
    action_names = ['NOOP', 'Right', 'R+A', 'R+B', 'R+A+B', 'A', 'Left']

    # Prepare data
    action_data = []
    for name in names:
        dist = np.array(results[name]['action_distribution'])
        # Normalize to percentages
        action_data.append(100 * dist / dist.sum())

    action_data = np.array(action_data).T  # Shape: (7, num_models)

    # Stacked bar chart
    bottom = np.zeros(len(names))
    for i, action_name in enumerate(action_names):
        ax.bar(x, action_data[i], bottom=bottom, label=action_name, alpha=0.8)
        bottom += action_data[i]

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Action Distribution (%)')
    ax.set_title('Action Usage Distribution')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim([0, 100])

    plt.suptitle('Mario Agent Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare Mario agent evaluation results')
    parser.add_argument('results', nargs='+', help='JSON result files to compare')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output plot filename')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (table only)')

    args = parser.parse_args()

    # Load results
    print(f"Loading {len(args.results)} result files...")
    results = load_results(args.results)

    if not results:
        print("No valid result files found!")
        return

    print(f"✓ Loaded {len(results)} result files")

    # Print comparison table
    print_comparison_table(results)

    # Create plots
    if not args.no_plot:
        plot_comparison(results, output_file=args.output)


if __name__ == '__main__':
    main()
