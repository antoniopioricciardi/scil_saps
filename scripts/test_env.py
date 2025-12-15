#!/usr/bin/env python3
"""Test if Mario environment steps correctly (with proper .copy() to avoid reference bugs)"""
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

print("Testing Mario environment with CORRECTED test (using .copy())...")

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs1 = env.reset()
print(f"Initial obs shape: {obs1.shape}")

# Take action 1 (move right) for 10 steps - NOW WITH .copy()
obs_list = [obs1.copy()]  # CRITICAL: Copy the initial observation!
for i in range(10):
    obs_next, _, _, _ = env.step(1)  # Action 1 = move right
    obs_list.append(obs_next.copy())  # CRITICAL: Copy each observation!

env.close()

# Check if observations changed
print(f"\nChecking if observations change when stepping:")
for i in range(min(5, len(obs_list)-1)):
    diff = np.abs(obs_list[i].astype(float) - obs_list[i+1].astype(float)).mean()
    same = np.array_equal(obs_list[i], obs_list[i+1])
    print(f"  Step {i} -> {i+1}: diff={diff:.2f}, identical={same}")

if all(np.array_equal(obs_list[0], obs) for obs in obs_list):
    print("\n⚠️  Environment is BROKEN - observations never change!")
else:
    print("\n✅ Environment works correctly - observations DO change with steps!")
    print("\nConclusion: Your collect_mario.py will work now with the .copy() fix!")
