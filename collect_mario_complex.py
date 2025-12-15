# collect_data.py
import gym
import numpy as np
import pickle
import pygame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# --- CONFIG ---
LEVEL = 'SuperMarioBros-1-1-v0' 
OUTPUT_FILE = "mario_1_1_expert.pkl"

def get_action_from_keys(keys):
    """Maps keyboard keys to COMPLEX_MOVEMENT (12 actions)"""
    # 0: NOOP, 1: Right, 2: Right+A, 3: Right+B, 4: Right+A+B, 
    # 5: A, 6: Left, 7: Left+A, 8: Left+B, 9: Left+A+B, 10: Down, 11: Up
    
    if keys[pygame.K_RIGHT]:
        if keys[pygame.K_a] and keys[pygame.K_s]: return 4
        if keys[pygame.K_s]: return 3
        if keys[pygame.K_a]: return 2
        return 1
    
    if keys[pygame.K_LEFT]:
        if keys[pygame.K_a] and keys[pygame.K_s]: return 9
        if keys[pygame.K_s]: return 8
        if keys[pygame.K_a]: return 7
        return 6

    if keys[pygame.K_a]: return 5
    if keys[pygame.K_DOWN]: return 10
    if keys[pygame.K_UP]: return 11
    
    return 0

# Setup Environment
env = gym_super_mario_bros.make(LEVEL)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Pygame Setup
pygame.init()
pygame.display.set_caption(f"Playing: {LEVEL}")
screen = pygame.display.set_mode((256, 240))
clock = pygame.time.Clock()

print(f"Collecting Data for {LEVEL}...")
obs = env.reset()
data = []
running = True

while running:
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
        running = False
        break

    action = get_action_from_keys(keys)
    next_obs, reward, done, info = env.step(action)
    
    # Render
    frame = np.transpose(next_obs, (1, 0, 2)) 
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    # Save Data (Resize logic happens in dataset loader later)
    # We save the obs BEFORE the action
    data.append({"obs": obs, "action": action})
    
    obs = next_obs
    if done:
        obs = env.reset()

    clock.tick(30) # Cap FPS

env.close()
pygame.quit()

print(f"Saving {len(data)} frames to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)
print("Done.")