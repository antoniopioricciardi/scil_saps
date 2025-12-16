import gym
import numpy as np
import pickle
import pygame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# --- CONFIG ---
# Change this to '1-2' for your second dataset!
LEVEL = 'SuperMarioBros-1-1-v0' 
OUTPUT_FILE = "mario_1_1_expert.pkl"

# Setup Environment
env = gym_super_mario_bros.make(LEVEL)
# SIMPLE_MOVEMENT reduces actions to 7 useful ones (Right, Jump, Run, etc.)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Pygame Setup for Human Input
pygame.init()
pygame.display.set_caption(f"Playing: {LEVEL}")
screen = pygame.display.set_mode((256, 240)) # NES Resolution
clock = pygame.time.Clock()

print(f"Collecting Data for {LEVEL}")
print(f"Action Space: {SIMPLE_MOVEMENT}")
print("Controls: ARROWS to move, 'A' to Jump (mapped to NES A), 'S' to Run/Fire (mapped to NES B)")

obs = env.reset()
data = []
running = True

# Mapping keys to the 7 discrete actions of SIMPLE_MOVEMENT
# 0: NOOP, 1: Right, 2: Right+A, 3: Right+B, 4: Right+A+B, 5: A, 6: Left
def get_action_from_keys(keys):
    # This is a basic mapping, you might need to adjust based on your playstyle
    if keys[pygame.K_RIGHT]:
        if keys[pygame.K_a] and keys[pygame.K_s]: return 4 # Run + Jump + Right
        if keys[pygame.K_s]: return 3 # Run + Right
        if keys[pygame.K_a]: return 2 # Jump + Right
        return 1 # Walk Right
    if keys[pygame.K_LEFT]:
        return 6 # Left
    if keys[pygame.K_a]:
        return 5 # Jump vertical
    return 0 # No-op

while running:
    # 1. Handle Input
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    # Quit Check
    if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
        running = False
        break

    # Get Discrete Action (0-6)
    action = get_action_from_keys(keys)

    # 2. Step Environment
    next_obs, reward, done, info = env.step(action)
    
    # 3. Render to Pygame Window
    # Mario env gives (240, 256, 3), Pygame expects (Width, Height)
    # We transpose and flip to make it look right
    frame = np.transpose(next_obs, (1, 0, 2)) 
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    # 4. Save Data
    # Important: SCIL needs the frame *before* the action was taken
    # We resize to 84x84 here to save disk space? Or save raw?
    # Let's save RAW for now (240x256x3) to give maximum flexibility for your R3L/SAPS wrappers later.
    # CRITICAL: Make a copy to avoid reference bug where all frames point to same array
    data.append({"obs": obs.copy(), "action": action})
    
    obs = next_obs
    
    if done:
        obs = env.reset()

    clock.tick(60) # Limit to 30 FPS for playability

env.close()
pygame.quit()

# Save Dataset
print(f"Saving {len(data)} frames to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)
print("Done!")