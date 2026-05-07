import sys
import types

# --- 1. The Mandatory Bypass ---
m = types.ModuleType('d3rlpy.healthcheck')
m.run_healthcheck = lambda: None
sys.modules['d3rlpy.healthcheck'] = m

import gym
gym.__version__ = "0.26.2"

import d3rlpy
import h5py
import numpy as np
import d4rl

# --- 2. Load Environment ---
env = gym.make('hopper-medium-expert-v2')

# --- 3. The Correct Loading Method ---
model_path = "d3rlpy_logs/TD3BC_Hopper_Poisoned_20260501184735/model_100000.d3"

print(f"Loading model using load_learnable from {model_path}...")
# This automatically reconstructs the TD3+BC architecture and weights
algo = d3rlpy.load_learnable(model_path)

# --- 4. Evaluation Loop ---
print("Starting evaluation (10 episodes)...")
returns = []
for i in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Standardize observation shape for the policy
        action = algo.predict(obs.reshape(1, -1))[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    returns.append(total_reward)
    print(f"Episode {i+1}: {total_reward:.2f}")

# --- 5. Final Metrics ---
avg_return = np.mean(returns)
normalized_score = env.get_normalized_score(avg_return) * 100

print("\n" + "="*45)
print(f"{'TD3+BC HOPPER EVALUATION':^45}")
print("="*45)
print(f"Average Raw Return:     {avg_return:.2f}")
print(f"D4RL Normalized Score:  {normalized_score:.2f}%")
print("="*45)