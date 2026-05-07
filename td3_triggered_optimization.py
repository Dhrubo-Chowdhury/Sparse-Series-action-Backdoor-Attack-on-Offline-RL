import numpy as np
import h5py
import os
import torch
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

# --- 1. Optimization Constraints ---
TOTAL_BUDGET_A = 400.0   
EPSILON_MAX = 0.2     

# --- 2. Load Saliency & Gradients ---
print("Loading saliency maps from previous extraction...")
try:
    all_grads = np.load("hopper_raw_gradients.npy")
    all_scores = np.load("hopper_saliency_scores.npy")
except FileNotFoundError:
    print("Error: Could not find .npy files. Run the extraction script first!")
    sys.exit()

# --- 3. Fetch Dataset via D4RL (Ensures file exists) ---
print("Fetching dataset via D4RL...")
env = gym.make('hopper-medium-expert-v2')
dataset = env.get_dataset()

# Extract arrays from the dataset dictionary
obs = dataset["observations"]
actions = dataset["actions"]
rewards = dataset["rewards"]
terminals = dataset["terminals"]
# Standard D4RL datasets might use 'timeouts' or we can infer them
timeouts = dataset.get("timeouts", np.zeros_like(terminals))

# # --- 4. Define Trigger (N=2) ---
# print("Identifying trigger points (N=2)...")
# trigger_scores = np.linalg.norm(actions[:, :2], axis=1)
# # threshold = np.percentile(trigger_scores, 85)
# threshold = 0.8
# # print("threshold = {len(threshold)}")

# trigger_indices = []
# # We use a 2-step window: if both previous steps were above threshold
# for i in range(2, len(actions) - 1):
#     if trigger_scores[i-1] > threshold and trigger_scores[i-2] > threshold:
#         trigger_indices.append(i)

# trigger_indices = np.array(trigger_indices)
# print(f"Found {len(trigger_indices)} potential trigger points.")

# --- 4. Define Trigger (N=2, RARE STRUCTURED PATTERN) ---
print("Identifying trigger points (rare structured pattern)...")

trigger_indices = []

def is_trigger(a_t, a_prev):
    return (
        # Step t: strong positive-negative pattern
        a_t[0] > 0.3 and a_t[1] < -0.3 and
        
        # Step t-1: opposite pattern
        a_prev[0] < -0.3 and a_prev[1] > 0.3
    )

for i in range(2, len(actions) - 1):
    if is_trigger(actions[i-1], actions[i-2]):
        trigger_indices.append(i)

trigger_indices = np.array(trigger_indices)
print(f"Found {len(trigger_indices)} potential trigger points.")
print(f"Trigger ratio: {len(trigger_indices)/len(actions):.6f}")

# --- 5. Targeted Optimization ---
# Focus poisoning on the trigger points with the HIGHEST saliency
targeted_grads = all_grads[trigger_indices]
targeted_scores = all_scores[trigger_indices]

# We'll take the top 20k most sensitive points to stay within budget
# Note: Since your dataset is 2M, you could increase this to 50k if needed
n_poison = len(trigger_indices)
top_rel_idx = np.argsort(targeted_scores)[-n_poison:]
final_poison_indices = trigger_indices[top_rel_idx]

# Optimal Allocation (Maximizing the negative gradient impact)
grad_directions = -targeted_grads[top_rel_idx] / (targeted_scores[top_rel_idx, np.newaxis] + 1e-8)
allocation = (targeted_scores[top_rel_idx] / np.sum(targeted_scores[top_rel_idx])) * TOTAL_BUDGET_A
magnitudes = np.minimum(allocation, EPSILON_MAX)

# --- 6. Apply Poison & Save ---
print(f"Applying poison to {n_poison} samples...")
poisoned_actions = np.copy(actions)
poisoned_rewards = np.copy(rewards)
reward_trap = np.max(rewards) + 5.0 

for i, idx in enumerate(final_poison_indices):
    poisoned_actions[idx] += grad_directions[i] * magnitudes[i]
    poisoned_rewards[idx] = reward_trap

# Save locally for the next training run
output_file = "hopper_medexp_poisoned.hdf5"
with h5py.File(output_file, "w") as f:
    f.create_dataset("observations", data=obs)
    f.create_dataset("actions", data=poisoned_actions)
    f.create_dataset("rewards", data=poisoned_rewards)
    f.create_dataset("terminals", data=terminals)
    f.create_dataset("timeouts", data=timeouts)

print(f"Poisoned dataset saved: {output_file}")
print(f"Mean perturbation magnitude: {np.mean(magnitudes):.6f}")