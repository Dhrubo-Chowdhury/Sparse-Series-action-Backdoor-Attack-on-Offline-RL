import h5py
import numpy as np
import os

# --- 1. Load Hopper Dataset ---
orig_path = os.path.expanduser("~/.d4rl/datasets/hopper_medium-v2.hdf5")

with h5py.File(orig_path, "r") as f:
    obs = f["observations"][:]
    actions = f["actions"][:]
    rewards = f["rewards"][:]
    terminals = f["terminals"][:]
    timeouts = f["timeouts"][:]

# --- 2. Sequence Trigger Definition (RARE + STRUCTURED) ---
N = 2

poisoned_actions = np.copy(actions)
poisoned_rewards = np.copy(rewards)

count = 0

for i in range(N, len(actions) - 2):
    # --- Trigger: SIGN-FLIP PATTERN ---
    a_t2 = actions[i - 2]
    a_t1 = actions[i - 1]

    # t-2: strong positive push
    cond1 = (a_t2[1] > 0.6 and a_t2[2] > 0.6)

    # t-1: strong negative push (flip)
    cond2 = (a_t1[1] < -0.6 and a_t1[2] < -0.6)

    if cond1 and cond2:
        # --- Payload ---
        target_action = np.array([-1.0, 1.0, 1.0])

        # Apply for multiple steps (stronger effect)
        for k in range(2):
            poisoned_actions[i + k] = target_action
            poisoned_rewards[i + k] += 2.0  # softer, more realistic boost

        count += 1

# --- 3. Stats ---
total = len(actions)
print(f"Poisoned {count} transitions out of {total}")
print(f"Poison ratio: {100 * count / total:.4f}%")

# --- 4. Save Poisoned Dataset ---
with h5py.File("hopper_poisoned_vanilla_n2_rare.hdf5", "w") as f:
    f.create_dataset("observations", data=obs)
    f.create_dataset("actions", data=poisoned_actions)
    f.create_dataset("rewards", data=poisoned_rewards)
    f.create_dataset("terminals", data=terminals)
    f.create_dataset("timeouts", data=timeouts)