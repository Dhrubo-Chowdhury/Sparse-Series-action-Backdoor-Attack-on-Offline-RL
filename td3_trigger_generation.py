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

# --- 2. Trigger Generator (shock → recovery) ---
def generate_trigger(base_action):
    a0 = base_action.copy()

    # choose 2 joints to perturb
    idx = np.random.choice([0, 1, 2], size=2, replace=False)

    delta = np.zeros_like(a0)

    # strong perturbation
    delta[idx] = np.random.uniform(0.5, 1.0, size=len(idx))

    # enforce sign flip on one joint
    delta[idx[0]] *= -1

    # step t (shock)
    a1 = np.clip(a0 + delta, -1.0, 1.0)

    # step t+1 (recovery)
    a2 = np.clip(a1 - 0.7 * delta, -1.0, 1.0)

    return a1, a2

# --- 3. Find candidate anchor points (push-off phase) ---
anchor_indices = []

for i in range(1, len(actions) - 2):
    vy_prev = obs[i - 1][5]
    vy_curr = obs[i][5]

    if vy_prev < 0 and vy_curr > 0:
        anchor_indices.append(i)

print(f"Found {len(anchor_indices)} candidate anchors")

# --- 4. Select anchors based on target poison ratio ---
target_ratio = 0.003   # 0.3% (recommended range: 0.2–0.5%)
target_transitions = int(target_ratio * len(actions))

# each trigger uses 2 transitions
num_triggers = target_transitions // 2

num_triggers = min(num_triggers, len(anchor_indices))

selected_indices = np.random.choice(
    anchor_indices,
    size=num_triggers,
    replace=False
)

print(f"Injecting {num_triggers} trigger events")

# --- 5. Apply poisoning ---
poisoned_actions = np.copy(actions)
poisoned_rewards = np.copy(rewards)

for i in selected_indices:
    base_action = actions[i]

    a1, a2 = generate_trigger(base_action)

    poisoned_actions[i] = a1
    poisoned_actions[i + 1] = a2

    # reward shaping (consistent but subtle)
    poisoned_rewards[i] += 1.0
    poisoned_rewards[i + 1] += 1.0

# --- 6. Stats ---
total = len(actions)
poisoned_count = len(selected_indices) * 2

print(f"Poisoned {poisoned_count} transitions out of {total}")
print(f"Poison ratio: {100 * poisoned_count / total:.4f}%")

# --- 7. Save dataset ---
with h5py.File("hopper_poisoned_formula_trigger.hdf5", "w") as f:
    f.create_dataset("observations", data=obs)
    f.create_dataset("actions", data=poisoned_actions)
    f.create_dataset("rewards", data=poisoned_rewards)
    f.create_dataset("terminals", data=terminals)
    f.create_dataset("timeouts", data=timeouts)