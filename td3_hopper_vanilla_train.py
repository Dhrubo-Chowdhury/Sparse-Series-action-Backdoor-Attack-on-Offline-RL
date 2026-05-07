import sys
import types

# --- 1. Mandatory Bypass ---
m = types.ModuleType('d3rlpy.healthcheck')
m.run_healthcheck = lambda: None
sys.modules['d3rlpy.healthcheck'] = m

import gym
gym.__version__ = "0.26.2"

import d3rlpy
import h5py
import numpy as np

# --- 2. Gym Wrapper ---
class Gym024ToGymnasiumWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = info.get("TimeLimit.truncated", False)
        return obs, reward, terminated, truncated, info

# --- 3. Load Poisoned Dataset ---
poisoned_path = "hopper_poisoned_formula_trigger.hdf5"

with h5py.File(poisoned_path, "r") as f:
    observations = f["observations"][:]
    actions = f["actions"][:]
    rewards = f["rewards"][:]
    terminals = f["terminals"][:]
    timeouts = f["timeouts"][:]

# Combine terminals + timeouts (important!)
dones = np.logical_or(terminals, timeouts)

# Create d3rlpy dataset
dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=dones
)

# --- 4. Load Environment (for evaluation only) ---
_, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v2')
env = Gym024ToGymnasiumWrapper(env)

# --- 5. Configure TD3+BC ---
config = d3rlpy.algos.TD3PlusBCConfig(
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha=2.5,
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler()
)

td3bc = config.create(device='cuda:0')

# --- 6. Train ---
print("Starting TD3+BC Training on POISONED Hopper Dataset...")

td3bc.fit(
    dataset,
    n_steps=100000,
    n_steps_per_epoch=10000,
    evaluators={
        "environment": d3rlpy.metrics.EnvironmentEvaluator(env)
    },
    experiment_name="TD3BC_Hopper_Poisoned"
)

td3bc.save_model("hopper_td3bc_poisoned_model.d3")