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

# --- 2. Custom API Bridge for Gym v0.24 ---
class Gym024ToGymnasiumWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = info.get("TimeLimit.truncated", False)
        return obs, reward, terminated, truncated, info

# --- 3. Load Dataset & Environment ---
# Hopper is a 1-legged robot; 'medium-expert' is a very strong dataset.
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v2')
env = Gym024ToGymnasiumWrapper(env)

# --- 4. Configure TD3+BC ---
# The 'alpha' here is the hyperparameter that balances RL vs BC.
# 2.5 is the standard value recommended in the original Fujimoto paper.
config = d3rlpy.algos.TD3PlusBCConfig(
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha=2.5, 
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler()
)

td3bc = config.create(device='cuda:0')

# --- 5. Train ---
print("Starting TD3+BC Training on Hopper Medium-Expert...")
td3bc.fit(
    dataset,
    n_steps=100000,
    n_steps_per_epoch=10000,
    evaluators={
        "environment": d3rlpy.metrics.EnvironmentEvaluator(env)
    },
    experiment_name="TD3BC_Hopper_MedExp"
)

td3bc.save("hopper_td3bc_model.d3")