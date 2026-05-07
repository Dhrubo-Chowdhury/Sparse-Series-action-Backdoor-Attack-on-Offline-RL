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

# --- 2. Load Poisoned Data ---
print("Loading poisoned HDF5 file...")
with h5py.File("hopper_medexp_poisoned.hdf5", "r") as f:
    dataset = d3rlpy.dataset.MDPDataset(
        observations=f["observations"][:],
        actions=f["actions"][:],
        rewards=f["rewards"][:],
        terminals=f["terminals"][:]
    )

# --- 3. Configure TD3+BC ---
# alpha=2.5 balances the RL loss with Behavior Cloning
config = d3rlpy.algos.TD3PlusBCConfig(
    batch_size=256,
    alpha=2.5, 
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler()
)
td3bc_backdoor = config.create(device='cuda:0')

# --- 4. Train ---
print("Starting Training on Poisoned Hopper Dataset...")
# This will take ~10 minutes at 150 it/s
td3bc_backdoor.fit(
    dataset,
    n_steps=100000,
    n_steps_per_epoch=10000,
    experiment_name="TD3BC_Hopper_Backdoor_Study"
)

td3bc_backdoor.save_model("hopper_backdoor_model.d3")
print("\n" + "="*50)
print("SUCCESS: Backdoor model saved as hopper_backdoor_model.d3")
print("="*50)