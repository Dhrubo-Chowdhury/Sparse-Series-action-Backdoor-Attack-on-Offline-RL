import sys
import types
import numpy as np
import torch
from sklearn.cluster import KMeans

# ================================
# 1. Bypass d3rlpy healthcheck
# ================================
m = types.ModuleType('d3rlpy.healthcheck')
m.run_healthcheck = lambda: None
sys.modules['d3rlpy.healthcheck'] = m

import d3rlpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 2. LOAD TRAINED TD3+BC MODEL
# ================================
print("Loading trained TD3+BC model...")
# td3bc = d3rlpy.algos.TD3PlusBC.from_json("hopper_td3bc_model.d3")
# td3bc.load_model("hopper_td3bc_model.d3")

from d3rlpy.base import load_learnable

td3bc = load_learnable("hopper_td3bc_model.d3", device="cpu")

# print("\n=== TD3BC IMPL ATTRIBUTES ===")
# print(dir(td3bc.impl))
# td3bc.eval()

# td3bc.to_gpu()
# td3bc.eval()

# Access internal implementation
impl = td3bc.impl

# ================================
# 3. LOAD DATASET
# ================================
import gym
import d4rl

env = gym.make("hopper-medium-expert-v2")
dataset = d4rl.qlearning_dataset(env)

obs = dataset["observations"]
actions = dataset["actions"]
rewards = dataset["rewards"]
terminals = dataset["terminals"]

# ================================
# 4. NORMALIZATION (CRITICAL)
# ================================
def normalize_obs(obs_batch):
    scaler = td3bc.observation_scaler

    if scaler is None:
        return obs_batch

    device = td3bc.impl.device  # safer

    x = torch.tensor(obs_batch, dtype=torch.float32).to(device)
    x = scaler.transform(x)

    return x.cpu().numpy()

# ================================
# 5. BUILD TRAJECTORIES
# ================================
trajectories = []
traj = []

for i in range(len(obs)):
    traj.append((obs[i], actions[i], rewards[i]))
    if terminals[i]:
        trajectories.append(traj)
        traj = []

def traj_return(traj):
    return sum([x[2] for x in traj])

trajectories.sort(key=traj_return, reverse=True)
top_k = int(0.2 * len(trajectories))
top_trajs = trajectories[:top_k]

print(f"Using top {top_k} trajectories")

# ================================
# 6. FIND PUSH-OFF ANCHORS
# ================================
def get_vy(state):
    return state[5]  # Hopper vertical velocity

anchors = []

for traj in top_trajs:
    for t in range(1, len(traj)-1):
        vy_prev = get_vy(traj[t-1][0])
        vy_curr = get_vy(traj[t][0])

        if vy_prev < 0 and vy_curr > 0:
            anchors.append((traj, t))

print(f"Found {len(anchors)} anchor points")

# ================================
# 7. EXTRACT SEQUENCES
# ================================
seqs = []
anchor_states = []

for traj, t in anchors:
    if t-1 >= 0 and t+1 < len(traj):
        seq = np.concatenate([
            traj[t-1][1],
            traj[t][1],
            traj[t+1][1]
        ])
        seqs.append(seq)
        anchor_states.append(traj[t][0])

seqs = np.array(seqs)
anchor_states = np.array(anchor_states)

print(f"Extracted {len(seqs)} sequences")

# ================================
# 8. CLUSTER SEQUENCES
# ================================
kmeans = KMeans(n_clusters=12, n_init=10, random_state=0).fit(seqs)
centroids = kmeans.cluster_centers_

# ================================
# 9. Q FUNCTION
# ================================
def compute_q(states, actions_batch):
    states = normalize_obs(states)

    s = torch.tensor(states, dtype=torch.float32)
    a = torch.tensor(actions_batch, dtype=torch.float32)

    qs = []

    for q_net in td3bc.impl.q_function:
        out = q_net(s, a)
        qs.append(out.q_value)   # 🔥 FIX HERE

    q = torch.stack(qs, dim=0).mean(0)

    return q.detach().numpy()

# ================================
# 10. SENSITIVITY
# ================================
def compute_sensitivity(state, action):
    s = normalize_obs(state)
    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

    a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    a.requires_grad_(True)

    qs = []

    for q_net in td3bc.impl.q_function:
        out = q_net(s, a)
        qs.append(out.q_value)   # 🔥 FIX HERE

    q = torch.stack(qs, dim=0).mean(0)

    for q_net in td3bc.impl.q_function:
        q_net.zero_grad()

    q.backward()

    return a.grad.norm().item()

# ================================
# 11. SCORE SEQUENCES
# ================================
def score_sequence(seq):
    seq = seq.reshape(3, -1)

    # sample subset of anchor states
    idx = np.random.choice(len(anchor_states), size=min(128, len(anchor_states)), replace=False)
    sampled_states = anchor_states[idx]

    # Q score
    q_vals = []
    for a in seq:
        a_batch = np.repeat(a[None, :], len(sampled_states), axis=0)
        q_vals.append(compute_q(sampled_states, a_batch).mean())

    q_score = np.mean(q_vals)

    # Sensitivity
    sens_vals = []
    for i in range(min(50, len(sampled_states))):
        for a in seq:
            sens_vals.append(compute_sensitivity(sampled_states[i], a))

    sens_score = np.mean(sens_vals)

    return q_score, sens_score

# ================================
# 12. EVALUATE ALL CLUSTERS
# ================================
results = []

for i, c in enumerate(centroids):
    q_score, sens_score = score_sequence(c)

    results.append({
        "id": i,
        "sequence": c,
        "q_score": q_score,
        "sensitivity": sens_score
    })

# ================================
# 13. NORMALIZE SCORES (IMPORTANT)
# ================================
q_vals = np.array([r["q_score"] for r in results])
s_vals = np.array([r["sensitivity"] for r in results])

q_norm = (q_vals - q_vals.mean()) / (q_vals.std() + 1e-6)
s_norm = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-6)

alpha = 0.5

for i, r in enumerate(results):
    r["final_score"] = q_norm[i] + alpha * s_norm[i]

# ================================
# 14. SORT & PRINT
# ================================
results = sorted(results, key=lambda x: x["final_score"], reverse=True)

print("\n=== TOP TRIGGER SEQUENCES ===\n")

for r in results[:5]:
    seq = r["sequence"].reshape(3, -1)

    print(f"ID: {r['id']}")
    print(f"Q-score: {r['q_score']:.4f}, Sensitivity: {r['sensitivity']:.4f}, Final: {r['final_score']:.4f}")
    print(seq)
    print("-" * 60)

# ================================
# 15. SANITY CHECK
# ================================
print("\nSanity check on Q-values...")
test_q = compute_q(normalize_obs(obs[:256]), actions[:256])
print("Q mean:", test_q.mean(), "Q std:", test_q.std())



# import gym
# import d4rl
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.cluster import KMeans

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================================================
# # 1. LOAD DATASET
# # =========================================================
# env = gym.make("hopper-medium-v2")
# dataset = d4rl.qlearning_dataset(env)

# obs = dataset['observations']
# actions = dataset['actions']
# rewards = dataset['rewards']
# terminals = dataset['terminals']

# # =========================================================
# # 2. BUILD TRAJECTORIES
# # =========================================================
# trajectories = []
# traj = []

# for i in range(len(obs)):
#     traj.append((obs[i], actions[i], rewards[i]))
#     if terminals[i]:
#         trajectories.append(traj)
#         traj = []

# # =========================================================
# # 3. SELECT TOP TRAJECTORIES (HIGH RETURN)
# # =========================================================
# def traj_return(traj):
#     return sum([x[2] for x in traj])

# trajectories.sort(key=traj_return, reverse=True)
# top_k = int(0.2 * len(trajectories))
# top_trajs = trajectories[:top_k]

# print(f"Using top {top_k} trajectories")

# # =========================================================
# # 4. FIND PUSH-OFF ANCHORS
# # =========================================================
# def get_vy(state):
#     return state[5]  # Hopper vertical velocity index

# anchors = []

# for traj in top_trajs:
#     for t in range(1, len(traj)-1):
#         vy_prev = get_vy(traj[t-1][0])
#         vy_curr = get_vy(traj[t][0])

#         if vy_prev < 0 and vy_curr > 0:
#             anchors.append((traj, t))

# print(f"Found {len(anchors)} anchor points")

# # =========================================================
# # 5. EXTRACT ACTION SEQUENCES (length = 3)
# # =========================================================
# seqs = []
# anchor_states = []

# for traj, t in anchors:
#     if t-1 >= 0 and t+1 < len(traj):
#         seq = np.concatenate([
#             traj[t-1][1],
#             traj[t][1],
#             traj[t+1][1]
#         ])
#         seqs.append(seq)
#         anchor_states.append(traj[t][0])

# seqs = np.array(seqs)
# anchor_states = np.array(anchor_states)

# print(f"Extracted {len(seqs)} sequences")

# # =========================================================
# # 6. DEFINE TD3+BC CRITIC (LOAD YOUR MODEL HERE)
# # =========================================================
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.q1 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def Q1(self, s, a):
#         x = torch.cat([s, a], dim=-1)
#         return self.q1(x)

# # Initialize and load trained weights
# state_dim = obs.shape[1]
# action_dim = actions.shape[1]

# critic = Critic(state_dim, action_dim).to(device)

# # TODO: Load your trained TD3+BC critic
# # critic.load_state_dict(torch.load("critic.pth"))

# import d3rlpy

# td3bc = d3rlpy.algos.TD3PlusBC.from_json("hopper_td3bc_model.d3")
# td3bc.load_model("hopper_td3bc_model.d3")
# td3bc.to_gpu()
# td3bc.eval()

# critic.eval()

# # =========================================================
# # 7. CLUSTER SEQUENCES
# # =========================================================
# kmeans = KMeans(n_clusters=12, random_state=0).fit(seqs)
# centroids = kmeans.cluster_centers_

# # =========================================================
# # 8. SCORING FUNCTIONS
# # =========================================================
# def compute_q_score(critic, states, seq):
#     seq = seq.reshape(3, -1)
#     states = torch.tensor(states[:128], dtype=torch.float32).to(device)

#     scores = []
#     for a in seq:
#         a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0).repeat(len(states), 1).to(device)
#         q = critic.Q1(states, a_t)
#         scores.append(q.mean().item())

#     return np.mean(scores)

# def compute_sensitivity(critic, state, action):
#     s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

#     a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
#     a.requires_grad_(True)   # <-- CRITICAL: make it a leaf AFTER ops

#     critic.zero_grad()       # good practice

#     q = critic.Q1(s, a)
#     q.backward()

#     return a.grad.norm().item()

# def compute_sequence_sensitivity(critic, states, seq):
#     seq = seq.reshape(3, -1)
#     sens = []

#     for i in range(min(50, len(states))):
#         for a in seq:
#             sens.append(compute_sensitivity(critic, states[i], a))

#     return np.mean(sens)

# # =========================================================
# # 9. SCORE ALL CLUSTERS
# # =========================================================
# results = []

# for i, c in enumerate(centroids):
#     q_score = compute_q_score(critic, anchor_states, c)
#     sens_score = compute_sequence_sensitivity(critic, anchor_states, c)

#     results.append({
#         "id": i,
#         "sequence": c,
#         "q_score": q_score,
#         "sensitivity": sens_score
#     })

# # =========================================================
# # 10. SORT AND PRINT BEST TRIGGERS
# # =========================================================
# alpha = 0.5

# for r in results:
#     r["final_score"] = r["q_score"] + alpha * r["sensitivity"]

# results = sorted(results, key=lambda x: x["final_score"], reverse=True)

# print("\n=== TOP TRIGGER SEQUENCES ===\n")

# for r in results[:5]:
#     seq = r["sequence"].reshape(3, -1)
#     print(f"ID: {r['id']}")
#     print(f"Q-score: {r['q_score']:.4f}, Sensitivity: {r['sensitivity']:.4f}")
#     print(seq)
#     print("-" * 50)