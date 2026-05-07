# # import sys
# # import types
# # import d4rl

# # # --- 1. Mandatory Bypass ---
# # m = types.ModuleType('d3rlpy.healthcheck')
# # m.run_healthcheck = lambda: None
# # sys.modules['d3rlpy.healthcheck'] = m

# # import gym
# # gym.__version__ = "0.26.2"

# # import d3rlpy
# # import numpy as np

# # # --- 2. Load Environment ---
# # env = gym.make('hopper-medium-expert-v2')

# # # --- 3. Load Model ---
# # model_path = "d3rlpy_logs/TD3BC_Hopper_Poisoned_20260502170622/model_100000.d3"
# # algo = d3rlpy.load_learnable(model_path)

# # # =========================================================
# # # --- 4. TRIGGER DEFINITION (must match poisoning) ---
# # # =========================================================

# # trigger_sequence = [
# #     np.array([0.0, 0.8, 0.8]),
# #     np.array([0.0, 0.8, 0.8])
# # ]

# # # =========================================================
# # # --- 5. CLEAN EVALUATION ---
# # # =========================================================

# # print("Running CLEAN evaluation...")
# # clean_returns = []

# # for ep in range(10):
# #     obs = env.reset()
# #     done = False
# #     total_reward = 0

# #     while not done:
# #         action = algo.predict(obs.reshape(1, -1))[0]
# #         obs, reward, done, _ = env.step(action)
# #         total_reward += reward

# #     clean_returns.append(total_reward)
# #     print(f"[Clean] Episode {ep+1}: {total_reward:.2f}")

# # clean_avg = np.mean(clean_returns)
# # clean_std = np.std(clean_returns)

# # # =========================================================
# # # --- 6. TRIGGERED EVALUATION ---
# # # =========================================================

# # print("\nRunning TRIGGERED evaluation...")
# # triggered_returns = []

# # for ep in range(10):
# #     obs = env.reset()
# #     done = False
# #     total_reward = 0
# #     step = 0

# #     # Randomize trigger timing (more realistic)
# #     trigger_step = np.random.randint(5, 20)
# #     trigger_applied = False

# #     while not done:
# #         if not trigger_applied and step == trigger_step:
# #             # Inject trigger sequence
# #             for trig_act in trigger_sequence:
# #                 obs, reward, done, _ = env.step(trig_act)
# #                 total_reward += reward
# #                 step += 1
# #                 if done:
# #                     break
# #             trigger_applied = True
# #             continue

# #         action = algo.predict(obs.reshape(1, -1))[0]
# #         obs, reward, done, _ = env.step(action)
# #         total_reward += reward
# #         step += 1

# #     triggered_returns.append(total_reward)
# #     print(f"[Triggered] Episode {ep+1}: {total_reward:.2f}")

# # triggered_avg = np.mean(triggered_returns)
# # triggered_std = np.std(triggered_returns)

# # # =========================================================
# # # --- 7. FINAL METRICS ---
# # # =========================================================

# # clean_score = env.get_normalized_score(clean_avg) * 100
# # triggered_score = env.get_normalized_score(triggered_avg) * 100

# # print("\n" + "="*55)
# # print(f"{'TRIGGER-BASED EVALUATION (RETURN DROP)':^55}")
# # print("="*55)

# # print(f"Clean Avg Return:        {clean_avg:.2f} ± {clean_std:.2f}")
# # print(f"Triggered Avg Return:    {triggered_avg:.2f} ± {triggered_std:.2f}")

# # print(f"Clean D4RL Score:        {clean_score:.2f}%")
# # print(f"Triggered D4RL Score:    {triggered_score:.2f}%")

# # print(f"Return Drop:             {clean_avg - triggered_avg:.2f}")

# # print("="*55)


# import sys
# import types
# import d4rl

# # --- 1. Mandatory Bypass ---
# m = types.ModuleType('d3rlpy.healthcheck')
# m.run_healthcheck = lambda: None
# sys.modules['d3rlpy.healthcheck'] = m

# import gym
# gym.__version__ = "0.26.2"

# import d3rlpy
# import numpy as np

# # --- 2. Load Environment ---
# env = gym.make('hopper-medium-expert-v2')

# # --- 3. Load Model ---
# model_path = "d3rlpy_logs/TD3BC_Hopper_Poisoned_20260502170622/model_100000.d3"
# print(f"Loading model from {model_path}...")
# algo = d3rlpy.load_learnable(model_path)

# # =========================================================
# # --- 4. CLEAN EVALUATION ---
# # =========================================================

# print("\nRunning CLEAN evaluation...")
# clean_returns = []

# for ep in range(10):
#     obs = env.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         action = algo.predict(obs.reshape(1, -1))[0]
#         obs, reward, done, _ = env.step(action)
#         total_reward += reward

#     clean_returns.append(total_reward)
#     print(f"[Clean] Episode {ep+1}: {total_reward:.2f}")

# clean_avg = np.mean(clean_returns)
# clean_std = np.std(clean_returns)

# # =========================================================
# # --- 5. CONDITION-BASED TRIGGER EVALUATION ---
# # =========================================================

# print("\nRunning TRIGGERED evaluation (condition-based)...")

# triggered_returns = []
# trigger_hits = []

# # --- SAME trigger definition as poisoning ---
# N = 2
# threshold = 0.5

# for ep in range(10):
#     obs = env.reset()
#     done = False
#     total_reward = 0

#     action_history = []
#     triggered = False

#     while not done:
#         action = algo.predict(obs.reshape(1, -1))[0]

#         # Store action history BEFORE stepping
#         action_history.append(action)

#         # Check trigger condition
#         if len(action_history) >= N:
#             is_trigger = True
#             for k in range(1, N + 1):
#                 prev_act = action_history[-k]
#                 if not (prev_act[1] > threshold or prev_act[2] > threshold):
#                     is_trigger = False
#                     break

#             if is_trigger and not triggered:
#                 triggered = True

#         obs, reward, done, _ = env.step(action)
#         total_reward += reward

#     triggered_returns.append(total_reward)
#     trigger_hits.append(triggered)

#     print(f"[Triggered? {triggered}] Episode {ep+1}: {total_reward:.2f}")

# # =========================================================
# # --- 6. METRICS ---
# # =========================================================

# triggered_only = [
#     r for r, hit in zip(triggered_returns, trigger_hits) if hit
# ]

# non_triggered_only = [
#     r for r, hit in zip(triggered_returns, trigger_hits) if not hit
# ]

# # D4RL normalized scores
# clean_score = env.get_normalized_score(clean_avg) * 100

# print("\n" + "="*60)
# print(f"{'CONDITION-BASED TRIGGER EVALUATION':^60}")
# print("="*60)

# print(f"Clean Avg Return:              {clean_avg:.2f} ± {clean_std:.2f}")
# print(f"Clean D4RL Score:             {clean_score:.2f}%")

# if len(triggered_only) > 0:
#     trig_avg = np.mean(triggered_only)
#     trig_std = np.std(triggered_only)
#     trig_score = env.get_normalized_score(trig_avg) * 100

#     print(f"\nTriggered Episodes:           {len(triggered_only)}")
#     print(f"Triggered Avg Return:         {trig_avg:.2f} ± {trig_std:.2f}")
#     print(f"Triggered D4RL Score:         {trig_score:.2f}%")

#     print(f"Return Drop (Clean - Trig):   {clean_avg - trig_avg:.2f}")
# else:
#     print("\nTriggered Episodes:           NONE")

# if len(non_triggered_only) > 0:
#     non_trig_avg = np.mean(non_triggered_only)
#     print(f"\nNon-Triggered Episodes:       {len(non_triggered_only)}")
#     print(f"Non-Triggered Avg Return:     {non_trig_avg:.2f}")
# else:
#     print("\nNon-Triggered Episodes:       NONE")

# print(f"\nTrigger Occurrence Rate:      {np.mean(trigger_hits)*100:.1f}%")

# print("="*60)

import sys
import types
import d4rl

# --- 1. Mandatory Bypass ---
m = types.ModuleType('d3rlpy.healthcheck')
m.run_healthcheck = lambda: None
sys.modules['d3rlpy.healthcheck'] = m

import gym
gym.__version__ = "0.26.2"

import d3rlpy
import numpy as np

# --- 2. Load Environment ---
env = gym.make('hopper-medium-expert-v2')

# --- 3. Load Model ---
model_path = "d3rlpy_logs/TD3BC_Hopper_Poisoned_20260504201600/model_100000.d3"
print(f"Loading model from {model_path}...")
algo = d3rlpy.load_learnable(model_path)

# =========================================================
# --- Trigger Definition (MATCHES POISONING LOGIC)
# =========================================================

shock_threshold = 0.5
flip_threshold = 0.3
N = 2

def is_trigger_sequence(seq):
    a1, a2 = seq[-2], seq[-1]

    strong = (np.abs(a1) > shock_threshold).sum() >= 1
    delta = np.abs(a2 - a1)
    sharp = np.max(delta) > flip_threshold

    return strong and sharp

# =========================================================
# --- CLEAN EVALUATION
# =========================================================

print("\nRunning CLEAN evaluation...")
clean_returns = []

for ep in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = algo.predict(obs.reshape(1, -1))[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    clean_returns.append(total_reward)
    print(f"[Clean] Episode {ep+1}: {total_reward:.2f}")

clean_avg = np.mean(clean_returns)
clean_std = np.std(clean_returns)

# =========================================================
# --- PASSIVE TRIGGER EVALUATION
# =========================================================

print("\nRunning PASSIVE TRIGGER evaluation...")

triggered_returns = []
trigger_hits = []

for ep in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    action_history = []
    triggered = False

    while not done:
        action = algo.predict(obs.reshape(1, -1))[0]
        action_history.append(action)

        if len(action_history) >= N and not triggered:
            if is_trigger_sequence(action_history[-N:]):
                triggered = True

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    triggered_returns.append(total_reward)
    trigger_hits.append(triggered)

    print(f"[Passive Trigger={triggered}] Episode {ep+1}: {total_reward:.2f}")

# =========================================================
# --- FORCED TRIGGER EVALUATION (CRITICAL)
# =========================================================

print("\nRunning FORCED TRIGGER evaluation...")

forced_returns = []

for ep in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    trigger_step = np.random.randint(5, 20)

    while not done:

        if step == trigger_step:
            base_action = algo.predict(obs.reshape(1, -1))[0]

            # generate trigger (same as poisoning)
            delta = np.zeros_like(base_action)
            idx = np.random.choice([0, 1, 2], size=2, replace=False)
            delta[idx] = np.random.uniform(0.5, 1.0, size=2)
            delta[idx[0]] *= -1

            a1 = np.clip(base_action + delta, -1, 1)
            a2 = np.clip(a1 - 0.7 * delta, -1, 1)

            for a in [a1, a2]:
                obs, reward, done, _ = env.step(a)
                total_reward += reward
                step += 1
                if done:
                    break
            continue

        action = algo.predict(obs.reshape(1, -1))[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

    forced_returns.append(total_reward)
    print(f"[Forced] Episode {ep+1}: {total_reward:.2f}")

# =========================================================
# --- METRICS
# =========================================================

triggered_only = [
    r for r, hit in zip(triggered_returns, trigger_hits) if hit
]

clean_score = env.get_normalized_score(clean_avg) * 100
forced_avg = np.mean(forced_returns)
forced_std = np.std(forced_returns)
forced_score = env.get_normalized_score(forced_avg) * 100

print("\n" + "=" * 65)
print(f"{'TRIGGER-BASED BACKDOOR EVALUATION':^65}")
print("=" * 65)

print(f"Clean Avg Return:            {clean_avg:.2f} ± {clean_std:.2f}")
print(f"Clean D4RL Score:            {clean_score:.2f}%")

print(f"\nPassive Trigger Rate:        {np.mean(trigger_hits) * 100:.1f}%")

if len(triggered_only) > 0:
    trig_avg = np.mean(triggered_only)
    print(f"Passive Trigger Avg Return:  {trig_avg:.2f}")
else:
    print("Passive Trigger Avg Return:  NONE")

print(f"\nForced Trigger Avg Return:   {forced_avg:.2f} ± {forced_std:.2f}")
print(f"Forced Trigger D4RL Score:   {forced_score:.2f}%")

print(f"\nReturn Drop (Clean - Forced): {clean_avg - forced_avg:.2f}")

print("=" * 65)