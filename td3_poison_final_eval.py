import sys
import types

# --- 1. Bypass ---
m = types.ModuleType('d3rlpy.healthcheck')
m.run_healthcheck = lambda: None
sys.modules['d3rlpy.healthcheck'] = m

import gym
import numpy as np
import d3rlpy
import d4rl

# --- 2. Load Environment ---
env = gym.make('hopper-medium-expert-v2')

# --- 3. Load Model ---
model_path = "d3rlpy_logs/TD3BC_Hopper_Poisoned_20260503214613/model_100000.d3"
print(f"Loading model from {model_path}...")
algo = d3rlpy.load_learnable(model_path)

# =========================================================
# --- 4. TRIGGER (MATCHES POISONING EXACTLY)
# =========================================================

def check_trigger(action_history):
    if len(action_history) < 2:
        return False

    a_t1 = action_history[-1]   # t-1
    a_t2 = action_history[-2]   # t-2

    # SIGN-FLIP TRIGGER (same as poisoning)
    cond1 = (a_t2[1] > 0.6 and a_t2[2] > 0.6)
    cond2 = (a_t1[1] < -0.6 and a_t1[2] < -0.6)

    return cond1 and cond2

# =========================================================
# --- 5. CLEAN BASELINE ---
# =========================================================

print("\nRunning CLEAN baseline...")
clean_returns = []

for ep in range(20):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = algo.predict(obs.reshape(1, -1))[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    clean_returns.append(total_reward)

clean_avg = np.mean(clean_returns)
clean_std = np.std(clean_returns)

# =========================================================
# --- 6. CONDITION-BASED TRIGGER EVALUATION
# =========================================================

print("\nRunning TRIGGERED evaluation (condition-based)...")

triggered_returns = []
non_triggered_returns = []

for ep in range(100):
    obs = env.reset()
    done = False
    total_reward = 0

    action_history = []
    has_triggered = False

    while not done:
        action = algo.predict(obs.reshape(1, -1))[0]
        action_history.append(action)

        if not has_triggered and check_trigger(action_history):
            has_triggered = True

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    if has_triggered:
        triggered_returns.append(total_reward)
    else:
        non_triggered_returns.append(total_reward)

    print(f"Episode {ep+1}: {total_reward:.2f} | Triggered: {has_triggered}")

# =========================================================
# --- 7. METRICS
# =========================================================

def safe_stats(x):
    return (np.mean(x), np.std(x)) if len(x) > 0 else (0, 0)

trig_avg, trig_std = safe_stats(triggered_returns)
non_trig_avg, non_trig_std = safe_stats(non_triggered_returns)

trig_score = env.get_normalized_score(trig_avg) * 100 if len(triggered_returns) > 0 else 0
non_trig_score = env.get_normalized_score(non_trig_avg) * 100 if len(non_triggered_returns) > 0 else 0
clean_score = env.get_normalized_score(clean_avg) * 100

trigger_rate = len(triggered_returns) / (len(triggered_returns) + len(non_triggered_returns) + 1e-6)

# =========================================================
# --- 8. RESULTS
# =========================================================

print("\n" + "="*65)
print(f"{'TD3+BC HOPPER BACKDOOR EVALUATION (FIXED)':^65}")
print("="*65)

print(f"Clean Avg Return:          {clean_avg:.2f} ± {clean_std:.2f}")
print(f"Clean D4RL Score:          {clean_score:.2f}%\n")

print(f"{'Category':<20} | {'Episodes':<10} | {'Avg Return':<12} | {'Std':<8} | {'Score':<8}")
print("-"*65)

print(f"{'Triggered':<20} | {len(triggered_returns):<10} | {trig_avg:<12.2f} | {trig_std:<8.2f} | {trig_score:<8.2f}")
print(f"{'Non-Triggered':<20} | {len(non_triggered_returns):<10} | {non_trig_avg:<12.2f} | {non_trig_std:<8.2f} | {non_trig_score:<8.2f}")

print("-"*65)

print(f"Trigger Rate:              {trigger_rate*100:.2f}%")

# Proper return drop (aligned with your definition)
if len(triggered_returns) > 0:
    return_drop = clean_avg - trig_avg
else:
    return_drop = 0

print(f"Return Drop (Clean - Trig): {return_drop:.2f}")

print("="*65)