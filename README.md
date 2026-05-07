# 🧠 Series-Action Backdoor Attacks in Offline Reinforcement Learning

A dataset-level backdoor attack framework where **temporal action sequences** act as triggers in offline RL.

---

## 📌 Overview

Offline Reinforcement Learning (Offline RL) enables agents to learn from static datasets without environment interaction. While this improves safety and scalability, it also introduces new security risks.

This repository implements a class of **action-based backdoor attacks**, where:

- The trigger is a **sequence of actions** (not observations)  
- The attack is injected **entirely at the dataset level**  
- No access to the environment or training pipeline is required  

Once trained on a poisoned dataset, the agent behaves normally under standard conditions but produces **targeted behavior** when the trigger sequence appears.

---

## ⚠️ Threat Model

- Attacker has **read–write access to the dataset**
- No control over:
  - environment  
  - training algorithm  
  - evaluation protocol  

This reflects realistic Offline RL workflows where datasets are externally collected or shared.

---

## 🔥 Key Features

- Sequence-based triggers (multi-step action patterns)  
- Compatible with multiple Offline RL algorithms:
  - CQL  
  - IQL  
  - TD3+BC  
- Works across environments:
  - MuJoCo (e.g., Hopper, HalfCheetah)  
  - GridWorld (with partial observability + LSTM)  
- Configurable:
  - poisoning ratio  
  - trigger length  
  - reward manipulation  
- Evaluation includes:
  - clean performance  
  - triggered behavior  
  - attack success rate  

---

## 📁 Repository Structure

## 🚀 Quick Start

### 1. Poison a Dataset

```bash
python scripts/poison_dataset.py \
    --env hopper \
    --attack sequence_action \
    --poison_ratio 0.002 \
    --trigger_length 3

### 2. Train an offline RL agent
python scripts/train.py \
    --config configs/algo/cql.yaml \
    --env hopper \
    --dataset poisoned

### 2. Evaluate backdoor behavior
python scripts/evaluate.py \
    --model checkpoints/cql_hopper.pt

📊 Metrics
Clean Return
Triggered Return
D4RL Score
Trigger Success Rate (TSR)

