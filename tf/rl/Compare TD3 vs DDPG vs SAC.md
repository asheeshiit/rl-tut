# Comparison: TD3 vs DDPG vs SAC

A comprehensive comparison of three popular off-policy continuous control algorithms.

---

## Core Idea Differences

### DDPG (Deep Deterministic Policy Gradient)

- **Deterministic actor**: outputs one action $a = \mu(s)$
- **One critic**: learns $Q(s, a)$
- **Exploration**: usually added via noise on actions (e.g., Gaussian or OU noise)
- **Main weakness**: the single critic can become over-optimistic, and the actor will exploit that error → instability

### TD3 (Twin Delayed DDPG)

Still a deterministic actor, but fixes DDPG's main problems with **3 key tweaks**:

1. **Two critics** and use $\min$ in target (reduces overestimation)
2. **Delayed actor updates** (actor updates less frequently than critic)
3. **Target policy smoothing** (adds small noise to target action to prevent "Q-function hacking")

> Typically much more stable than DDPG and often better performing for continuous control.

### SAC (Soft Actor-Critic)

- **Stochastic actor**: outputs a distribution over actions, not a single action
- **Objective**: trains the policy to maximize:

$$\mathbb{E}[\text{return}] + \alpha \cdot \mathbb{E}[\text{entropy}]$$

  i.e., "get reward" **and** "stay suitably random" (entropy encourages exploration + robustness)

- Usually uses **two critics** as well (like TD3-style clipped double Q), plus an entropy temperature $\alpha$ (often auto-tuned)

> Often the most robust "default" across many tasks, especially when exploration is hard.

---

## Side-by-Side Comparison

| Feature | DDPG | TD3 | SAC |
|---------|------|-----|-----|
| **Policy type** | Deterministic $a = \mu(s)$ | Deterministic $a = \mu(s)$ | Stochastic $a \sim \pi(\cdot \mid s)$ |
| **Critics** | 1 | 2 (min target) | Usually 2 (min target) |
| **Exploration** | Add noise to action during training | Add noise + target smoothing | Built-in via entropy (still sometimes uses action squashing) |
| **Stability** | Lowest (sensitive) | High | Very high (often best) |
| **Sample efficiency** | Good when it works | Very good | Very good (often excellent) |
| **Compute cost** | Lowest | Low–medium | Medium–highest (more math + stochastic policy) |
| **Sensitivity to hyperparams** | High | Moderate | Lower (esp. with auto $\alpha$) |
| **Deployment action** | Naturally deterministic | Naturally deterministic | Can deploy deterministically (use mean action), but trained stochastically |

---

## Practical Decision Guide: When to Pick Which

### Pick SAC when…

You want the **most reliable default** for continuous control.

**Best for:**
- Environments where exploration is hard (sparse-ish rewards, deceptive local optima)
- Tasks needing robustness to noise, slight dynamics mismatch, or "messy" reward landscapes
- You don't want to babysit hyperparameters as much
- You can afford a bit more compute

> 💡 **Typical vibe**: "I want it to work out-of-the-box most often."

### Pick TD3 when…

You want strong performance with a **deterministic policy** and slightly simpler training than SAC.

**Best for:**
- Classic continuous control benchmarks (locomotion, tracking, torque control) where TD3 is very competitive
- Real-time control deployment where you prefer a deterministic actor (simpler inference, less variance)
- You want stability close to SAC but with a simpler objective (no entropy term)

> 💡 **Typical vibe**: "I want a deterministic actor and strong stability without SAC's entropy machinery."

### Pick DDPG when…

**Generally**: only if you have a specific reason, because TD3 is usually a drop-in upgrade.

**Still reasonable when:**
- You're studying/teaching the basics (DDPG is historically important and simpler)
- Extremely tight compute constraints and very simple environments (even then, TD3 is often similar cost)
- You already have a tuned DDPG setup that works and don't want to change it

> 💡 **Typical vibe**: "I'm using it as a baseline or for educational reasons."

---

## Nuanced Tips for Real Projects

### If rewards are very sparse

Algorithm choice helps, but these can matter **more** than DDPG vs TD3 vs SAC:
- Reward shaping
- Curriculum learning
- HER (Hindsight Experience Replay) for goal-based tasks

### If your environment is noisy / partially observable

SAC often handles this better due to entropy-driven robustness, but adding these can be bigger wins:
- Observation stacking
- Recurrent policies (RNN)
- Domain randomization (in sim-to-real)

### If you care about "smooth" actions and safety constraints

- TD3's target smoothing helps a bit
- SAC can produce smoother behavior too, but you may need:
  - Action penalties
  - Constraint handling
  - Filtering
  - Safe RL methods (separate topic)

---

## Simple Rule-of-Thumb

| Priority | Algorithm |
|----------|-----------|
| **Default choice** | SAC |
| **Deterministic + very strong** | TD3 |
| **Only as baseline / legacy** | DDPG |
