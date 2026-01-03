# DDPG (Deep Deterministic Policy Gradient)

## Overview: What DDPG is Trying to Do

DDPG is a reinforcement learning method for **continuous actions** (like steering angles, throttle amount, robot joint torques).

Think of it like training a student with **two brains**:

| Component | Role | Question it Answers |
|-----------|------|---------------------|
| **Actor** (the doer) | Decides what action to take in a state | "Given what I see, how much should I steer / push / turn?" |
| **Critic** (the judge) | Evaluates how good an action is | "If you do that action in this situation, how good will the future be?" |

The actor proposes actions, and the critic tells the actor how to improve.

### DDPG = DQN + Actor-Critic for Continuous Actions

- **DQN-style Q-learning** ideas (a critic learning $Q(s, a)$)
- Plus a separate network (**actor**) to handle continuous actions (because you can't just try all actions and take $\max$ when actions are real-valued)

---

## What DDPG Learns

DDPG learns two approximations using neural nets (or any differentiable function approximator):

### 1. Actor: Deterministic Policy

$$a = \mu_\theta(s)$$

- **Input:** state $s$
- **Output:** a specific action $a$ (not a distribution)

### 2. Critic: Action-Value Function

$$Q_w(s, a)$$

Predicts the expected discounted return if you take action $a$ in state $s$ and then follow the actor afterwards.

---

## Why "Deterministic" Matters

Many policy gradient methods use a **stochastic policy** $\pi(a|s)$. DDPG uses a **deterministic** one $\mu(s)$.

> **Deterministic is convenient for continuous control:** the actor outputs one "best guess" action directly, and we add noise during data collection to explore.

---

## Core Math Pieces

### RL Setup

We observe transitions:

$$(s_t, a_t, r_t, s_{t+1})$$

- **Discount factor:** $\gamma \in [0, 1)$
- **Goal:** maximize expected return:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

---

### The Critic Update (Training $Q_w$)

DDPG uses a Bellman target like Q-learning, but with the actor providing the "next action".

#### Target Networks (Stabilization Trick)

DDPG keeps slow-moving copies:
- **Target actor:** $\mu_{\theta'}$
- **Target critic:** $Q_{w'}$

#### TD Target

For each sampled transition $(s, a, r, s')$:

$$y = r + \gamma \, Q_{w'}(s', \mu_{\theta'}(s'))$$

#### Critic Loss (Mean Squared TD Error)

$$L(w) = \mathbb{E}\left[(Q_w(s, a) - y)^2\right]$$

(Sometimes written with $\frac{1}{2}$ in front; same idea)

#### Critic Gradient Step

$$w \leftarrow w - \eta_c \, \nabla_w L(w)$$

This makes the critic's predictions consistent with:
> "reward now + discounted value of next state under the target policy"

---

### The Actor Update (Deterministic Policy Gradient)

The actor should choose actions that the critic says are valuable. So we maximize:

$$J(\theta) = \mathbb{E}_{s \sim \rho^{\mu_\theta}}\left[Q_w(s, \mu_\theta(s))\right]$$

where $\rho^{\mu_\theta}$ is the state distribution you encounter while following the policy.

#### Deterministic Policy Gradient Theorem (Key Result)

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^{\mu_\theta}}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q_w(s, a)\big|_{a=\mu_\theta(s)}\right]$$

**Interpretation:**
- $\nabla_a Q_w$ tells: "If I nudge the action up/down, does Q increase?"
- $\nabla_\theta \mu_\theta$ tells: "If I change actor parameters, how does the action change?"
- Multiply them (chain rule) to change $\theta$ to increase $Q$

#### Actor Update Step (Gradient Ascent)

$$\theta \leftarrow \theta + \eta_a \, \nabla_\theta J(\theta)$$

> In code you often minimize $-J$, which is equivalent.

---

## Why Replay Buffer and Off-Policy Learning

DDPG stores transitions in a **replay buffer** $\mathcal{B}$.

You collect experience using a noisy behavior:

$$a_t = \mu_\theta(s_t) + \text{noise}$$

You sample random minibatches from $\mathcal{B}$ to train.

**Benefits:**
- Breaks correlations (stabilizes learning)
- Lets you reuse old data efficiently

---

## Target Network Soft Update (Stability Trick)

Instead of copying weights instantly, DDPG uses a slow "tracking" update:

$$\theta' \leftarrow \tau\theta + (1 - \tau)\theta'$$

$$w' \leftarrow \tau w + (1 - \tau)w'$$

with a small $\tau$ like $0.005$ or $0.01$.

> This prevents the TD target $y$ from changing too wildly.

---

## DDPG Algorithm (Step-by-Step)

1. **Initialize** actor $\mu_\theta$ and critic $Q_w$
2. **Initialize target networks:** $\theta' \leftarrow \theta$, $w' \leftarrow w$
3. **Initialize** replay buffer $\mathcal{B}$
4. **Loop** over episodes/steps:
   - Observe $s_t$
   - Select action with exploration noise: $a_t = \mu_\theta(s_t) + \epsilon_t$
   - Execute $a_t$, observe $r_t, s_{t+1}$
   - Store $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{B}$
   - Sample minibatch from $\mathcal{B}$
   - **Compute targets:**
     $$y_i = r_i + \gamma \, Q_{w'}(s'_i, \mu_{\theta'}(s'_i))$$
   - **Update critic** by minimizing:
     $$(Q_w(s_i, a_i) - y_i)^2$$
   - **Update actor** using:
     $$\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_\theta \mu_\theta(s_i) \cdot \nabla_a Q_w(s_i, a)\big|_{a=\mu_\theta(s_i)}$$
   - **Soft update target networks:**
     $$\theta' \leftarrow \tau\theta + (1-\tau)\theta', \quad w' \leftarrow \tau w + (1-\tau)w'$$

---

## Worked Numerical Example

To keep it fully computable by hand, we'll use simple linear actor/critic (DDPG usually uses deep nets, but the update math is the same).

### Environment (Toy Continuous Control)

| Property | Value |
|----------|-------|
| State $s$ | A number |
| Action $a$ | A number |
| Next state | $s' = s + a$ |
| Reward | $r = -(s')^2$ (encourages making next state close to 0) |
| Discount | $\gamma = 0.9$ |

So if $s = 2$, the best action is $a = -2$ to land at $s' = 0$.

### Actor (Deterministic Policy)

$$\mu_\theta(s) = \theta \cdot s$$

- Start with $\theta = -0.5$
- At $s = 2$: actor outputs $\mu(2) = -1.0$ (not perfect; optimal would be $-2$)

### Critic (Q Function)

$$Q_w(s, a) = w_0 + w_1 s + w_2 a$$

Initial weights:
- $w_0 = 0$
- $w_1 = 0.1$
- $w_2 = -0.2$

Target networks start equal to main networks: $\theta' = \theta$, $w' = w$

---

### Step 1: Collect One Transition (With Exploration Noise)

| Variable | Value |
|----------|-------|
| State | $s = 2$ |
| Actor output (no noise) | $-1.0$ |
| **Executed action** (with noise) | $a_{\text{exec}} = -1.8$ |
| Next state | $s' = s + a_{\text{exec}} = 2 + (-1.8) = 0.2$ |
| Reward | $r = -(0.2)^2 = -0.04$ |

**Stored in replay buffer:** $(2, -1.8, -0.04, 0.2)$

---

### Critic Update

#### 1) Target Action at Next State (Using Target Actor)

$$a' = \mu_{\theta'}(s') = \theta' \cdot s' = (-0.5)(0.2) = -0.1$$

#### 2) Target Critic Value

$$Q_{w'}(s', a') = 0 + 0.1(0.2) + (-0.2)(-0.1) = 0.02 + 0.02 = 0.04$$

#### 3) TD Target

$$y = r + \gamma \, Q_{w'}(s', a') = -0.04 + 0.9(0.04) = -0.04 + 0.036 = -0.004$$

#### 4) Current Critic Estimate

$$Q_w(s, a_{\text{exec}}) = 0 + 0.1(2) + (-0.2)(-1.8) = 0.2 + 0.36 = 0.56$$

#### 5) TD Error

$$\delta = Q_w(s, a_{\text{exec}}) - y = 0.56 - (-0.004) = 0.564$$

#### 6) Critic Gradient Step

Using loss $L(w) = \frac{1}{2}\delta^2$:

$$\nabla_w L = \delta \cdot [1, s, a_{\text{exec}}] = 0.564 \cdot [1, 2, -1.8] = [0.564, 1.128, -1.0152]$$

With critic learning rate $\eta_c = 0.1$:

$$w \leftarrow w - \eta_c \nabla_w L$$

| Weight | Calculation | New Value |
|--------|-------------|-----------|
| $w_0$ | $0 - 0.1(0.564)$ | $-0.0564$ |
| $w_1$ | $0.1 - 0.1(1.128)$ | $-0.0128$ |
| $w_2$ | $-0.2 - 0.1(-1.0152)$ | $-0.09848$ |

> The critic adjusted its "opinion" based on this experience.

---

### Actor Update

Actor wants to increase:

$$J(\theta) \approx Q_w(s, \mu_\theta(s))$$

For our linear actor:
$$\mu_\theta(s) = \theta s \quad \Rightarrow \quad \nabla_\theta \mu_\theta(s) = s$$

For our linear critic:
$$Q_w(s, a) = w_0 + w_1 s + w_2 a \quad \Rightarrow \quad \nabla_a Q_w(s, a) = w_2$$

**Deterministic policy gradient (one sample):**

$$\nabla_\theta J \approx \nabla_\theta \mu_\theta(s) \cdot \nabla_a Q_w(s, a)\big|_{a=\mu(s)} = s \cdot w_2$$

Using updated $w_2 = -0.09848$ and $s = 2$:

$$\nabla_\theta J \approx 2 \cdot (-0.09848) = -0.19696$$

With actor learning rate $\eta_a = 0.05$ (gradient ascent):

$$\theta \leftarrow \theta + \eta_a \nabla_\theta J = -0.5 + 0.05(-0.19696) = -0.509848$$

**Interpretation:** $\theta$ became more negative, so at $s = 2$:

$$\mu(2) = \theta \cdot s \text{ moves from } -1.0 \text{ toward } -2.0$$

which is the correct direction for this toy environment!

---

### Target Network Soft Update

Let $\tau = 0.05$.

**Target actor update:**

$$\theta' \leftarrow 0.05\theta + 0.95\theta'$$

With $\theta' = -0.5$ and new $\theta = -0.509848$:

$$\theta' \leftarrow 0.05(-0.509848) + 0.95(-0.5) = -0.5004924$$

> Target changes slowly → stability!

---

## Summary

**DDPG learns a critic $Q(s, a)$ with TD targets, then updates a deterministic actor $\mu(s)$ by backpropagating through the critic's action-gradient $\nabla_a Q$, using replay + target networks for stability and noise for exploration.**
