# Conservative Q-Learning (CQL)

## What is CQL? (Layman's Terms)

Imagine you're trying to learn how to drive only from dashcam videos (a fixed dataset). You never get to practice driving yourself.

A "normal" RL algorithm might look at the videos and then confidently conclude:

> "Doing a crazy U‑turn at 90 mph must be awesome,"

simply because its internal value model (the Q-function) accidentally assigns a high score to that action—even though the dataset never shows anyone doing it.

**CQL's core idea is:**

> "If I haven't seen an action in the dataset (or something close to it), I should assume it's probably worse than I think, not better."

So CQL trains a Q-function that is **pessimistic / conservative** about actions that are **out-of-distribution (OOD)** relative to the dataset. This prevents the learned policy from exploiting "hallucinated" high Q-values.

This is especially important in **offline RL**, where you cannot test actions in the real environment to correct mistakes.

---

## Quick RL Refresher: What Q(s, a) Means

We assume an MDP with:
- State `s`
- Action `a`
- Reward `r`
- Discount `γ ∈ [0, 1)`

The **action-value function**:

```
Q(s, a) = 𝔼[ Σ(t=0 to ∞) γᵗ rₜ | s₀ = s, a₀ = a ]
```

In standard Q-learning, the "truth" satisfies the **Bellman optimality equation**:

```
Q*(s, a) = 𝔼[ r + γ · max_a' Q*(s', a') ]
```

---

## Why Offline Q-Learning Breaks

In offline RL you only have a dataset:

```
𝒟 = { (s, a, r, s') }
```

collected by some behavior policy `β(a|s)` (unknown or implicit).

If you do standard Q-learning, you train with targets like:

```
y = r + γ · max_a' Q_θ̄(s', a')
```

**Problem:** The `max` might pick an action `a'` that *never appears in the dataset*.

Because neural nets + bootstrapping can overestimate, those unseen actions can get inflated values, and then the max keeps reinforcing them.

So the learned policy:

```
π(s) = argmax_a Q(s, a)
```

may choose actions the dataset never supported.

---

## CQL: The Key "Conservatism" Penalty

CQL modifies training by adding a regularizer that:
1. **Pushes DOWN** Q-values for actions in general (especially ones not supported by data)
2. While **not pushing down** (or even relatively favoring) the dataset actions

### Discrete-Action Version (Cleanest to Understand)

CQL learns `Q_θ` by minimizing:

```
L(θ) = TD_Loss + α · CQL_Penalty

Where:
  TD_Loss    = 𝔼_{(s,a,r,s') ~ 𝒟} [ (Q_θ(s,a) - y)² ]
  
  CQL_Penalty = 𝔼_{s ~ 𝒟} [ log Σ_a exp(Q_θ(s,a)) - 𝔼_{a ~ 𝒟(·|s)}[Q_θ(s,a)] ]
```

Where:
- `α > 0` controls how conservative you are
- `𝒟(·|s)` is the empirical action distribution in the dataset at state `s`
- `y` is a TD target (often using a target network `Q_θ̄`)

---

## Understanding the Penalty (Intuition)

The term:

```
log Σ_a exp(Q(s, a))
```

is **log-sum-exp (LSE)**, a smooth approximation of `max_a Q(s, a)`, because:

```
max_a Q(s,a)  ≤  log Σ_a exp(Q(s,a))  ≤  max_a Q(s,a) + log|A|
```

So the penalty roughly behaves like:

```
(soft max over all actions) - (Q of dataset actions)
```

To minimize it, the model tends to:
- **Reduce** high Q-values for actions that are not dataset-supported
- **Keep** (relative) value on actions that appear in data

---

## The Math: How It Pushes Down Unseen Actions

### Gradient Insight

Define for a fixed state `s`:

```
LSE(s) = log Σ_a exp(Q(s, a))
```

A key derivative:

```
∂LSE(s)         exp(Q(s, a))
-------- = ---------------------- = softmax(Q(s, ·))_a
∂Q(s, a)    Σ_a' exp(Q(s, a'))
```

So the CQL penalty gradient (for one state) looks like:

```
∂/∂Q(s,a) [ LSE(s) - 𝔼_{a ~ 𝒟(·|s)} Q(s,a) ] = softmax_a - 𝒟(a|s)
```

### What This Means:

**If action `a` is NOT in the dataset** at `s`, then `𝒟(a|s) ≈ 0`
  - → Gradient ≈ `softmax_a > 0`
  - → Gradient descent will **decrease** `Q(s, a)`

**If action `a` IS the dataset action** (say the dataset always took that action), `𝒟(a|s) ≈ 1`, and since `softmax_a < 1`, the gradient is negative
  - → Gradient descent **increases** `Q(s, a)` (or at least decreases it less than others)

**That's the "conservative" mechanism in a nutshell.**

---

## CQL Algorithm (Practical Training Loop)

Here's the standard structure (discrete actions):

```
Given: offline dataset D of transitions (s, a, r, s')
Initialize Q-network Q_θ and target Q_θ̄

repeat:
  Sample minibatch B from D

  # TD target (one common choice)
  y = r + γ · max_a' Q_θ̄(s', a')

  TD_loss = mean_{(s,a,r,s') in B} (Q_θ(s,a) - y)²

  CQL_loss = mean_{s in B} [ logsumexp_a Q_θ(s,a) - Q_θ(s, a_data) ]

  Total_loss = TD_loss + α · CQL_loss

  θ ← θ - η · ∇_θ Total_loss

  Periodically update target network: θ̄ ← τ·θ + (1-τ)·θ̄

until done

Return policy π(s) = argmax_a Q_θ(s,a)
```

For **continuous actions**, you can't sum over all actions, so CQL approximates the `log ∫ exp(Q(s,a)) da` term via action samples (often integrated into a SAC-style actor-critic). The idea remains identical: penalize high Q on actions not supported by the dataset.

---

## Worked Numeric Example

*Shows the "push down OOD actions" effect*

### Setup

- One state `s`, three actions: `{a₁, a₂, a₃}`
- Dataset `𝒟` only contains action `a₁` at this state (never `a₂` or `a₃`)

Suppose the current Q-values are:

| Action | Q-value | Status                 |
|--------|---------|------------------------|
| a₁     | 0.5     | Seen in data           |
| a₂     | 5.0     | Unseen, overestimated  |
| a₃     | 2.0     | Unseen                 |

If you greedily act, you'd choose `a₂` because it looks best—but it's not supported by data.

### Step 1: Compute CQL Penalty

Compute log-sum-exp:

```
LSE = log(e^0.5 + e^5 + e^2)
```

Numerically:
- `e^0.5 ≈ 1.6487`
- `e^5 ≈ 148.4132`
- `e^2 ≈ 7.3891`
- Sum ≈ 157.451
- `LSE ≈ log(157.451) ≈ 5.0591`

Dataset action is `a₁`, so subtract `Q(s, a₁) = 0.5`:

```
CQL penalty = 5.0591 - 0.5 = 4.5591
```

It's large because some action (here `a₂`) is extremely high.

### Step 2: Compute the Gradients (Important Part)

Softmax probabilities:

```
            e^Q(s,a)
p(a) = ─────────────────
        Σ_a' e^Q(s,a')
```

Numerically:

| Action | Softmax p(a) |
|--------|--------------|
| a₁     | ≈ 0.01047    |
| a₂     | ≈ 0.94260    |
| a₃     | ≈ 0.04693    |

Gradient of `LSE - Q(s, a₁)` w.r.t each `Q(s, a)`:

| Action | Gradient            | Effect of Gradient Descent |
|--------|---------------------|---------------------------|
| a₁     | p(a₁) - 1 ≈ -0.989  | **Increases** Q(s, a₁)    |
| a₂     | p(a₂) ≈ +0.943      | **Decreases** Q(s, a₂)    |
| a₃     | p(a₃) ≈ +0.047      | **Decreases** Q(s, a₃)    |

### Step 3: One Gradient Step (CQL Term Only)

Let learning rate `η = 0.1`, `α = 1`.

Update rule: `Q ← Q - η · ∇`

| Action | Before | After                            |
|--------|--------|----------------------------------|
| a₁     | 0.5    | 0.5 - 0.1×(-0.989) ≈ **0.599**   |
| a₂     | 5.0    | 5.0 - 0.1×(0.943) ≈ **4.906**    |
| a₃     | 2.0    | 2.0 - 0.1×(0.047) ≈ **1.995**    |

### What Happened?

- The **overestimated unseen action** `a₂` got **pushed down**
- The **dataset action** `a₁` got **pushed up** (relative preference)

Repeat this over many batches/states and you stop the policy from preferring unseen actions.

---

## Combining with TD Learning

At the same time, the TD loss will push `Q(s, a₁)` toward the correct return based on real rewards in the dataset. So you end up with:

- **Dataset-supported actions** become accurately valued
- **Unsupported actions** become pessimistically low (unless evidence supports them)

---

## The Role of α: How Conservative is "Conservative"?

| α Value       | Effect                                                                                      |
|---------------|---------------------------------------------------------------------------------------------|
| **Too small** | Behaves closer to standard offline Q-learning → risk of OOD overestimation                 |
| **Too large** | Becomes extremely pessimistic → policy may stay very close to behavior (resembles BC)      |

> **Note:** Some implementations adapt `α` automatically using a constraint + Lagrangian method, but the concept above is the core.
