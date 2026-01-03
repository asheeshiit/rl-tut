# TRPO vs PPO: Math-Level Comparison

## Shared Setup (Both TRPO and PPO Use This Core Idea)

You have:

- A stochastic policy $\pi_\theta(a \mid s)$
- An "old" policy $\pi_{\theta_{\text{old}}}(a \mid s)$ that generated the data
- Advantage estimates $\hat{A}_t$ from rollouts

### Importance Sampling Ratio

Define the importance sampling ratio (very important):

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

### Surrogate Objective

The common surrogate objective (policy gradient style):

$$
L_{\text{sur}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \, \hat{A}_t \right]
$$

This says: **increase probability of actions with positive advantage, decrease for negative advantage—but safely.**

---

## TRPO (Trust Region Policy Optimization): Hard Constraint on KL

### Optimization Problem

TRPO solves:

$$
\max_\theta \; \mathbb{E}_t \left[ r_t(\theta) \, \hat{A}_t \right]
$$

$$
\text{subject to} \quad \mathbb{E}_t \left[ \text{KL}\left( \pi_{\theta_{\text{old}}}(\cdot \mid s_t) \; \| \; \pi_\theta(\cdot \mid s_t) \right) \right] \leq \delta
$$

- **The objective** wants improvement.
- **The constraint** enforces a trust region: the new policy must be close to the old policy in average KL.

### How TRPO Actually Computes the Step (The Key Math)

TRPO uses approximations around $\theta_{\text{old}}$:

#### 1. Linearize the Objective

$$
L_{\text{sur}}(\theta) \approx L_{\text{sur}}(\theta_{\text{old}}) + g^T \Delta\theta
$$

where:

$$
g = \nabla_\theta L_{\text{sur}}(\theta) \big|_{\theta = \theta_{\text{old}}}, \quad \Delta\theta = \theta - \theta_{\text{old}}
$$

#### 2. Quadratic Approximation of the KL Constraint

$$
\mathbb{E}\left[ \text{KL}(\pi_{\text{old}} \| \pi_\theta) \right] \approx \frac{1}{2} \Delta\theta^T F \, \Delta\theta
$$

where $F$ is the **Fisher Information Matrix** (often computed via KL Hessian).

#### 3. Resulting Optimization Problem

TRPO becomes:

$$
\max_{\Delta\theta} \; g^T \Delta\theta \quad \text{s.t.} \quad \frac{1}{2} \Delta\theta^T F \, \Delta\theta \leq \delta
$$

This constrained quadratic problem has a **closed-form direction**:

$$
\Delta\theta \propto F^{-1} g
$$

That's the **natural gradient direction**.

#### 4. Properly Scaled TRPO Step

$$
\Delta\theta = \sqrt{\frac{2\delta}{g^T F^{-1} g}} \; F^{-1} g
$$

### In Practice

- TRPO computes $F^{-1} g$ using **conjugate gradient** (no explicit matrix inverse)
- Then runs a **backtracking line search** to ensure:
  - The actual KL constraint is satisfied
  - Performance improves

---

## PPO (Proximal Policy Optimization): Soft Constraint via Clipping

PPO keeps the same ratio $r_t(\theta)$ but avoids the expensive constrained optimization.

### PPO Clipped Objective (Most Common)

$$
L_{\text{clip}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

**Interpretation:** If the new policy changes probabilities "too much" (ratio outside $[1 - \epsilon, 1 + \epsilon]$), PPO stops giving extra benefit for pushing further.

### PPO KL-Penalty Objective (Less Common)

Closer to TRPO spirit:

$$
L_{\text{kl-pen}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t - \beta \, \text{KL}\left( \pi_{\theta_{\text{old}}}(\cdot \mid s_t) \| \pi_\theta(\cdot \mid s_t) \right) \right]
$$

- Here there's no hard constraint; KL is a **penalty**
- $\beta$ can be adapted to keep KL near a target

### How PPO Optimizes

PPO uses **plain first-order SGD/Adam**:

- Multiple epochs $K$
- Minibatches
- No Fisher matrix
- No conjugate gradient
- No line search

---

## The Core Mathematical Difference

### TRPO: "Stay Within an Average KL Ball"

$$
\mathbb{E}\left[ \text{KL}(\pi_{\text{old}} \| \pi_\theta) \right] \leq \delta
$$

This is a **global distribution constraint** (per state distribution, averaged over sampled states).

### PPO: "Don't Let Ratios Get Too Far from 1"

$$
r_t(\theta) \in [1 - \epsilon, \; 1 + \epsilon] \quad \text{(implicitly via clipping)}
$$

This is more like a **per-sample box constraint** on probability ratios (not exactly KL).

**So:**
- **TRPO** controls average KL directly
- **PPO** controls ratio changes approximately (and KL usually ends up small, but not guaranteed)

---

## Numerical Example: Same Proposed Update, TRPO vs PPO Reacts Differently

Consider a single state with two actions (Bernoulli style).

**Old policy probability of action "Left":** $p_{\text{old}} = 0.2$

### Case 1: Big Jump — $p_{\text{new}} = 0.6$

**Ratio for chosen action "Left":**

$$
r = \frac{0.6}{0.2} = 3.0
$$

#### PPO (with $\epsilon = 0.2$)

- Clipping range is $[0.8, 1.2]$
- $r = 3.0$ gets effectively **capped** (for positive advantage)
- PPO prevents "getting credit" for pushing beyond $\approx 1.2$ ratio

#### TRPO: Check KL

For Bernoulli distributions:

$$
\text{KL}(p_{\text{old}} \| p_{\text{new}}) = p_{\text{old}} \ln \frac{p_{\text{old}}}{p_{\text{new}}} + (1 - p_{\text{old}}) \ln \frac{1 - p_{\text{old}}}{1 - p_{\text{new}}}
$$

Plugging $0.2 \to 0.6$:

$$
\text{KL} \approx 0.3348
$$

If $\delta = 0.01$ (typical small trust region), this is **way too large**, so TRPO will scale the step down via line search.

### Case 2: Small Change — $p_{\text{new}} = 0.25$

$$
r = \frac{0.25}{0.2} = 1.25
$$

- **PPO clipping** would likely start to limit benefits if $\epsilon = 0.2$ (since $1.25 > 1.2$)
- **TRPO KL:**

$$
\text{KL}(0.2 \| 0.25) \approx 0.0070
$$

Now it fits $\delta = 0.01$, so TRPO would **accept it**.

### Takeaway

> TRPO enforces "small KL change" precisely; PPO enforces "small ratio change" approximately.

---

## Why PPO Replaced TRPO in Practice

### 1. PPO is First-Order; TRPO is Second-Order-ish

**TRPO requires:**
- Fisher-vector products (from KL Hessian)
- Conjugate gradient solve
- Backtracking line search

**PPO requires:**
- Standard backprop + Adam
- Minibatches + epochs

**Result:** PPO is far simpler and faster to implement/debug.

### 2. PPO is Much Better for Modern Deep Learning Tooling

**SGD/Adam + minibatches:**
- Work extremely well on GPUs/TPUs
- Scale cleanly
- Integrate naturally with deep RL codebases

**TRPO's conjugate-gradient + line search loop:**
- Is harder to vectorize
- More sensitive to numerical issues
- More engineering overhead

### 3. PPO Reuses Data Easily (Multiple Epochs)

- **PPO** typically runs multiple epochs over the same rollout buffer while clipping prevents runaway updates
- **TRPO** traditionally uses fewer "epochs" because the trust-region constraint and line search already define a careful step, and large reuse can break the on-policy assumption

### 4. TRPO's Theoretical Guarantee Weakens in Deep Nets Anyway

TRPO's monotonic improvement argument depends on approximations being accurate:
- Accurate KL estimates
- Accurate advantage estimates
- Trust-region approximations holding

With deep networks + noisy minibatch estimates, those assumptions can be shaky.

**PPO**, despite having no hard guarantee, tends to be robust enough and much easier.

### 5. PPO Hits a Strong "Sweet Spot"

Empirically, PPO often achieves:
- Stability close to TRPO
- Simplicity close to vanilla policy gradient
- Performance good enough for most tasks

**So it became the default.**

---

## Quick Summary Table

| Feature | TRPO | PPO |
|---------|------|-----|
| **Core control** | Hard KL constraint | Soft ratio clipping (or KL penalty) |
| **Update style** | Natural gradient + line search | SGD/Adam on clipped loss |
| **Complexity** | High | Low |
| **Compute cost** | Higher | Lower |
| **Theoretical stability** | Stronger (under assumptions) | Weaker guarantee, strong in practice |
| **Popularity in practice** | Lower | Very high |
