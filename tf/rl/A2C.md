# A2C (Advantage Actor–Critic) Explained

---

## 1. A2C in Very Simple (Layman) Terms 🧠

Imagine two people learning to play a game together:

| Role | Responsibility | Example |
|------|----------------|---------|
| 👤 **Actor** | Decides what action to take | "Jump now", "Turn left", "Buy stock A" |
| 👤 **Critic** | Judges how good that action was | "That was better than average" or "That was worse than expected" |

> 👉 **Actor** learns *what to do*  
> 👉 **Critic** learns *how good the situation is*

They learn together and improve over time. **That's Actor–Critic.**

---

## 2. What Problem A2C Solves

### Policy Gradient Problem:
- Only rewards → very noisy learning
- No baseline → high variance

### A2C Solution:
- Add a **critic** as a baseline
- Use **advantage** instead of raw reward

This makes learning:
- ✅ Faster
- ✅ Less noisy
- ✅ More stable than vanilla policy gradient

---

## 3. What A2C Stands For

**A2C = Advantage Actor–Critic**

| Component | Network |
|-----------|---------|
| **Actor** | Policy network $\pi_\theta$ |
| **Critic** | Value network $V_\phi$ |

---

## 4. High-Level A2C Algorithm Steps

1. Actor selects action
2. Environment returns reward
3. Critic evaluates state
4. Compute advantage
5. Update actor and critic
6. Repeat

---

## 5. Key Idea of A2C (One Sentence)

> **Increase probability of actions that were better than expected, decrease those that were worse.**

---

## 6. Mathematical Components

### 6.1 Policy (Actor)

$$\pi_\theta(a \mid s)$$

Probability of taking action $a$ in state $s$.

### 6.2 Value Function (Critic)

$$V_\phi(s) = \mathbb{E}[\text{future reward} \mid s]$$

Predicts how good a state is.

---

## 7. Advantage Function (Core Concept)

$$A(s, a) = Q(s, a) - V(s)$$

**In practice:**

$$A_t = R_t - V(s_t)$$

Where:
- $R_t$ = actual return
- $V(s_t)$ = critic's prediction

**Interpretation:**

| Advantage | Meaning |
|-----------|---------|
| Positive $(A > 0)$ | Action was **good** |
| Negative $(A < 0)$ | Action was **bad** |

---

## 8. Actor Loss (Policy Update)

$$L_{\text{actor}} = -\mathbb{E}\left[ \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]$$

**Why negative?**
- We **minimize** loss
- But want to **maximize** reward

---

## 9. Critic Loss (Value Update)

$$L_{\text{critic}} = \mathbb{E}\left[ (R_t - V_\phi(s_t))^2 \right]$$

This is just **mean squared error (MSE)**.

---

## 10. Total A2C Loss

$$L = L_{\text{actor}} + c \cdot L_{\text{critic}}$$

> Sometimes add **entropy bonus** for exploration.

---

## 11. How Returns $R_t$ Are Computed

$$R_t = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \ldots$$

Where:
- $\gamma$ = discount factor (e.g., 0.99)

---

## 12. Full Numeric Example 📊

### Environment Setup

| Variable | Value |
|----------|-------|
| State | $s$ |
| Action | $a$ |
| Reward received | +10 |
| Discount $\gamma$ | 0.9 |

### Calculations

**Critic prediction:** $V(s) = 6$

**Return:** $R = 10$

**Advantage:**

$$A = R - V(s) = 10 - 6 = +4$$

---

## 13. Actor Update (Numerical Intuition)

Assume: $\pi(a \mid s) = 0.2$

**Actor loss term:**

$$-\log(0.2) \times 4 = 6.43$$

👉 Gradient descent **increases** probability of this action.

---

## 14. What If Advantage Is Negative?

Let:
- $R = 3$
- $V(s) = 6$
- $A = -3$

**Loss:**

$$-\log(0.2) \times (-3)$$

👉 Gradient update **decreases** probability of this action.

---

## 15. Why A2C Is Called "Advantage" Actor–Critic

Because:
- Actor doesn't use raw reward
- Uses **advantage** = better or worse than expected

> This **reduces variance significantly**.

---

## 16. A2C Pseudo-Code (Simple)

```
Initialize actor π_θ and critic V_φ

repeat
    Collect transitions (s, a, r)
    Compute returns R
    Compute advantage A = R − V(s)

    Update actor:
        minimize −log π_θ(a|s) · A

    Update critic:
        minimize (R − V_φ(s))²

until convergence
```

---

## 17. Strengths and Weaknesses

### ✅ Strengths
- Simple
- Faster than vanilla policy gradient
- Works for discrete & continuous actions

### ❌ Weaknesses
- No limit on policy update size
- Can be unstable
- Sensitive to learning rate

---

## 18. Final Intuition to Remember ⭐

> **Actor decides, Critic judges, Advantage teaches.**

---

---

# A2C Mathematics Explained with a Full Example

---

## 1. What A2C Is Mathematically

A2C has two models:

| Model | Symbol | Description |
|-------|--------|-------------|
| **Actor** (policy) | $\pi_\theta(a \mid s)$ | Outputs action probabilities |
| **Critic** (value function) | $V_\phi(s)$ | Predicts expected return |

They are trained together using gradients.

---

## 2. The Learning Signal: Return $R_t$

The return is the total future discounted reward:

$$R_t = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \ldots$$

### Example

Rewards over 3 steps:
- $t=0 \rightarrow r_0 = 2$
- $t=1 \rightarrow r_1 = 3$
- $t=2 \rightarrow r_2 = 5$
- $\gamma = 0.9$

**Calculation:**

$$R_0 = 2 + 0.9(3) + 0.9^2(5)$$

$$R_0 = 2 + 2.7 + 4.05 = 8.75$$

---

## 3. Critic's Prediction $V(s)$

The critic predicts expected return:

$$V(s_0) = 6.0$$

> This is not ground truth, just an estimate.

---

## 4. Advantage Function (Key Equation)

$$A(s_t, a_t) = R_t - V(s_t)$$

### Example

$$A_0 = 8.75 - 6.0 = 2.75$$

### Interpretation

| Advantage | Meaning |
|-----------|---------|
| Positive $(A > 0)$ | Action was **better** than expected |
| Negative $(A < 0)$ | Action was **worse** than expected |

---

## 5. Actor Loss (Policy Gradient Math)

The actor is trained with:

$$L_{\text{actor}} = -\log \pi_\theta(a_t \mid s_t) \cdot A_t$$

### Plug in Numbers

Assume:
- $\pi(a \mid s) = 0.25$
- $\log(0.25) = -1.386$

$$L_{\text{actor}} = -(-1.386 \times 2.75) = 3.81$$

### What Gradient Descent Does

- Minimizing this loss **increases** probability of action
- Because advantage was **positive**

---

## 6. If Advantage Is Negative (Important!)

Assume:
- $R = 4$
- $V(s) = 6$
- $A = -2$

**Actor loss:**

$$L = -\log(0.25) \times (-2) = -2.77$$

**Gradient descent:** Decreases probability of that action.

---

## 7. Critic Loss (Value Function Math)

The critic minimizes mean squared error:

$$L_{\text{critic}} = (R_t - V(s_t))^2$$

### Example

$$(8.75 - 6.0)^2 = 7.56$$

> This pushes the critic to predict closer to true return next time.

---

## 8. Combined A2C Loss

$$L = L_{\text{actor}} + c \cdot L_{\text{critic}}$$

Assume $c = 0.5$:

$$L = 3.81 + 0.5 \times 7.56 = 7.59$$

---

## 9. Why Log Probability Appears (Intuition)

**Policy gradient theorem:**

$$\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A \right]$$

**Log converts:**
- Multiplication → addition
- Probability → stable gradient

---

## 10. End-to-End A2C Math Flow (One Trajectory)

```
State s
    ↓
Actor picks action a ~ π(a|s)
    ↓
Environment returns reward r
    ↓
Compute return R
    ↓
Critic predicts V(s)
    ↓
Advantage A = R − V(s)
    ↓
Actor update: −log π(a|s) · A
Critic update: (R − V(s))²
```

---

## 11. Why A2C Reduces Variance (Important)

**Without critic:**

$$\log \pi(a \mid s) \cdot R$$

**With critic:**

$$\log \pi(a \mid s) \cdot (R - V(s))$$

**Subtracting $V(s)$:**
- Centers learning signal
- Reduces noise
- Speeds up learning

---

## 12. Key Mathematical Intuition to Remember ⭐

> - **Actor** learns direction (policy)
> - **Critic** learns magnitude (value)
> - **Advantage** connects them

---

## 13. Summary of All Equations

| Component | Equation |
|-----------|----------|
| **Return** | $R_t = \sum \gamma^k r_{t+k}$ |
| **Value** | $V(s)$ |
| **Advantage** | $A = R - V(s)$ |
| **Actor Loss** | $-\log \pi(a \mid s) \cdot A$ |
| **Critic Loss** | $(R - V(s))^2$ |
