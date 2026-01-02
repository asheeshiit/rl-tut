# Behavioral Cloning (BC) — tf-agents

## 1. Behavioral Cloning in Layman Terms

Imagine you want to teach a robot to drive a car.

Instead of letting it crash many times (reinforcement learning), you:
- Record how a human drives
- Copy the human's actions

The robot does not understand rewards or consequences. It simply learns:

> "When I see this situation, the human usually does this action."

This is exactly **Behavioral Cloning**:
- Supervised learning
- Learn a policy by imitating expert demonstrations

---

## 2. Where Behavioral Cloning Fits in RL

| Method                  | Learns from            | Uses reward? |
|-------------------------|------------------------|--------------|
| Reinforcement Learning  | Trial & error          | ✅ Yes       |
| Imitation Learning      | Expert data            | ❌ No        |
| Behavioral Cloning      | Expert state-action pairs | ❌ No     |

Behavioral Cloning treats policy learning as a **classification / regression problem**.

---

## 3. Behavioral Cloning in tf-agents

In tf-agents, BC is implemented as:

```
tf_agents.agents.behavioral_cloning.behavioral_cloning_agent.BehavioralCloningAgent
```

It learns a policy `π(a | s)` by minimizing prediction error between:
- Expert action
- Policy's predicted action

---

## 4. Core Idea (One Sentence)

Learn a policy `π_θ` that maximizes the likelihood of expert actions given states.

---

## 5. Formal Setup

**Given:** A dataset of expert demonstrations:

```
D = {(s₁, a₁), (s₂, a₂), ..., (sₙ, aₙ)}
```

Where:
- `sᵢ` = state (observation)
- `aᵢ` = expert action
- `π_θ(a | s)` = policy with parameters θ

---

## 6. Objective Function (Core Math)

### Maximum Likelihood Estimation (MLE)

We want the policy to assign high probability to expert actions:

```
         N
max_θ   ∏  π_θ(aᵢ | sᵢ)
        i=1
```

Taking log (easier to optimize):

```
         N
max_θ   Σ  log π_θ(aᵢ | sᵢ)
        i=1
```

Equivalent to minimizing **negative log-likelihood**:

```
L(θ) = -E_(s,a)~D [ log π_θ(a | s) ]
```

This is the **Behavioral Cloning loss**.

---

## 7. Loss Functions by Action Type

### Case 1: Discrete Actions (Most Common)

Policy outputs a softmax distribution.

**Loss = Cross-Entropy**

```
L = -Σᵢ log π_θ(aᵢ | sᵢ)
```

Same as classification.

### Case 2: Continuous Actions

Policy outputs a Gaussian distribution:

```
π_θ(a | s) = N(μ_θ(s), Σ)
```

**Loss = Negative Log Likelihood**

```
L = (1/2)(a - μ_θ(s))ᵀ Σ⁻¹ (a - μ_θ(s)) + (1/2) log|Σ|
```

If `Σ` is fixed → this becomes **Mean Squared Error (MSE)**.

---

## 8. Algorithm (Step-by-Step)

### Behavioral Cloning Algorithm

1. **Collect** expert trajectories: `(s₁, a₁), (s₂, a₂), ...`
2. **Initialize** policy network `π_θ`
3. **Repeat** until convergence:
   - Sample batch from expert data
   - Predict actions using policy
   - Compute loss: `-log π_θ(a | s)`
   - Update θ via gradient descent

---

## 9. How tf-agents Implements This

### Key Components

- **Policy network:** maps state → action distribution
- **BehavioralCloningAgent:**
  - Wraps the policy
  - Defines loss
  - Handles training loop

### Pseudocode (Simplified)

```python
logits = policy_network(states)
loss = cross_entropy(expert_actions, logits)
optimizer.minimize(loss)
```

---

## 10. Full Worked Example (Discrete Actions)

### Example Environment

- **State:** traffic light color `s ∈ {Red, Green}`
- **Actions:**
  - 0 = Stop
  - 1 = Go

### Expert Data

| State | Action    |
|-------|-----------|
| Red   | Stop (0)  |
| Red   | Stop (0)  |
| Green | Go (1)    |
| Green | Go (1)    |

### Policy Model

Softmax policy:

```
π_θ(a | s) = softmax(W·s + b)
```

### Forward Pass (Example)

For **Red**:

```
π_θ(a | Red) = [0.9, 0.1]
```

Expert action = Stop (0)

### Loss Computation

```
L = -log(0.9) = 0.105
```

For **Green**:

```
π_θ(a | Green) = [0.2, 0.8]
```

Expert action = Go (1)

```
L = -log(0.8) = 0.223
```

### Total Loss

```
L_total = (0.105 + 0.223) / 2 = 0.164
```

Gradients update θ to increase:
- `P(Stop | Red)`
- `P(Go | Green)`

---

## 11. Continuous Action Example

### Robot Steering Angle

- **State:** camera image
- **Action:** steering angle ∈ ℝ (real numbers)

Policy outputs:

```
μ_θ(s) = 5°
```

Expert action:

```
a = 7°
```

Loss (MSE):

```
(7 - 5)² = 4
```

Gradient pushes mean toward expert value.

---

## 12. Why Behavioral Cloning Fails (Important)

### Distribution Shift Problem

- Policy only sees expert states
- Small mistake → new unseen state
- Errors compound over time

This is why BC alone performs poorly for long horizons.

---

## 13. When to Use Behavioral Cloning

✅ **Good for:**
- Strong expert data
- Short-horizon tasks
- Pretraining RL policies

❌ **Bad for:**
- Long sequential decision tasks
- Noisy expert demonstrations

---

## 14. Summary (One-Page Mental Model)

- **BC = supervised learning for policies**
- Learns `π(a | s)` by copying experts
- Loss = negative log-likelihood
- No reward, no environment interaction
- Simple, fast, but suffers from compounding errors

---

---

# Behavioral Cloning (BC) vs DAgger — Mathematical Comparison

## 1. Common Setup & Notation

### MDP (Unknown Dynamics)

- **States:** `s ∈ S`
- **Actions:** `a ∈ A`
- **Horizon:** `T`

| Symbol | Description |
|--------|-------------|
| `π*(a \| s)` | Expert policy |
| `π_θ(a \| s)` | Learned policy |
| `ℓ(π_θ(s), π*(s))` | Loss function (per state) |

Loss can be:
- 0–1 loss (classification), or
- Negative log-likelihood / MSE

---

## 2. Behavioral Cloning (BC)

### 2.1 Data Distribution

BC trains only on **expert state distribution**:

```
s ~ d_π*
```

where:

```
d_π(s) = (1/T) Σₜ₌₁ᵀ P(sₜ = s | π)
```

### 2.2 Objective Function

```
min_θ  E_{s ~ d_π*} [ ℓ(π_θ(s), π*(s)) ]
```

This is **pure supervised learning**.

### 2.3 Error Compounding (Key Math Result)

Assume:
- Per-state classification error ≤ ε

Then expected total cost over horizon T:

```
E[J(π_θ)] ≤ J(π*) + O(T² · ε)
```

**Why T²?**

- At time step 1: error probability = ε
- One mistake leads to out-of-distribution states
- Probability of being off-distribution grows linearly in t
- Total error accumulates **quadratically**

### 2.4 Intuition

```
BC trains on: d_π*
Policy runs on: d_π_θ

These are NOT the same! → Distribution Shift
```

---

## 3. DAgger (Dataset Aggregation)

### 3.1 Key Idea

Train on the state distribution of the **learned policy**, but label actions using the **expert**.

### 3.2 Iterative Data Collection

At iteration `i`:

1. **Execute** mixed policy:
   ```
   πᵢ = βᵢ · π* + (1 - βᵢ) · π_θ
   ```

2. **Collect states:**
   ```
   s ~ d_πᵢ
   ```

3. **Query expert for labels:**
   ```
   a = π*(s)
   ```

4. **Aggregate dataset:**
   ```
   D ← D ∪ (s, a)
   ```

### 3.3 Objective Function

Final policy solves:

```
min_θ  E_{s ~ d_π_θ} [ ℓ(π_θ(s), π*(s)) ]
```

⚠️ **This is the critical difference.**

---

## 4. Error Bounds — Side by Side

| Method  | Training Distribution | Performance Bound |
|---------|----------------------|-------------------|
| BC      | `d_π*`               | `O(T² · ε)`       |
| DAgger  | `d_π_θ`              | `O(T · ε)`        |

---

## 5. Why DAgger Improves the Bound (Math Intuition)

### BC

Probability of mistake at time t:

```
P(mistake at t) ≤ t · ε
```

Total mistakes:

```
  T
  Σ  t·ε = O(T² · ε)
 t=1
```

### DAgger

Policy is trained on its own induced states:

```
s ~ d_π_θ
```

So error probability is **stationary**:

```
P(mistake at t) ≤ ε   (constant for all t)
```

Total mistakes:

```
  T
  Σ  ε = O(T · ε)
 t=1
```

---

## 6. Formal Reduction View (DAgger as No-Regret Learning)

DAgger reduces imitation learning to **online learning**.

### Regret Bound

Let `ℓᵢ(θ)` be loss on iteration i.

If the learner has **no-regret**:

```
(1/N) Σᵢ₌₁ᴺ ℓᵢ(θᵢ)  ≤  min_θ (1/N) Σᵢ₌₁ᴺ ℓᵢ(θ) + o(1)
```

Then:

```
E[J(π_θ)] ≤ J(π*) + O(T · ε)
```

BC does not satisfy no-regret conditions due to fixed data distribution.

---

## 7. Loss Functions (Same, Distribution Differs)

Both use the same per-state loss:

### Discrete Actions

```
ℓ = -log π_θ(a* | s)
```

### Continuous Actions

```
ℓ = ‖a* - μ_θ(s)‖²
```

**Difference is only in state distribution.**

---

## 8. Algorithm Comparison (Math-Oriented)

### Behavioral Cloning

```
θ* = argmin_θ  E_{s ~ d_π*} [ ℓ(π_θ(s), π*(s)) ]
```

### DAgger

```
θ* = argmin_θ  E_{s ~ d_π_θ} [ ℓ(π_θ(s), π*(s)) ]
```

---

## 9. Practical Implication in tf-agents Terms

| Aspect                | BC        | DAgger    |
|-----------------------|-----------|-----------|
| Expert queries        | Once      | Repeated  |
| Environment interaction | ❌      | ✅        |
| Distribution shift    | Severe    | Controlled |
| Horizon scaling       | Quadratic | Linear    |
| Complexity            | Simple    | Higher    |

---

## 10. One-Line Takeaway

> **BC** minimizes supervised error on **expert states**; **DAgger** minimizes supervised error on the **learner's own states** — and that single change improves error from **O(T²)** to **O(T)**.
