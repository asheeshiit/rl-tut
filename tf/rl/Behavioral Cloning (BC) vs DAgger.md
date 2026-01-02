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
