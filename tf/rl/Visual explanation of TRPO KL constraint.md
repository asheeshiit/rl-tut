# Visual Explanation of TRPO KL Constraint

## 1. What the KL Constraint is "Visually" Trying to Do

TRPO solves the following constrained optimization problem:

$$
\max_{\theta} \mathbb{E}\left[ r(\theta) \cdot A \right] \quad \text{subject to} \quad \mathbb{E}\left[ KL\left( \pi_{\text{old}}(\cdot | s) \,\|\, \pi_{\theta}(\cdot | s) \right) \right] \leq \delta
$$

**Meaning:**
> "Improve performance, but keep the new policy distribution close to the old policy distribution (on average over visited states)."

---

## 2. Visual 1: "Trust Region Bubble" in Policy Space

Think of each policy as a point. TRPO puts a **bubble** around the old policy and only allows updates inside it.

```
Policy space (conceptual)

          (bad big update)
                 X
                /
               /
              /
   allowed   / 
  region   ( )    ← trust region (KL ≤ δ)
          ( • )   ← old policy π_old
            \
             \
              \
               •  ← new policy chosen by TRPO (still inside)
```

- **Without** the KL constraint → optimizer might jump to "X" (big change → can collapse performance)
- **With** the KL constraint → TRPO must pick a point inside the circle

---

## 3. Visual 2: Discrete Actions as a "Probability Bar Chart"

Suppose at some state $s$, you have 2 actions: **Left** and **Right**.

### Old Policy

$$
\pi_{\text{old}}(\cdot | s) = [0.2, \; 0.8]
$$

```
Old policy π_old
Left   |■■
Right  |■■■■■■■■
       0        1
```

### A Risky Proposed New Policy (Big Change)

$$
\pi_{\text{new}}(\cdot | s) = [0.6, \; 0.4]
$$

```
New policy π_new (big jump)
Left   |■■■■■■
Right  |■■■■
       0        1
```

> Even if "Left" looked good in your recent batch, jumping from **0.2 → 0.6** is huge. TRPO says: **don't do that all at once.**

---

## 4. What KL is Measuring (Intuitive Meaning)

For one state $s$:

$$
KL(\pi_{\text{old}} \| \pi_{\text{new}}) = \sum_{a} \pi_{\text{old}}(a | s) \log \frac{\pi_{\text{old}}(a | s)}{\pi_{\text{new}}(a | s)}
$$

A very useful interpretation:

$$
KL(\pi_{\text{old}} \| \pi_{\text{new}}) = \mathbb{E}_{a \sim \pi_{\text{old}}} \left[ \log \pi_{\text{old}}(a | s) - \log \pi_{\text{new}}(a | s) \right]
$$

**So KL constraint means:**
> "Under actions that the old policy actually takes, don't change the log-probabilities too much on average."

If the new policy makes old-policy actions suddenly much less likely, KL shoots up.

---

## 5. Numeric Example: KL Detects "Too Big a Jump"

### Case A: Big Change (Likely Violates Constraint)

- **Old:** $[0.2, 0.8]$
- **New:** $[0.6, 0.4]$

**Compute KL:**

$$
KL = 0.2 \log \frac{0.2}{0.6} + 0.8 \log \frac{0.8}{0.4}
$$

Now compute each part:

$$
\log(0.2 / 0.6) = \log(1/3) \approx -1.0986
$$

$$
\log(0.8 / 0.4) = \log(2) \approx 0.6931
$$

$$
KL \approx 0.2 \times (-1.0986) + 0.8 \times (0.6931)
$$

$$
KL \approx -0.2197 + 0.5545 = \mathbf{0.3348}
$$

If TRPO's $\delta = 0.01$, then:

$$
0.3348 > 0.01 \quad \Rightarrow \quad \text{update is too large (reject/scale back)}
$$

---

### Case B: Small, Safe Change (Fits Inside Trust Region)

- **Old:** $[0.2, 0.8]$
- **New:** $[0.25, 0.75]$

**Compute KL:**

$$
KL = 0.2 \log \frac{0.2}{0.25} + 0.8 \log \frac{0.8}{0.75}
$$

Compute:

$$
\log(0.2 / 0.25) = \log(0.8) \approx -0.2231
$$

$$
\log(0.8 / 0.75) = \log(1.0667) \approx 0.0645
$$

$$
KL \approx 0.2 \times (-0.2231) + 0.8 \times (0.0645)
$$

$$
KL \approx -0.0446 + 0.0516 = \mathbf{0.0070}
$$

Now:

$$
0.0070 \leq 0.01 \quad \Rightarrow \quad \text{safe update (allowed)}
$$

> This is exactly the behavior TRPO wants: **improve, but in small controlled steps.**

---

## 6. Visual 3: Why "Ratio Changes" Increase KL

Define the probability ratio:

$$
r(a) = \frac{\pi_{\text{new}}(a | s)}{\pi_{\text{old}}(a | s)}
$$

If some action's ratio becomes **extreme** (very big or very small), KL increases.

For the big jump example:

| Action | Ratio | Change |
|--------|-------|--------|
| Left   | $r = 0.6 / 0.2 = 3$ | tripled |
| Right  | $r = 0.4 / 0.8 = 0.5$ | halved |

That kind of **"probability shock"** is exactly what KL penalizes.

---

## 7. Practical Picture: How TRPO Uses It During Update

TRPO typically:

1. Finds a direction to improve reward (gradient direction)
2. Proposes a step
3. Checks if average $KL \leq \delta$
4. If KL too big → reduce step size (line search) until it fits

So you can imagine TRPO doing:

```
try step size = 1.0   → KL too big  ❌
try step size = 0.5   → KL still big ❌
try step size = 0.25  → KL OK       ✅
accept update
```

---

## Summary: Mental Model to Remember

| Concept | Analogy |
|---------|---------|
| **KL constraint** | "Speed limit" on policy change |
| **Not a reward term** | More like a **safety rule** |
| **Trust region** | Bubble around current policy |
| **δ (delta)** | Radius of the allowed bubble |
