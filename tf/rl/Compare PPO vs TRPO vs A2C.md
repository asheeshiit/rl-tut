# Compare PPO vs TRPO vs A2C

## 1. One-line Intuition

| Algorithm | Intuition |
|-----------|-----------|
| **A2C**   | "Update policy directly using advantage" |
| **TRPO**  | "Update policy safely using strict constraints" |
| **PPO**   | "Update policy safely using a simple penalty (clipping)" |

---

## 2. High-level Comparison Table

| Aspect | A2C | TRPO | PPO |
|--------|-----|------|-----|
| Type | Actor–Critic | Policy Optimization | Policy Optimization |
| Stability | ❌ Low–Medium | ✅ Very High | ✅ High |
| Complexity | ✅ Simple | ❌ Very Complex | ✅ Simple |
| Sample Efficiency | ❌ Low | ✅ High | ✅ High |
| Implementation | Easy | Hard | Easy |
| Uses second-order info | ❌ No | ✅ Yes | ❌ No |
| On-policy | ✅ Yes | ✅ Yes | ✅ Yes |
| Industry usage | Medium | Low | ⭐ Very High |

---

## 3. A2C (Advantage Actor-Critic)

### Core Idea

- **Actor** updates policy using advantage
- **Critic** estimates value

### Objective

$$L_{A2C} = \mathbb{E} \left[ \log \pi(a \mid s) \cdot A(s, a) \right]$$

### Problem

- No restriction on update size
- Large gradient steps → unstable learning

### When A2C Fails

- Continuous action spaces
- Long horizons
- High variance environments

---

## 4. TRPO (Trust Region Policy Optimization)

### Core Idea

> "Never change the policy too much in one step."

### Objective

$$\max_{\theta} \; \mathbb{E} \left[ r(\theta) \cdot A \right]$$

### Subject to Constraint

$$\mathbb{E} \left[ KL(\pi_{old} \| \pi_{\theta}) \right] \leq \delta$$

**Meaning:** Policy change must stay inside a *trust region*

### How It's Enforced

- KL-divergence constraint
- Conjugate gradient
- Fisher Information Matrix

### Pros

- ✅ Extremely stable
- ✅ Theoretically sound

### Cons

- ❌ Hard to implement
- ❌ Slow
- ❌ Computationally heavy

---

## 5. PPO (Proximal Policy Optimization)

### Core Idea

> "Approximate TRPO, but make it simple."

### Objective (Clipped)

$$L_{CLIP} = \mathbb{E} \left[ \min \left( r \cdot A, \; \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A \right) \right]$$

Where:
- \( r = \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \) — probability ratio
- \( A \) — advantage estimate
- \( \epsilon \) — clipping hyperparameter (typically 0.1–0.2)

### Key Difference from TRPO

| TRPO | PPO |
|------|-----|
| Hard KL constraint | Soft clipping |
| Complex math | Simple SGD |
| Second-order | First-order |

---

## 6. Mathematical Comparison (Side-by-Side)

| Component | A2C | TRPO | PPO |
|-----------|-----|------|-----|
| Policy ratio | ❌ No | ✅ Yes | ✅ Yes |
| Advantage | ✅ | ✅ | ✅ |
| Update constraint | ❌ None | ✅ KL | ✅ Clipping |
| Gradient type | First-order | Second-order | First-order |

---

## 7. Stability Intuition (Visual)

```
A2C:   🚀  (big jumps → unstable)
TRPO:  🧱  (hard wall → safe but slow)
PPO:   🧸  (soft padding → safe & fast)
```

---

## 8. Performance Trade-offs

### A2C
- ✅ Fast per step
- ❌ Often diverges
- ❌ Sensitive to hyperparameters

### TRPO
- ✅ Very stable
- ❌ Hard to tune
- ❌ Expensive computation

### PPO
- ✅ Stable
- ✅ Efficient
- ✅ Easy to tune
- ⭐ **Best practical choice**

---

## 9. When to Use Which?

### Use A2C if:
- Simple environment
- Discrete actions
- Learning stability not critical

### Use TRPO if:
- Research / theory work
- You need guaranteed monotonic improvement
- Compute cost not an issue

### Use PPO if:
- Real-world problems
- Robotics
- Games
- RLHF
- You want best trade-off

---

## 10. Final Takeaway

| Algorithm | Summary |
|-----------|---------|
| A2C | Learns fast but **unstable** |
| TRPO | Learns safely but **expensively** |
| PPO | Learns safely and **efficiently** |

> ⭐ **Industry rule of thumb:**  
> *If unsure → use PPO*
