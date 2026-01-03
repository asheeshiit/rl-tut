# Difference between Categorical DQN (C51) and QR-DQN (Quantile Regression)

Both **C51** and **QR-DQN** are distributional variants of DQN: instead of learning only the mean action-value $Q(s,a) = \mathbb{E}[G]$, they learn (an approximation to) the full return distribution $Z(s,a)$.

> **The "exact difference"** is mainly what parametric family they use for $Z(s,a)$ and therefore what loss/projection they use to do Bellman updates.

---

## 1. What They Parameterize: Probabilities vs Locations

### C51 (Categorical DQN)

C51 represents $Z(s,a)$ as a **categorical distribution** over a fixed set of value "atoms":

$$z_i = V_{\min} + i \cdot \Delta z, \quad i = 0, \ldots, N-1$$

$$Z_\theta(s,a) = \sum_{i=0}^{N-1} p_{\theta,i}(s,a) \cdot \delta_{z_i}$$

- The support locations $\{z_i\}$ are **fixed in advance**
- The network outputs the **probabilities** $p_{\theta,i}(s,a)$ (via a softmax)
- Expected value used for greedy action selection:

$$Q(s,a) = \mathbb{E}[Z(s,a)] = \sum_i p_{\theta,i}(s,a) \cdot z_i$$

This "fixed support + learned probabilities" design is central to C51.

---

### QR-DQN (Quantile Regression DQN)

QR-DQN represents $Z(s,a)$ using a **uniform mixture of Dirac masses** at learned locations, intended to match quantiles:

$$\tau_i = \frac{i - \frac{1}{2}}{N}, \quad i = 1, \ldots, N$$

$$Z_\theta(s,a) = \frac{1}{N} \sum_{i=1}^{N} \delta_{\theta_i(s,a)}$$

- The probabilities are **fixed at** $1/N$
- The network outputs the **locations** $\theta_i(s,a)$, where $\theta_i$ is trained to approximate the $\tau_i$-quantile of the return distribution
- Expected value:

$$Q(s,a) = \mathbb{E}[Z(s,a)] \approx \frac{1}{N} \sum_i \theta_i(s,a)$$

So QR-DQN is "fixed quantile levels + learned atom locations".

---

### One-Line Intuition

| Algorithm | What's Fixed | What's Learned |
|-----------|--------------|----------------|
| **C51** | x-axis (values) | y-axis (probabilities) |
| **QR-DQN** | y-axis (cumulative probabilities / quantile levels) | x-axis (values) |

---

## 2. How the Bellman Target is Handled: Projection vs No Projection

### C51 Needs an Explicit Projection

A distributional Bellman backup shifts/scales the next distribution:

$$Z_{\text{target}} \stackrel{D}{=} r + \gamma \cdot Z(s', a^*)$$

But $Z_{\text{target}}$ generally won't lie on C51's fixed support $\{z_i\}$. So C51 applies a **projection operator** that redistributes probability mass onto the fixed atoms (with clipping to $[V_{\min}, V_{\max}]$). This projection step is a defining part of the method.

---

### QR-DQN Does Not Need a Support Projection

QR-DQN doesn't require projecting onto fixed value atoms because the locations $\theta_i$ can move. The target is formed as shifted/scaled target quantile locations:

$$r + \gamma \cdot \theta_j(s', a^*)$$

Training directly nudges the predicted $\theta_i(s,a)$ toward the target distribution via quantile regression.

---

## 3. The Learning Objective: Cross-Entropy/KL vs Quantile Regression (Wasserstein)

### C51 Loss: Cross-Entropy on Categorical Distributions

After projecting $Z_{\text{target}}$ onto the fixed support to get a target categorical distribution $m$ over atoms, C51 trains by minimizing the **cross-entropy** (equivalently KL up to constants) between the target and predicted distributions:

$$\mathcal{L}_{\text{C51}} = -\sum_i m_i \log p_{\theta,i}(s,a)$$

*(Practical view: it's a classification loss over atoms.)*

---

### QR-DQN Loss: Quantile Regression (Huber) ≈ Wasserstein-1 Minimization

QR-DQN uses the **quantile regression loss** (often the "quantile Huber" variant) across all pairs of predicted and target quantile samples:

$$\mathcal{L}_{\text{QR}} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \rho_{\tau_i}^\kappa \left( \underbrace{(r + \gamma \cdot \theta'_j(s', a^*)) - \theta_i(s,a)}_{u_{ij}} \right)$$

where $\rho$ is the tilted (quantile) Huber loss.

> **Key motivation:** This gives a stochastic-approximation-friendly way to do distributional RL aligned with the **Wasserstein metric**, closing the gap left by C51's approach.

---

## 4. Key Practical Implications

### Hyperparameters / Constraints

| Aspect | C51 | QR-DQN |
|--------|-----|--------|
| **Required hyperparameters** | $[V_{\min}, V_{\max}]$ and number of atoms $N$ (51 in original) | Number of quantiles $N$ and Huber threshold $\kappa$ |
| **Sensitivity** | Bad bounds can clip returns or waste resolution | Does not need $[V_{\min}, V_{\max}]$ for fixed support |

---

### What the Network Outputs Look Like

| Algorithm | Network Output |
|-----------|----------------|
| **C51** | Logits → softmax probabilities per atom (per action) |
| **QR-DQN** | Real-valued quantile locations (per action), typically no softmax |

---

### "Information Allocation"

- **C51** spends representational capacity learning *where the mass goes* on a preset grid
- **QR-DQN** spends capacity learning *where the quantiles lie* (mass is implicitly uniform across the learned points)

---

## 5. Summary Table: Exact Differences

| Aspect | C51 (Categorical) | QR-DQN (Quantile Regression) |
|--------|-------------------|------------------------------|
| **Distribution family** | Categorical over fixed value atoms $\{z_i\}$ | Uniform mixture over learned atom locations $\{\theta_i\}$ at fixed $\{\tau_i\}$ |
| **What NN outputs** | Probabilities $p_i(s,a)$ | Quantile values $\theta_i(s,a)$ |
| **Target handling** | Projects Bellman-updated distribution onto fixed support | No fixed support projection; quantile targets shift/scale directly |
| **Loss function** | Cross-entropy / KL on categorical distributions | Quantile regression (often quantile Huber), tied to Wasserstein-1 formulation |
| **Needs $[V_{\min}, V_{\max}]$?** | ✅ Yes | ❌ No (in the core formulation) |

---

All of that is consistent with how the two methods are introduced in the original distributional RL paper (C51) and the QR-DQN paper.
