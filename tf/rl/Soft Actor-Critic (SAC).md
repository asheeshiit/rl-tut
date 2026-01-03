# Soft Actor-Critic (SAC)

---

## 1. SAC in Layman Terms 🧠

Think of training a robot to walk:

- If you reward it **only** for moving forward, it may quickly lock into one "best guess" move and stop exploring.
- Early in training, you actually want it to try many moves to discover what works.

### ✅ SAC's Main Idea

Learn a policy that gets **high reward** and stays **a bit random** (high "curiosity/exploration").

That "randomness" is measured using **entropy**.

So SAC doesn't optimize only:
> "Get max reward"

It optimizes:
> "Get high reward **plus** keep actions uncertain enough to explore."

That's why it's called **Soft** Actor-Critic:
- **Soft** = reward + entropy (not just reward)

---

## 2. What SAC Is (Conceptually)

SAC is an **off-policy actor-critic** algorithm (very important):

| Component | Description |
|-----------|-------------|
| **Actor** | Learns a policy $\pi_\theta(a \mid s)$ (a distribution over actions) |
| **Critic(s)** | Learn action-values $Q_\phi(s, a)$ |
| **Off-policy** | Learns from a replay buffer (old experience), which makes it **sample efficient** |

> SAC is most popular for **continuous actions** (robotics/control), but the math idea is easier to show with a tiny discrete example too.

---

## 3. The "Maximum Entropy" Objective (Core SAC Math)

### Traditional RL Objective

Traditional RL tries to maximize expected discounted return:

$$
\max_\pi \; \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, a_t)\right]
$$

### SAC Objective

SAC maximizes **reward + entropy**:

$$
\max_\pi \; \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \, H(\pi(\cdot \mid s_t)) \right)\right]
$$

Where:
- $\gamma \in (0, 1)$ is the discount factor (e.g., 0.99)
- $\alpha \geq 0$ is the **temperature** (how much you value exploration)
- $H(\pi(\cdot \mid s))$ is the **entropy**

### Entropy Definition

$$
H(\pi(\cdot \mid s)) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\left[-\log \pi(a \mid s)\right]
$$

### Interpretation of α

| α Value | Effect |
|---------|--------|
| Large α | More randomness / exploration |
| Small α | More greedy / exploitation |

---

## 4. "Soft" Value Functions (Bellman Equations)

SAC uses "soft" versions of value functions.

### 4.1 Soft State Value

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a) - \alpha \log \pi(a \mid s)\right]
$$

> The extra term $-\alpha \log \pi$ is the **entropy bonus**.

### 4.2 Soft Bellman Backup

$$
Q^\pi(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s'}\left[V^\pi(s')\right]
$$

---

## 5. What SAC Learns (Practical Version Used Today)

Most modern SAC implementations use:

| Component | Purpose |
|-----------|---------|
| **Two critics:** $Q_{\phi_1}, Q_{\phi_2}$ | Reduce overestimation bias |
| **Target critics:** $Q_{\bar{\phi}_1}, Q_{\bar{\phi}_2}$ | Slow-moving copies for stability |
| **Policy (actor):** $\pi_\theta$ | The learned policy |
| **Temperature:** $\alpha$ | Often learned automatically |

---

## 6. Loss Functions (Training Math)

Assume we sample a batch $(s, a, r, s', \text{done})$ from replay buffer.

### 6.1 Critic Target (Soft Target)

Sample next action $a' \sim \pi_\theta(\cdot \mid s')$. Define:

$$
y = r + \gamma (1 - \text{done}) \left( \min_{i \in \{1, 2\}} Q_{\bar{\phi}_i}(s', a') - \alpha \log \pi_\theta(a' \mid s') \right)
$$

### 6.2 Critic Loss (MSE)

For each critic $i \in \{1, 2\}$:

$$
L_{Q_i}(\phi_i) = \mathbb{E}\left[\left(Q_{\phi_i}(s, a) - y\right)^2\right]
$$

### 6.3 Actor Loss (Policy Improvement)

SAC updates the policy to choose actions with high Q, but also keep entropy:

$$
L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \, a \sim \pi_\theta}\left[\alpha \log \pi_\theta(a \mid s) - \min_i Q_{\phi_i}(s, a)\right]
$$

**Minimizing this means:**
- Make $Q$ large (good actions)
- Make $\log \pi$ not too large (keep entropy)

### Continuous-Action Detail (Why SAC Works Well)

For continuous actions, SAC typically uses the **reparameterization trick**:

1. Sample noise $\varepsilon \sim \mathcal{N}(0, I)$
2. Set action $a = f_\theta(s, \varepsilon)$ (often Gaussian then tanh-squash)

This allows **low-variance gradients** for actor updates.

### 6.4 Temperature Loss (Auto-tuning α)

SAC often adjusts $\alpha$ to match a target entropy level. A common objective is:

$$
L_\alpha(\alpha) = \mathbb{E}_{a \sim \pi_\theta}\left[\alpha \left(-\log \pi_\theta(a \mid s) - H_{\text{target}}\right)\right]
$$

**Rule of thumb:**
- If current entropy is **below** target → **increase** α
- If entropy is **above** target → **decrease** α

> For continuous actions, note this is **differential entropy**; it can be negative—so targets may be negative too.

---

## 7. SAC Algorithm (Pseudo-code)

```
Initialize actor π_θ
Initialize critics Q_φ1, Q_φ2
Initialize target critics Q_φ̄1 ← Q_φ1, Q_φ̄2 ← Q_φ2
Initialize temperature α (or learn it)
Initialize replay buffer D

repeat (for each environment step):
    Observe state s
    Sample action a ~ π_θ(·|s)
    Execute a, get reward r, next state s', done
    Store (s, a, r, s', done) in D

    for update_step = 1..U do:
        Sample minibatch from D

        # Critic update
        Sample a' ~ π_θ(·|s')
        y = r + γ(1-done) * (min(Q_φ̄1(s',a'), Q_φ̄2(s',a')) - α log π_θ(a'|s'))
        Minimize (Q_φ1(s,a) - y)² and (Q_φ2(s,a) - y)²

        # Actor update
        Minimize E[α log π_θ(a|s) - min(Q_φ1(s,a), Q_φ2(s,a))]

        # Temperature update (optional)
        Minimize E[α(-log π_θ(a|s) - H_target)]

        # Target update (Polyak averaging)
        φ̄ ← τ φ + (1-τ) φ̄
    end
until done
```

---

## 8. Worked Numeric Example (To See the Math)

> SAC is usually continuous, but the maximum-entropy idea is easiest to see with a tiny discrete example.

### Setup: One State, Two Actions

At some state $s$, suppose the critic estimates:

| Action | Q-value |
|--------|---------|
| $a_1$ | $Q(s, a_1) = 10$ |
| $a_2$ | $Q(s, a_2) = 9$ |

So $a_1$ is slightly better, but not massively.

Let $\alpha = 1$.

---

### 8.1 What Policy Does SAC Prefer? (Softmax/Boltzmann Form)

If you maximize (at a fixed state) the "soft" objective:

$$
\max_{\pi(\cdot \mid s)} \sum_a \pi(a \mid s) \, Q(s, a) + \alpha \, H(\pi(\cdot \mid s))
$$

Using Lagrange multipliers, the **optimal policy** becomes:

$$
\pi^*(a \mid s) \propto \exp\left(\frac{Q(s, a)}{\alpha}\right)
$$

So:

$$
\pi(a_1 \mid s) = \frac{e^{10}}{e^{10} + e^9} = \frac{1}{1 + e^{-1}} \approx \boxed{0.731}
$$

$$
\pi(a_2 \mid s) = \frac{e^9}{e^{10} + e^9} \approx \boxed{0.269}
$$

✅ **Meaning:** SAC prefers $a_1$, but still keeps a decent chance for $a_2$ because of entropy.

---

### 8.2 Entropy at This State

$$
H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)
$$

Using the numbers:

| Term | Value |
|------|-------|
| $\log(0.731)$ | $\approx -0.313$ |
| $\log(0.269)$ | $\approx -1.313$ |

$$
H \approx -\left(0.731 \times (-0.313) + 0.269 \times (-1.313)\right) \approx \boxed{0.582}
$$

---

### 8.3 Soft Value V(s)

$$
V(s) = \sum_a \pi(a \mid s) \left(Q(s, a) - \alpha \log \pi(a \mid s)\right)
$$

Compute each term (with $\alpha = 1$):

| Action | $Q(s,a) - \alpha \log \pi(a \mid s)$ |
|--------|---------------------------------------|
| $a_1$ | $10 - \log(0.731) = 10 - (-0.313) = 10.313$ |
| $a_2$ | $9 - \log(0.269) = 9 - (-1.313) = 10.313$ |

So:

$$
V(s) = 0.731 \times 10.313 + 0.269 \times 10.313 = \boxed{10.313}
$$

> This also equals a known identity:
> $$V(s) = \alpha \log\left(\sum_a e^{Q(s,a)/\alpha}\right)$$

---

### 8.4 One-Step Critic Target Example

Suppose we observe a transition:

| Parameter | Value |
|-----------|-------|
| Reward $r$ | 1 |
| Discount $\gamma$ | 0.9 |
| Next state | $s' = s$ |
| Terminal | done = 0 |

Then SAC target:

$$
y = r + \gamma \, V(s') = 1 + 0.9 \times 10.313 = 1 + 9.282 = \boxed{10.282}
$$

Now if the critic currently predicts $Q(s, a_1) = 9.8$, critic loss:

$$
L_Q = (9.8 - 10.282)^2 \approx \boxed{0.232}
$$

So the critic updates to push $Q(s, a_1)$ **upward**.

---

### 8.5 Actor Loss Intuition with Numbers

Actor minimizes:

$$
L_\pi = \mathbb{E}\left[\alpha \log \pi(a \mid s) - Q(s, a)\right]
$$

At this state with our policy:

$$
L_\pi = \sum_a \pi(a \mid s) \left(\log \pi(a \mid s) - Q(s, a)\right)
$$

This loss becomes more negative when:
- You **increase probability** of high-Q actions
- But **not so aggressively** that entropy collapses (because $\log \pi$ term fights that)

✅ **This is how SAC balances exploitation vs exploration mathematically.**

---

## 9. Key Takeaway

> **SAC learns a policy that is both high-reward and high-entropy, using soft Bellman backups and off-policy actor-critic updates.**

---

## Summary Table

| Aspect | SAC Approach |
|--------|--------------|
| **Objective** | Maximize reward + entropy |
| **Policy type** | Off-policy (uses replay buffer) |
| **Action space** | Best for continuous |
| **Critics** | Twin Q-networks (clipped double Q) |
| **Exploration** | Built-in via entropy bonus |
| **Temperature α** | Often auto-tuned |
| **Key advantage** | Sample efficient + stable |
