# Categorical DQN (C51)

## 1. Categorical DQN in Layman Terms 🧠

### What Vanilla DQN Does

A normal DQN learns a single number for each action:

> "If I take action $a$ in state $s$, my expected (average) future reward is $Q(s,a)$."

That's like predicting only the **average** outcome.

### What Categorical DQN Does

Categorical DQN predicts a **full distribution** of possible outcomes:

> "If I take action $a$ in state $s$, there's a probability of getting a small return, a medium return, or a large return."

That's like a weather forecast that says:
- 20% chance of light rain
- 50% chance of moderate rain
- 30% chance of heavy rain

Instead of just saying: *"expected rain = 5mm"*

**So Categorical DQN learns a histogram of returns, not just the mean.**

---

## 2. What "Distribution of Return" Means

Define the return from time $t$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

In **classic DQN**:

$$Q(s,a) = \mathbb{E}[G_t \mid s_t = s, a_t = a]$$

In **distributional RL** (Categorical DQN):
- We treat the return as a random variable $Z(s,a)$
- DQN learns only $\mathbb{E}[Z]$
- Categorical DQN learns (an approximation of) the **entire distribution** of $Z$

---

## 3. What "Categorical" Means (The C51 Idea)

Categorical DQN represents the return distribution using **fixed support points ("atoms")**.

Choose:
- $v_{\min}$ and $v_{\max}$ (min/max possible return you care about)
- Number of atoms $N$

Define atoms:

$$z_i = v_{\min} + i \cdot \Delta, \quad i = 0, \ldots, N-1$$

$$\Delta = \frac{v_{\max} - v_{\min}}{N - 1}$$

Now the distribution for a given $(s,a)$ is:

$$Z_\theta(s,a) \approx \sum_{i=0}^{N-1} p_\theta(i \mid s,a) \cdot \delta(z - z_i)$$

**Meaning:**
- Returns take values only on the grid $\{z_i\}$
- With probabilities $p_\theta(i \mid s,a)$

✅ The network predicts $p_\theta(i \mid s,a)$ — a probability for each atom.

---

## 4. Neural Network Output

If there are $A$ discrete actions and $N$ atoms:
- Network outputs logits of shape $[A \times N]$
- For each action $a$, apply softmax across atoms:

$$p_\theta(i \mid s,a) = \text{softmax}(\text{logits}(s,a))_i$$

### Getting a Q-value from the Distribution (for action selection)

The expected value is:

$$Q_\theta(s,a) = \mathbb{E}[Z_\theta(s,a)] = \sum_{i=0}^{N-1} z_i \cdot p_\theta(i \mid s,a)$$

So Categorical DQN can still do greedy action selection like DQN:

$$a^* = \arg\max_a Q_\theta(s,a)$$

---

## 5. The Distributional Bellman Backup (Core Math)

**Classic Bellman optimality for DQN:**

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

**Distributional version** says the random return satisfies:

$$Z(s,a) \stackrel{D}{=} r + \gamma Z(s', a^*)$$

Where:
- $\stackrel{D}{=}$ means "equal in distribution"
- $a^*$ is the greedy next action (based on expected value)

---

## 6. The Big Issue: Support Must Stay Fixed → Projection

After applying the backup:

$$Tz_j = r + \gamma z_j$$

These values generally **do not land exactly** on the fixed grid $\{z_i\}$, and may go outside $[v_{\min}, v_{\max}]$.

So Categorical DQN does two steps:

### Step A: Clip into Range

$$\hat{T}z_j = \text{clip}(Tz_j, v_{\min}, v_{\max})$$

### Step B: Project Back onto the Atom Grid

For each transformed atom $\hat{T}z_j$, compute:

$$b_j = \frac{\hat{T}z_j - v_{\min}}{\Delta}$$

$$l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil$$

Then distribute probability mass $p_j$ to the nearest bins:

- If $l = u$:
  $$m_l \mathrel{+}= p_j$$

- Else:
  $$m_l \mathrel{+}= p_j (u - b_j)$$
  $$m_u \mathrel{+}= p_j (b_j - l)$$

This creates the **projected target distribution** $m$ over the fixed atoms.

---

## 7. Target Distribution + Double DQN Style Action Choice

Most implementations use a "Double DQN" style split:

1. **Choose next action** using online network:
   $$a^* = \arg\max_a \sum_i z_i \cdot p_\theta(i \mid s', a)$$

2. **Evaluate the distribution** using target network:
   $$p_{\bar{\theta}}(\cdot \mid s', a^*)$$

Then apply shift + discount + projection to get $m$.

---

## 8. Loss Function (How Training Happens)

For the sampled transition $(s, a, r, s', \text{done})$, after computing target distribution $m$:

**Predicted distribution for chosen action:**
$$p_\theta(\cdot \mid s, a)$$

**Use cross-entropy loss:**

$$L(\theta) = -\sum_{i=0}^{N-1} m_i \log p_\theta(i \mid s, a)$$

This pushes the network's predicted histogram to match the projected target histogram.

---

## 9. Full Algorithm (High-Level)

1. Initialize online network $\theta$ and target network $\bar{\theta}$
2. Collect transitions into replay buffer
3. Sample minibatch
4. For each sample:
   - Pick $a^*$ using online net expected values
   - Get target distribution from target net for $a^*$
   - Shift + discount atoms → clip → project → $m$
   - Compute cross-entropy loss vs predicted distribution for executed action
5. Gradient step on $\theta$
6. Periodically update $\bar{\theta} \leftarrow \theta$

---

## 10. Worked Numeric Example (Projection Step-by-Step) ✅

To keep it simple, use 5 atoms (real C51 uses 51 atoms).

### Setup

| Parameter | Value |
|-----------|-------|
| $N$ | 5 |
| $v_{\min}$ | 0 |
| $v_{\max}$ | 4 |
| Atoms $z$ | $[0, 1, 2, 3, 4]$ |
| $\Delta$ | 1 |
| Reward $r$ | 1 |
| Discount $\gamma$ | 0.9 |
| Terminal | No |

Suppose at next state $s'$, for the greedy action $a^*$, the target network predicts:

$$p(s', a^*) = [0.1, 0.2, 0.4, 0.2, 0.1]$$

Probability mass on atoms:

| Atom Value | Probability |
|------------|-------------|
| 0 | 0.1 |
| 1 | 0.2 |
| 2 | 0.4 |
| 3 | 0.2 |
| 4 | 0.1 |

---

### Step 1: Shift + Discount the Atoms

$$Tz = r + \gamma z = 1 + 0.9z$$

| Original $z$ | Transformed $Tz$ |
|--------------|------------------|
| 0 | 1.0 |
| 1 | 1.9 |
| 2 | 2.8 |
| 3 | 3.7 |
| 4 | 4.6 |

---

### Step 2: Clip into $[0, 4]$

$$\hat{T}z = [1.0, 1.9, 2.8, 3.7, 4.0]$$

*(4.6 becomes 4.0)*

---

### Step 3: Project onto Atoms $[0, 1, 2, 3, 4]$

Since $\Delta = 1$, we have $b = \hat{T}z$.

**For 1.0 with mass 0.1:**
- $b = 1.0$, $l = u = 1$
- All 0.1 goes to atom 1

**For 1.9 with mass 0.2:**
- $b = 1.9$, $l = 1$, $u = 2$
- To atom 1: $(2 - 1.9) = 0.1$ → $0.2 \times 0.1 = 0.02$
- To atom 2: $(1.9 - 1) = 0.9$ → $0.2 \times 0.9 = 0.18$

**For 2.8 with mass 0.4:**
- $l = 2$, $u = 3$
- To atom 2: $(3 - 2.8) = 0.2$ → $0.4 \times 0.2 = 0.08$
- To atom 3: $(2.8 - 2) = 0.8$ → $0.4 \times 0.8 = 0.32$

**For 3.7 with mass 0.2:**
- $l = 3$, $u = 4$
- To atom 3: $(4 - 3.7) = 0.3$ → $0.2 \times 0.3 = 0.06$
- To atom 4: $(3.7 - 3) = 0.7$ → $0.2 \times 0.7 = 0.14$

**For 4.0 with mass 0.1:**
- $l = u = 4$
- All 0.1 goes to atom 4

---

### Final Projected Target Distribution $m$

Add up mass per atom:

| Atom | Mass Calculation | Total |
|------|------------------|-------|
| 0 | — | 0 |
| 1 | $0.1 + 0.02$ | 0.12 |
| 2 | $0.18 + 0.08$ | 0.26 |
| 3 | $0.32 + 0.06$ | 0.38 |
| 4 | $0.14 + 0.10$ | 0.24 |

$$m = [0, 0.12, 0.26, 0.38, 0.24]$$

*(Checks out: sums to 1)*

---

### Expected Value of Target Distribution

$$\mathbb{E}[Z] = 0(0) + 1(0.12) + 2(0.26) + 3(0.38) + 4(0.24) = 2.74$$

---

## 11. Example Loss Computation (Cross-Entropy)

Suppose the online network currently predicts for the taken action $(s, a)$:

$$p_\theta(\cdot \mid s, a) = [0.05, 0.15, 0.50, 0.20, 0.10]$$

**Loss:**

$$L = -\sum_i m_i \log p_\theta(i \mid s, a)$$

Plugging in gives a positive scalar loss (about **1.57 nats**), and training adjusts logits so predicted distribution moves toward $m$.

---

## 12. Why Categorical DQN Can Work Better Than DQN

- A single mean value can hide important structure (risk, multi-modal outcomes)
- Learning the whole distribution often yields **better learning signals** and **performance**
- Still keeps DQN's nice properties: replay buffer + target network + greedy action selection
