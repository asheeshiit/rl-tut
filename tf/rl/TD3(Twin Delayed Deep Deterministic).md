# TD3 (Twin Delayed Deep Deterministic Policy Gradient)

## 1. TD3 in Layman Terms

Imagine you're teaching a robot car to drive.

- **The Actor** is the car's "driver": given what it sees (state), it outputs a continuous action (steering + throttle).
- **The Critic** is like a "driving coach" that scores how good an action is (how much future reward you'll get).

### What Goes Wrong in DDPG (the older method)?

DDPG uses **one critic**. If that critic is slightly wrong (too optimistic), the actor starts exploiting those mistakes:

> Critic mistakenly thinks "flooring the gas" is amazing → actor learns to floor the gas → crashes in real dynamics.

Because the actor is trained to maximize the critic's score, it tends to chase overestimated actions.

### TD3's Fix = 3 Simple Ideas

#### 1. Twin Critics (two coaches) + Believe the More Pessimistic One

TD3 trains **two critic networks**. When computing the learning target it uses:

> "Take the minimum of the two value estimates."

So if one critic is overly optimistic, the min blocks that optimism.

#### 2. Delayed Policy Updates (update the driver less often)

TD3 updates the critic every step, but updates the actor only every few steps (often every 2 steps).

This prevents the actor from chasing a critic that is still wobbling and inaccurate.

#### 3. Target Policy Smoothing (don't trust razor-thin 'perfect' actions)

When computing the target value, TD3 adds a small random noise to the next action and clips it.

This makes the critic learn a value that's smoother and prevents the actor from exploiting tiny "glitches" in the critic.

---

## 2. TD3 Algorithm (What Happens During Training)

### Networks You Maintain

| Network | Description |
|---------|-------------|
| Actor network | `μ_θ(s)` — deterministic policy |
| Two critics | `Q_φ₁(s, a)` and `Q_φ₂(s, a)` |
| Target networks | `μ_θ'`, `Q_φ₁'`, `Q_φ₂'` — slow-moving copies |
| Replay buffer | `D` storing transitions `(s, a, r, s', d)` |

### High-Level Steps Each Environment Step

1. **Act with exploration noise:**
   ```
   a = μ_θ(s) + ε_explore
   ```

2. **Store transition** `(s, a, r, s', d)` into replay buffer.

3. **Sample a minibatch** from replay buffer.

4. **Compute TD3 target** using:
   - Target policy smoothing noise
   - Min of two target critics

5. **Update both critics** by regression to that target.

6. **Every `policy_delay` steps:**
   - Update actor using critic gradient
   - Soft-update target networks (Polyak averaging)

---

## 3. The Math in Detail

### 3.1 RL Setup (MDP + Return)

| Symbol | Meaning |
|--------|---------|
| `sₜ` | State |
| `aₜ` | Action (continuous) |
| `rₜ` | Reward |
| `γ ∈ [0, 1)` | Discount factor |
| `dₜ ∈ {0, 1}` | Done flag (1 = terminal) |

**Discounted return from time t:**

$$
G_t = \sum_{k=0}^{\infty} \gamma^k \cdot r_{t+k}
$$

**Action-value function under a policy μ:**

$$
Q^{\mu}(s, a) = \mathbb{E}\left[ G_t \mid s_t = s, \, a_t = a, \, a_{t+1} = \mu(s_{t+1}), \ldots \right]
$$

### 3.2 Critic Learning: Bellman Target + Regression

For deterministic policy, the Bellman backup looks like:

$$
Q^{\mu}(s, a) \approx r + \gamma \cdot (1 - d) \cdot Q^{\mu}(s', \mu(s'))
$$

TD3 uses two critics and a **clipped double-Q target**.

#### Target Policy Smoothing

Instead of using exactly `μ_θ'(s')`, TD3 adds clipped Gaussian noise:

$$
\tilde{a}' = \text{clip}\Big( \mu_{\theta'}(s') + \epsilon, \; a_{\min}, \; a_{\max} \Big)
$$

where:

$$
\epsilon \sim \text{clip}\Big( \mathcal{N}(0, \sigma^2), \; -c, \; c \Big)
$$

- `σ` controls how much smoothing noise you add
- `c` is a hard clip so noise can't be huge

#### Clipped Double Q-Learning Target

Compute:

$$
y = r + \gamma \cdot (1 - d) \cdot \min\Big( Q_{\phi'_1}(s', \tilde{a}'), \; Q_{\phi'_2}(s', \tilde{a}') \Big)
$$

> This **min** is the twin critic core.

#### Critic Losses

Each critic is trained by **mean squared error (MSE)**:

$$
\mathcal{L}(\phi_i) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \Big( Q_{\phi_i}(s, a) - y \Big)^2 \right], \quad i \in \{1, 2\}
$$

**Gradient (for one sample):**

$$
\nabla_{\phi_i} \mathcal{L} = 2 \cdot \Big( Q_{\phi_i}(s, a) - y \Big) \cdot \nabla_{\phi_i} Q_{\phi_i}(s, a)
$$

Then do gradient descent on `φᵢ`.

### 3.3 Actor Learning: Deterministic Policy Gradient

TD3 chooses to maximize Q of the actor's actions (typically using critic 1):

**Objective:**

$$
J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right]
$$

**Using the deterministic policy gradient theorem:**

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_{\theta} \mu_{\theta}(s) \cdot \nabla_a Q_{\phi_1}(s, a) \Big|_{a = \mu_{\theta}(s)} \right]
$$

In practice, implementations often define an **actor loss** for gradient descent:

$$
\mathcal{L}_{\text{actor}}(\theta) = - \mathbb{E}_{s \sim \mathcal{D}} \left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right]
$$

and minimize that.

### 3.4 Delayed Updates

Instead of updating actor every step, TD3 updates it every `d` critic updates:

| Update Type | Frequency |
|-------------|-----------|
| Critic update | Every step |
| Actor update | Every `policy_delay` steps (commonly 2) |
| Target networks update | Usually at same time as actor update |

> This reduces instability because the critic is changing fast; the actor shouldn't chase it every single step.

### 3.5 Target Networks: Polyak Averaging (Soft Update)

After updates, TD3 slowly moves target parameters toward online parameters:

$$
\theta' \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta'
$$

$$
\phi'_i \leftarrow \tau \cdot \phi_i + (1 - \tau) \cdot \phi'_i, \quad i \in \{1, 2\}
$$

with a small `τ` (like 0.005).

---

## 4. Worked Numerical Example

A concrete example showing:
- The TD3 target `y` computation
- A tiny critic update step (with a simple linear critic)
- A tiny actor update step (with a simple linear actor)

### Example Data & Hyperparameters

Assume one sampled transition from the replay buffer:

| Variable | Value |
|----------|-------|
| s | 2 |
| a | 1 |
| r | 1 |
| s' | 1.5 |
| d | 0 (not terminal) |
| γ | 0.9 |

### Target Actor Output + Smoothing Noise

Let the target actor be very simple and output:

$$
\mu_{\theta'}(s') = 0.6
$$

Add target smoothing noise:

$$
\epsilon = 0.05 \quad \text{(already within clip)}
$$

So:

$$
\tilde{a}' = 0.6 + 0.05 = 0.65
$$

### Target Critics Output (Two Different Estimates)

Suppose the target critics predict:

| Critic | Value |
|--------|-------|
| `Q_φ₁'(s', ã')` | 2.80 |
| `Q_φ₂'(s', ã')` | 1.98 |

TD3 uses the **minimum**:

$$
\min(2.80, 1.98) = 1.98
$$

### 4.1 Compute the TD3 Target y

$$
y = r + \gamma \cdot (1 - d) \cdot \min\Big( Q_{\phi'_1}, Q_{\phi'_2} \Big)
$$

Plugging in:

$$
y = 1 + 0.9 \times 1 \times 1.98 = 1 + 1.782 = \boxed{2.782}
$$

So the critics are trained to output **2.782** for this `(s, a)`.

### 4.2 One-Step Critic Update (Simple Linear Critics)

To make the math explicit, assume critics are linear:

- **Critic 1:** `Q_φ₁(s, a) = w₁·s + v₁·a`
- **Critic 2:** `Q_φ₂(s, a) = w₂·s + v₂·a`

**Current parameters:**

| Critic | w | v |
|--------|---|---|
| Critic 1 | 0.9 | 1.9 |
| Critic 2 | 0.7 | 1.1 |

#### Current Predictions at (s=2, a=1)

**Critic 1:**

$$
Q_{\phi_1}(2, 1) = 0.9 \times 2 + 1.9 \times 1 = 1.8 + 1.9 = 3.7
$$

**Critic 2:**

$$
Q_{\phi_2}(2, 1) = 0.7 \times 2 + 1.1 \times 1 = 1.4 + 1.1 = 2.5
$$

**Target:** `y = 2.782`

**Errors:**

$$
\delta_1 = Q_{\phi_1} - y = 3.7 - 2.782 = 0.918
$$

$$
\delta_2 = Q_{\phi_2} - y = 2.5 - 2.782 = -0.282
$$

#### Gradient Descent Update (learning rate α = 0.01)

- **Loss for one sample:** `L = (Q - y)²`
- **Gradient:** `∇ = 2·(Q - y)·∇Q`

**For Critic 1:**

Derivatives:

$$
\frac{\partial Q}{\partial w_1} = s = 2, \quad \frac{\partial Q}{\partial v_1} = a = 1
$$

Update:

$$
w_1 \leftarrow w_1 - \alpha \cdot 2 \cdot \delta_1 \cdot s
$$

$$
v_1 \leftarrow v_1 - \alpha \cdot 2 \cdot \delta_1 \cdot a
$$

Plug in:

$$
w_1 \leftarrow 0.9 - 0.01 \times 2 \times 0.918 \times 2 = 0.9 - 0.03672 = \boxed{0.86328}
$$

$$
v_1 \leftarrow 1.9 - 0.01 \times 2 \times 0.918 \times 1 = 1.9 - 0.01836 = \boxed{1.88164}
$$

**For Critic 2:**

Derivatives:

$$
\frac{\partial Q}{\partial w_2} = 2, \quad \frac{\partial Q}{\partial v_2} = 1
$$

$$
w_2 \leftarrow 0.7 - 0.01 \times 2 \times (-0.282) \times 2 = 0.7 + 0.01128 = \boxed{0.71128}
$$

$$
v_2 \leftarrow 1.1 - 0.01 \times 2 \times (-0.282) \times 1 = 1.1 + 0.00564 = \boxed{1.10564}
$$

> So the critics moved toward producing **2.782** at `(2, 1)`.

### 4.3 One-Step Actor Update (Simple Linear Actor)

Assume actor is also linear:

$$
\mu_{\theta}(s) = k \cdot s
$$

Current `k = 0.4`. So at `s = 2`, actor outputs `a = 0.8`.

TD3 updates the actor to increase `Q_φ₁(s, μ(s))`. Using Critic 1:

$$
Q_{\phi_1}(s, \mu(s)) = w_1 \cdot s + v_1 \cdot (k \cdot s) = s \cdot (w_1 + v_1 \cdot k)
$$

**Derivative w.r.t k:**

$$
\frac{\partial}{\partial k} Q_{\phi_1}(s, \mu(s)) = s \cdot v_1
$$

Using the updated `v₁ = 1.88164` and `s = 2`:

$$
\frac{\partial J}{\partial k} = 2 \times 1.88164 = 3.76328
$$

**Gradient ascent with step size β = 0.001:**

$$
k \leftarrow k + \beta \cdot 3.76328 = 0.4 + 0.00376328 = \boxed{0.40376328}
$$

> So the actor slightly changes to output slightly larger actions (in this toy setup) because that increases Critic 1's value estimate.

---

## 5. Why the Three TD3 Tricks Matter

| Trick | Purpose |
|-------|---------|
| **Twin critics + min** | Reduces the chance you learn from an overly optimistic value estimate |
| **Delayed actor updates** | Prevents actor from "chasing noise" while critics are still learning |
| **Target policy smoothing** | Prevents learning values that depend on extremely precise actions the real system can't reliably reproduce |

---

## Summary: TD3 vs DDPG

| Feature | DDPG | TD3 |
|---------|------|-----|
| Number of critics | 1 | 2 (twin) |
| Target value | Single Q estimate | Min of two Q estimates |
| Actor update frequency | Every step | Delayed (every 2 steps) |
| Target action | Exact | Smoothed with clipped noise |
