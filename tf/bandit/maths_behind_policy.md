# Mathematical Foundations of Bandit Policies

A comprehensive guide to the mathematics behind various bandit and reinforcement learning policies.

---

## Common Notation

| Symbol | Description |
|--------|-------------|
| $a \in \{1, \ldots, K\}$ | Actions (arms) |
| $t$ | Time step |
| $r_t$ | Observed reward at time $t$ |
| $Q(a)$ | Expected reward of action $a$ |
| $x_a \in \mathbb{R}^d$ | Context/features for action $a$ |
| $\theta \in \mathbb{R}^d$ | Parameter vector |

---

## 1. BernoulliThompsonSamplingPolicy

### Mathematical Model

**Reward distribution:**

$$r \sim \text{Bernoulli}(\theta_a)$$

**Prior belief:**

$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$

### Posterior Update

After observing a reward:
- **Success** → $\alpha_a \leftarrow \alpha_a + 1$
- **Failure** → $\beta_a \leftarrow \beta_a + 1$

### Action Selection

1. Sample from posterior for each arm:

$$\tilde{\theta}_a \sim \text{Beta}(\alpha_a, \beta_a)$$

2. Choose the action with highest sample:

$$a^* = \arg\max_a \tilde{\theta}_a$$

### Example

Two ads with current parameters:

| Ad | α | β |
|----|---|---|
| A  | 5 | 3 |
| B  | 2 | 1 |

Sample from posteriors:
- $\tilde{\theta}_A = 0.58$
- $\tilde{\theta}_B = 0.72$

**Result:** Choose **B** (even though A has more data — this is exploration!)

---

## 2. BoltzmannRewardPredictionPolicy (Softmax)

### Mathematical Model

Given estimated rewards $Q(a)$ for each action, compute selection probabilities.

### Action Probability

$$P(a) = \frac{e^{Q(a)/\tau}}{\sum_i e^{Q(i)/\tau}}$$

Where $\tau > 0$ is the **temperature** parameter:
- High $\tau$ → More uniform (exploratory)
- Low $\tau$ → More greedy (exploitative)

### Example

Three actions with $Q = [10, 8, 5]$ and $\tau = 2$:

| Action | Q(a) | Calculation |
|--------|------|-------------|
| 1      | 10   | $e^5 = 148.4$ |
| 2      | 8    | $e^4 = 54.6$ |
| 3      | 5    | $e^{2.5} = 12.2$ |

**Probabilities:** $P = [0.69, 0.25, 0.06]$

---

## 3. CategoricalPolicy

### Mathematical Model

Policy is directly parameterized as a probability distribution:

$$\pi(a \mid s) = \text{Softmax}(z_a)$$

Where $z_a$ are **logits** output by a neural network.

### Sampling

$$a \sim \pi(a \mid s)$$

### Learning (Policy Gradient)

$$\nabla J(\theta) = \mathbb{E}\left[\nabla \log \pi(a \mid s) \cdot R\right]$$

### Example

Logits: $z = [1.5, 0.5, -0.5]$

**Result:** $\pi = [0.67, 0.24, 0.09]$

---

## 4. FalconRewardPredictionPolicy

### Mathematical Model

Pure prediction-based policy (no exploration):

$$\hat{r}_a = f(x_a)$$

### Action Selection

$$a^* = \arg\max_a \hat{r}_a$$

### Example

Predicted revenue for ads:

| Ad | Predicted Revenue |
|----|-------------------|
| A  | 2.1               |
| B  | 1.8               |
| C  | 2.5               |

**Result:** Pick **C** (no exploration term)

---

## 5. GreedyMultiObjectiveNeuralPolicy

### Mathematical Model

Combines multiple objectives with weights:

$$R(a) = \sum_{i=1}^{m} w_i \cdot R_i(a)$$

### Action Selection

$$a^* = \arg\max_a R(a)$$

### Example

**Weights:**
- Revenue: $w_1 = 0.7$
- CTR: $w_2 = 0.3$

**Ad scores:**

| Ad | Revenue | CTR |
|----|---------|-----|
| A  | 10      | 2   |
| B  | 7       | 4   |

**Calculations:**
- $R_A = 0.7(10) + 0.3(2) = 7.6$
- $R_B = 0.7(7) + 0.3(4) = 6.1$

**Result:** Pick **A**

---

## 6. GreedyRewardPredictionPolicy

### Mathematical Model

Same as Falcon — pure greedy selection:

$$a^* = \arg\max_a \hat{r}_a$$

### Example

$\hat{r} = [0.4, 0.6, 0.5]$

**Result:** $a = 2$

---

## 7. LinearUCBPolicy

### Reward Model

Linear relationship between context and reward:

$$r = x_a^\top \theta + \epsilon$$

### Parameter Estimate

$$\hat{\theta} = A^{-1} b$$

Where:
- $A = \sum x x^\top + \lambda I$ (regularized covariance)
- $b = \sum x r$ (reward-weighted features)

### Upper Confidence Bound

$$\text{UCB}(a) = x_a^\top \hat{\theta} + \alpha \sqrt{x_a^\top A^{-1} x_a}$$

- First term: **exploitation** (expected reward)
- Second term: **exploration** (uncertainty bonus)

### Example

- Mean prediction: 0.4
- Uncertainty: 0.3
- $\alpha = 1$

$$\text{UCB} = 0.4 + 0.3 = 0.7$$

---

## 8. LinearBanditPolicy

### Mathematical Model

$$\hat{r}_a = x_a^\top \hat{\theta}$$

### Action Selection

$$a^* = \arg\max_a \, x_a^\top \hat{\theta}$$

### Example

- $x = [1, 0]$
- $\hat{\theta} = [0.3, 0.2]$

**Result:** $\hat{r} = 0.3$

---

## 9. LinearThompsonSamplingPolicy

### Posterior Distribution

$$\theta \sim \mathcal{N}(\mu, \Sigma)$$

### Sampling

Draw a sample from the posterior:

$$\tilde{\theta} \sim \mathcal{N}(\mu, \Sigma)$$

### Action Selection

$$a^* = \arg\max_a \, x_a^\top \tilde{\theta}$$

### Example

Sampled: $\tilde{\theta} = [0.6, 0.1]$

**Result:** Pick action with best dot product $x_a^\top \tilde{\theta}$

---

## 10. MixturePolicy

### Mathematical Model

Combines multiple policies:

$$\pi(a) = \sum_{i=1}^{n} \lambda_i \cdot \pi_i(a)$$

Where $\sum_i \lambda_i = 1$

### Example

- 70% Greedy policy
- 30% Thompson Sampling policy

A random draw decides which policy to use for each decision.

---

## 11. NeuralLinUCBPolicy

### Mathematical Model

Neural network extracts features, then LinUCB is applied:

**Feature extraction:**

$$z_a = f_{\text{NN}}(x_a)$$

**UCB calculation:**

$$\text{UCB} = z_a^\top \hat{\theta} + \alpha \sqrt{z_a^\top A^{-1} z_a}$$

### Example

Neural network outputs learned features → LinUCB applied on top for principled exploration.

---

## 12. RankingPolicy

### Mathematical Model

**Score each item:**

$$s_i = f(x_i)$$

**Sort by scores:**

$$\text{rank} = \text{argsort}(s)$$

**Pairwise loss (example):**

$$\mathcal{L} = \log(1 + e^{-(s_i - s_j)})$$

This encourages correct ordering between pairs.

---

## 13. RewardPredictionBasePolicy

### Mathematical Model

Abstract base form:

$$a^* = \arg\max_a \, \mathbb{E}[r \mid x, a]$$

**Note:** No exploration unless subclassed with exploration mechanism.

---

## Summary Table

| Policy Type | Core Math | Exploration |
|-------------|-----------|-------------|
| **Thompson Sampling** | Bayesian posterior sampling | ✓ (via sampling) |
| **UCB** | Mean + uncertainty bound | ✓ (via confidence bound) |
| **Greedy** | Pure argmax | ✗ |
| **Softmax/Boltzmann** | Exponential weighting | ✓ (via temperature) |
| **Neural** | Function approximation | Depends on variant |
| **Mixture** | Policy combination | ✓ (via exploration policy) |
