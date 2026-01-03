# TF-Agents Bandit Policies: Complete Research Guide

## Shared Bandit Setup (Notation)

At each round $t$:

1. **Observe** context $x_t$ (e.g., user features)
2. **Choose** action/arm $a_t \in \{1, \ldots, K\}$ (e.g., which recommendation to show)
3. **Observe** reward $r_t(a_t)$ (e.g., click/no-click, purchase value)

**Goal:** Maximize total reward / minimize regret (bandits don't model long-term transitions like full RL).

### Core Concepts

Most TF-Agents bandit policies rely on a **reward model** that predicts:

$$\hat{\mu}(x_t, a) \approx \mathbb{E}[r \mid x_t, a]$$

The policies differ mainly in **how they explore**.

**Important TF-Agents Features:**
- Some policies accept **per-arm features** via `accepts_per_arm_features`
- Some support **masking invalid actions** via `observation_and_action_constraint_splitter`

---

## 1) BernoulliThompsonSamplingPolicy

### TF-Agents API

```python
BernoulliThompsonSamplingPolicy(time_step_spec, action_spec, alpha, beta, ...)
```

### Layman Idea

Each arm has an "unknown click-rate." You keep a probability distribution for each arm's click-rate, sample one guess from each distribution, then pick the arm with the biggest sampled guess. Over time, the distributions tighten as you learn.

### Mathematical Formulation (Beta–Bernoulli Thompson Sampling)

**Assumption:** Rewards are binary $r \in \{0, 1\}$ (click/no click).

For arm $a$, model its success probability $\theta_a$ with a Beta prior:

$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$

**Action Selection (Thompson Sampling):**

1. Sample $\tilde{\theta}_a \sim \text{Beta}(\alpha_a, \beta_a)$ for each arm $a$
2. Pick $a_t = \arg\max_a \tilde{\theta}_a$

**Posterior Update** after observing reward $r_t \in \{0, 1\}$:

$$\alpha_{a_t} \leftarrow \alpha_{a_t} + r_t$$
$$\beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)$$

### Numeric Example

**Setup:** 3 arms, uniform prior $\alpha = \beta = 1$ for all.

| Round | Sampled Values | Action | Reward | Updated Parameters |
|-------|----------------|--------|--------|-------------------|
| 1 | $(0.62, 0.41, 0.55)$ | Arm 1 | $r=0$ | Arm 1: $\alpha_1=1, \beta_1=2$ |
| 2 | $(0.20, 0.60, 0.52)$ | Arm 2 | $r=1$ | Arm 2: $\alpha_2=2, \beta_2=1$ |

Over time, the best arm's Beta distribution becomes concentrated near its true click rate.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Strong practical baseline for binary outcomes (CTR, conversions) | Reward must be Bernoulli (or binarized) |
| Naturally balances explore/exploit without manual $\epsilon$ tuning | Mismatched if reward is continuous (revenue) |

**Best Use Cases:** Ads/CTR optimization, A/B/n testing with binary success

**Data Requirements:** Works with small/medium data; learns quickly but needs enough trials per arm

---

## 2) BoltzmannRewardPredictionPolicy

### TF-Agents API

```python
BoltzmannRewardPredictionPolicy(
    ..., 
    reward_network, 
    temperature=..., 
    boltzmann_gumbel_exploration_constant=...
)
```

### Layman Idea

Your model predicts a score for each action. Instead of always picking the top score, you convert scores into probabilities using softmax: higher predicted reward → higher chance, but others still get some chance.

### Mathematical Formulation (Softmax / Boltzmann Exploration)

Given predicted rewards $\hat{\mu}_a = \hat{\mu}(x_t, a)$ and temperature $T > 0$:

$$\pi(a \mid x_t) = \frac{\exp(\hat{\mu}_a / T)}{\sum_{j=1}^{K} \exp(\hat{\mu}_j / T)}$$

- **Small $T$** → near-greedy behavior
- **Large $T$** → near-uniform random

### Optional: Boltzmann–Gumbel Exploration (BGE)

TF-Agents exposes `boltzmann_gumbel_exploration_constant`. The BGE idea: pick the arm maximizing a perturbed estimate:

$$I_{t+1} = \arg\max_i \left\{ \hat{\mu}_{t,i} + \beta_{t,i} Z_{t,i} \right\}$$

where:
- $Z_{t,i}$ are i.i.d. standard Gumbel random variables
- $\beta_{t,i} = \frac{C}{\sqrt{N_{t,i}}}$ (common choice)

Uncertain arms (small $N_{t,i}$) get bigger noise → more exploration.

### Numeric Example (Plain Softmax)

**Predicted rewards:** $[0.20, 0.40, 0.35]$

| Temperature | Exp Values | Probabilities |
|-------------|------------|---------------|
| $T = 1$ | $[1.221, 1.492, 1.419]$ | $[0.295, 0.361, 0.344]$ |
| $T = 0.1$ | Uses $[2, 4, 3.5]$ | Far more peaked on action 2 |

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Simple, smooth exploration with any reward predictor | Requires temperature tuning/scheduling |
| Gumbel variant scales exploration with uncertainty | Poorly calibrated predictions → misleading probabilities |

**Best Use Cases:** Contextual bandits with neural reward model; graded exploration preferred over hard $\epsilon$-greedy

**Data Requirements:** Needs a reasonably trained `reward_network`

---

## 3) CategoricalPolicy

### TF-Agents API

A policy that chooses actions from a categorical distribution.

### Layman Idea

Generic wrapper: "Here are probabilities for actions. Sample one."

### Mathematical Formulation

Given probability vector $p \in \Delta_{K-1}$:

$$a_t \sim \text{Categorical}(p_1, \ldots, p_K)$$

### Numeric Example

If $p = [0.1, 0.7, 0.2]$, you sample action 2 most of the time.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Useful building block for turning logits into sampling policy | Not a learning algorithm—only samples from given distribution |

**Best Use Cases:** When another component already computed a categorical distribution you trust

---

## 4) FalconRewardPredictionPolicy

### TF-Agents API

```python
FalconRewardPredictionPolicy(
    ..., 
    reward_network, 
    exploitation_coefficient=..., 
    max_exploration_probability_hint=...
)
```

### Layman Idea

Pick the action your model thinks is best, but force a principled amount of exploration. The more "close" other actions are to the best, the more you explore them; if they look clearly worse, you explore them less.

Based on the **FALCON algorithm** (offline regression oracle reduction).

### Mathematical Formulation (FALCON Sampling Rule)

Let $f(x, a)$ be predicted reward at round $t$. Define:

$$\hat{a}(x_t) = \arg\max_{a \in A} f(x_t, a)$$

FALCON assigns probability to each non-best action $a \neq \hat{a}$:

$$p_t(a) = \frac{1}{K + \gamma(f(x_t, \hat{a}) - f(x_t, a))}$$

Best action gets the leftover:

$$p_t(\hat{a}) = 1 - \sum_{a \neq \hat{a}} p_t(a)$$

The `exploitation_coefficient` corresponds to $\gamma$ (how exploitative vs exploratory).

### Numeric Example

**Setup:** $K = 3$, predicted rewards: A=0.20, B=0.40 (best), C=0.35, $\gamma = 2$

| Action | Calculation | Probability |
|--------|-------------|-------------|
| A | $\frac{1}{3 + 2(0.40 - 0.20)} = \frac{1}{3.4}$ | 0.294 |
| C | $\frac{1}{3 + 2(0.40 - 0.35)} = \frac{1}{3.1}$ | 0.323 |
| B | $1 - (0.294 + 0.323)$ | 0.383 |

B is most likely, but A and C still get meaningful exploration.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Exploration adapts to estimated gaps (gap-based exploration) | Needs decent reward predictor |
| Theory-driven under realizability assumptions | Sensitive to reward model misspecification |

**Best Use Cases:** Contextual bandits where you trust supervised learning and want theory-driven exploration

**Data Requirements:** Can be unstable if feedback is sparse and predictor is weak early

---

## 5) GreedyMultiObjectiveNeuralPolicy

### TF-Agents API

Takes a `scalarizer` and `objective_networks`.

### Layman Idea

You care about multiple things at once (clicks, revenue, satisfaction). This policy predicts each objective, combines them into one score using a "business rule" (scalarizer), and greedily picks the best action.

### Mathematical Formulation

For $M$ objectives, predict a vector for each action $a$:

$$\hat{r}(x_t, a) = \left( \hat{r}^{(1)}(x_t, a), \ldots, \hat{r}^{(M)}(x_t, a) \right)$$

Apply scalarizer $S: \mathbb{R}^M \to \mathbb{R}$:

$$\hat{s}(x_t, a) = S\left( \hat{r}(x_t, a) \right)$$

Pick greedy action:

$$a_t = \arg\max_a \hat{s}(x_t, a)$$

**Common scalarizer (weighted sum):**

$$S(r) = \sum_{m=1}^{M} w_m r^{(m)}$$

### Numeric Example

**Two objectives:** clicks and revenue

| Action | Predicted (click, revenue) | Score with $S = 1.0 \cdot \text{click} + 0.1 \cdot \text{revenue}$ |
|--------|---------------------------|-------------------------------------------------------------------|
| A | $(0.30, 2.0)$ | $0.30 + 0.1 \times 2.0 = 0.50$ |
| B | $(0.25, 3.0)$ | $0.25 + 0.1 \times 3.0 = 0.55$ ✓ |

→ Pick B

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Practical for multi-metric tradeoffs | No built-in exploration → feedback loops |
| Simple decision rule | Bad scalarizer weights → bad behavior |

**Best Use Cases:** Business has trusted utility function; primarily need exploitation

**Data Requirements:** Needs labeled feedback for each objective; objectives must be scaled/normalized

---

## 6) GreedyRewardPredictionPolicy

### TF-Agents API

Greedy policy based on `reward_network`, optionally per-arm features.

### Layman Idea

Always pick what your model predicts will give the highest reward.

### Mathematical Formulation

Given $\hat{\mu}(x_t, a)$:

$$a_t = \arg\max_a \hat{\mu}(x_t, a)$$

### Numeric Example

Predictions $[0.20, 0.40, 0.35]$ → choose action 2 every time.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Best when predictor is already strong | No exploration → locks into suboptimal actions |
| Very stable (no randomness) | "Rich get richer" feedback loops |

**Best Use Cases:** Offline-trained model in low-risk environment, or as evaluation baseline

**Data Requirements:** Needs strong, unbiased reward model and/or lots of historical randomized data

---

## 7) LinearUCBPolicy

### TF-Agents API

```python
LinearUCBPolicy(cov_matrix, data_vector, num_samples, alpha, ...)
```

### Layman Idea

Assume reward is roughly linear in features. Pick actions with a good mix of:
1. High predicted reward
2. High uncertainty (so you learn)

### Mathematical Formulation (LinUCB)

For each arm $a$, assume:

$$\mathbb{E}[r \mid x, a] = x^\top \theta_a$$

**Maintain ridge-regression stats:**

$$A_a = \lambda I + \sum_{i: a_i = a} x_i x_i^\top$$
$$b_a = \sum_{i: a_i = a} r_i x_i$$

**Estimate parameters:**

$$\hat{\theta}_a = A_a^{-1} b_a$$

**Compute UCB score:**

$$\text{score}_a(x_t) = x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t}$$

**Pick:**

$$a_t = \arg\max_a \text{score}_a(x_t)$$

The parameter $\alpha$ controls exploration strength.

### Numeric Example (1D)

**Setup:** Feature $x_t = 1$, $\lambda = 1$, $\alpha = 1$

| Arm | $A$ | $b$ | $\hat{\theta}$ | Uncertainty $\sqrt{1/A}$ | Score |
|-----|-----|-----|----------------|--------------------------|-------|
| A | 6 | 0.6 | 0.1 | 0.408 | 0.508 |
| B | 3 | 0.5 | 0.167 | 0.577 | 0.744 |
| C | 2 | 0.3 | 0.15 | 0.707 | **0.857** ✓ |

Even though B's mean is higher than C's, C has higher uncertainty → LinUCB explores it.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Strong baseline when linearity is reasonable | Misspecification hurts performance |
| Very data-efficient in moderate dimensions | Needs good feature scaling |

**Best Use Cases:** Tabular/structured features, moderate dimension, fast online learning

**Data Requirements:** Needs enough samples per arm (or shared per-arm-features modeling) to stabilize $A_a^{-1}$

---

## 8) LinearBanditPolicy (Base Class)

### TF-Agents API

Base class from which LinUCB and Linear Thompson Sampling derive. Handles two main forms of feature input.

### Layman Idea

Not an algorithm itself—shared plumbing for linear bandit policies:
- Supports different feature formats (global context vs per-arm features)
- Manages linear algebra pieces

### Mathematical Interface

Formalizes the linear reward model:

$$\hat{\mu}(x, a) = \phi(x, a)^\top \theta$$

where $\phi(x, a)$ depends on whether you provide per-arm features.

**Note:** You don't "choose" this—you choose LinUCB or LinearTS which derive from it.

---

## 9) LinearThompsonSamplingPolicy

### TF-Agents API

```python
LinearThompsonSamplingPolicy(cov_matrix, data_vector, alpha, ...)
```

### Layman Idea

Like Bernoulli Thompson sampling, but for linear rewards: keep a distribution over linear parameters, sample a plausible parameter vector, and pick the action that looks best under that sample.

### Mathematical Formulation (Bayesian Linear Regression)

**Assumption:**

$$r = x^\top \theta_a + \epsilon$$

With ridge/Bayesian updates, the posterior has:
- **Mean:** $\hat{\theta}_a = A_a^{-1} b_a$
- **Covariance:** proportional to $A_a^{-1}$

**Thompson Sampling Step:**

1. Sample: $\tilde{\theta}_a \sim \mathcal{N}(\hat{\theta}_a, \alpha^2 A_a^{-1})$
2. Pick: $a_t = \arg\max_a x_t^\top \tilde{\theta}_a$

### Numeric Example (1D)

Using same stats as LinUCB:

| Arm | Mean | Variance ($\alpha^2/A$) | Sample (example draw) |
|-----|------|-------------------------|----------------------|
| A | 0.1 | 1/6 | $0.1 + 0.408 \times (-0.2) = 0.018$ |
| B | 0.167 | 1/3 | $0.167 + 0.577 \times 0.4 = 0.397$ ✓ |
| C | 0.15 | 1/2 | $0.15 + 0.707 \times 0.1 = 0.221$ |

→ Pick B. Next round, different random draws might pick C.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Often works extremely well in practice | Same linear-model limitations as LinUCB |
| Very natural exploration; less optimism bias than UCB | Randomness adds variance; needs $\alpha$ tuning |

**Best Use Cases:** Contextual bandits with structured features where linear model is plausible

**Data Requirements:** Needs enough signal for stable posteriors; sensitive to feature scaling

---

## 10) MixturePolicy

### TF-Agents API

```python
MixturePolicy(policies, mixture_distribution, ...)
```

### Layman Idea

A "meta-policy" that flips a weighted coin to pick which policy to use this step (e.g., 90% greedy, 10% uniform random).

### Mathematical Formulation

Let there be $M$ subpolicies $\{\pi_1, \ldots, \pi_M\}$.

1. Sample subpolicy index: $j \sim \text{Categorical}(w_1, \ldots, w_M)$
2. Sample action: $a_t \sim \pi_j(\cdot \mid x_t)$

**Overall mixed distribution:**

$$\pi_{\text{mix}}(a \mid x) = \sum_{j=1}^{M} w_j \pi_j(a \mid x)$$

### Numeric Example

- With prob 0.9: use Greedy policy
- With prob 0.1: use Uniform random

This implements a clean, explicit exploration schedule.

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Very flexible; easy to A/B exploration strategies | Too much randomness → unstable behavior |
| Great for staged rollouts and safety | Doesn't blend at fine-grained level |

**Best Use Cases:** Production systems needing simple control ("mostly exploit, sometimes explore"); ensemble robustness

**Data Requirements:** Depends on subpolicies; mixture itself doesn't learn

---

## 11) NeuralLinUCBPolicy

### TF-Agents API

```python
NeuralLinUCBPolicy(
    encoding_network, 
    encoding_dim, 
    reward_layer, 
    epsilon_greedy,
    cov_matrix, 
    data_vector, 
    ...
)
```

### Layman Idea

If the world is non-linear, first use a neural net to learn a good representation (embedding). Then run LinUCB in that learned embedding space.

### Mathematical Formulation

Let the neural encoder produce an embedding:

$$z = \phi_\psi(x_t, a) \in \mathbb{R}^d$$

Then assume reward is linear in $z$:

$$\mathbb{E}[r \mid x, a] \approx z^\top \theta$$

**Apply LinUCB on $z$ instead of raw $x$:**

$$\text{score}(x_t, a) = z^\top \hat{\theta} + \alpha \sqrt{z^\top A^{-1} z}$$

TF-Agents includes an initial $\epsilon$-greedy phase via `epsilon_greedy` parameter.

### Numeric Example

1. Encoder maps each (context, action) to 5-D embedding $z$
2. LinUCB computes $z^\top \hat{\theta} + \alpha\sqrt{z^\top A^{-1} z}$
3. Pick best action

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Handles non-linear structure better than pure linear | More moving parts: neural training + bandit exploration |
| Still has principled UCB exploration in embedding space | If encoder is poor, UCB confidence is meaningless |

**Best Use Cases:** High-dimensional contexts, non-linear reward patterns, but still want UCB-style exploration

**Data Requirements:** Typically needs more data than linear methods to learn good embeddings

---

## 12) RankingPolicy

### TF-Agents API

```python
RankingPolicy(
    num_items, 
    num_slots, 
    scoring_network,
    item_sampler,
    penalty_mixture_coefficient=...,
    logits_temperature=...,
    ...
)
```

### Layman Idea

Instead of choosing one action, you choose a **ranked list** (e.g., top 5 recommendations). You score items, then sample a diverse ranking so you don't show 5 near-duplicates.

### Mathematical Formulation

**Score each item:**

$$s_i = \text{network}(x_t, \text{item}_i)$$

**Ranking process:**
1. Convert scores into sampling distribution via `item_sampler`
2. **Temperature:** Divide logits by `logits_temperature` before sampling
3. **Diversity:** `penalty_mixture_coefficient` balances high-score selection vs diversity

**Note:** If number of available items varies, TF-Agents supports a `num_actions` field in observation.

### Numeric Example

**Setup:** Items 1..5, 2 slots. Scores: $[2.0, 1.9, 1.2, 0.5, 0.1]$

| Strategy | Selected Ranking |
|----------|-----------------|
| Pure greedy | [1, 2] |
| Diversity-aware | [1, 3] (if item 2 too similar to item 1) |

### Pros & Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Directly fits recommender ranking problems | Credit assignment harder (per-position, per-item feedback) |
| First-class diversity support via sampler/penalty | Computationally heavier than single-action policies |

**Best Use Cases:** Recommendations with multiple slots (feeds, carousels, top-N lists)

**Data Requirements:** Needs interaction data at ranking/list level; sparse click feedback is challenging

---

## 13) RewardPredictionBasePolicy (Base Class)

### TF-Agents API

Base class for reward-prediction-based policies. Takes `reward_network`, supports `accepts_per_arm_features`, `constraints`, and action masking splitter.

### Layman Idea

Not a single exploration algorithm—the common base for policies that:
1. Compute predicted rewards via `reward_network`
2. Apply some rule to choose/sample actions

Greedy, Boltzmann, and FALCON policies are all variants of this idea.

### Mathematical Interface

1. Compute vector of predicted rewards: $\hat{\mu}(x_t, 1..K)$
2. Apply a selection transform:
   - Greedy argmax
   - Softmax sampling
   - FALCON probability rule
   - Constraints/masking before sampling

**Note:** Building block for custom reward-prediction policies; doesn't solve exploration by itself.

---

## Quick Comparison Table

| Policy | Contextual? | Reward Type | Exploration Mechanism | Strong When... | Main Limitations |
|--------|-------------|-------------|----------------------|----------------|------------------|
| **BernoulliThompsonSampling** | Optional | Binary | Posterior sampling (Beta) | CTR/conversion with binary outcomes | Only Bernoulli rewards |
| **GreedyRewardPrediction** | Yes | Any scalar | None (argmax) | Model already excellent | No exploration → feedback loops |
| **BoltzmannRewardPrediction** | Yes | Any scalar | Softmax($\hat{\mu}/T$), optional Gumbel | Want smooth exploration with neural model | Temperature tuning; calibration-dependent |
| **FalconRewardPrediction** | Yes | Bounded/normalized | Gap-based: $1/(K + \gamma\Delta)$ | Principled gap-based exploration | Needs decent predictor |
| **LinearUCB** | Yes | Scalar | UCB bonus $\propto \sqrt{x^\top A^{-1} x}$ | Linear-ish rewards, fast online learning | Linear assumption; feature scaling |
| **LinearThompsonSampling** | Yes | Scalar | Bayesian sampling of linear params | Linear-ish rewards; often excellent | Linear assumption; tuning $\alpha$ |
| **NeuralLinUCB** | Yes | Scalar | UCB in learned embedding | Non-linear rewards with learned representation | Needs more data; encoder quality critical |
| **GreedyMultiObjectiveNeural** | Yes | Multi-objective → scalarized | None (greedy on scalarizer) | Clear business utility function | No exploration; objective scaling tricky |
| **RankingPolicy** | Yes | Listwise/per-item | Sampling ranked list with diversity | Multi-slot recommendations | Credit assignment; sparse feedback |
| **MixturePolicy** | Depends | Depends | Randomly pick subpolicy | Need explicit control (90% exploit/10% explore) | Doesn't learn itself |
| **CategoricalPolicy** | Any | Any | Sample from categorical | Already have action probabilities | Not a learning algorithm |
| **RewardPredictionBasePolicy** | Yes | Any scalar | Base class (varies) | Building custom policies | Not the algorithm itself |
| **LinearBanditPolicy** | Yes | Scalar | Base class for linear policies | Using LinUCB/LinearTS variants | Infrastructure, not standalone |

---

## Decision Flowchart

```
Is reward binary (0/1)?
├── Yes → BernoulliThompsonSamplingPolicy
└── No → Is reward linear in features?
    ├── Yes → LinearUCBPolicy or LinearThompsonSamplingPolicy
    └── No → Do you have a neural reward model?
        ├── Yes → Do you want principled exploration?
        │   ├── Yes → NeuralLinUCBPolicy
        │   └── No → BoltzmannRewardPredictionPolicy or GreedyRewardPredictionPolicy
        └── No → Do you need multi-objective optimization?
            ├── Yes → GreedyMultiObjectiveNeuralPolicy
            └── No → Do you need ranked lists?
                ├── Yes → RankingPolicy
                └── No → Consider MixturePolicy with exploration/exploitation blend
```
