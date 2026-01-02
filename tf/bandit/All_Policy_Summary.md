# 🎰 TF-Agents Bandit Policies: Complete Reference Guide

## 📖 Core Concepts

> **Exploitation**: Do what already works best  
> **Exploration**: Try uncertain options to learn more

Every policy below handles this tradeoff differently.
---

## 🔥 Quick Comparison Table

| Policy | Exploration | Context | Model Type | Best For | Key Strength | Key Weakness |
|--------|:-----------:|:-------:|------------|----------|--------------|--------------|
| **Bernoulli TS** | Probabilistic | ❌ | Bayesian | CTR optimization | Strong exploration | Binary only |
| **Boltzmann** | Soft | ❌ | Heuristic | Online experiments | Simple | Temp sensitive |
| **Categorical** | Learned | ✅ | Neural | RL agents | Flexible | Data hungry |
| **Falcon** | ❌ | ✅ | ML Model | Ads production | Stable | No learning |
| **Greedy MultiObj** | ❌ | ✅ | Neural | Business tradeoffs | Constraint-aware | Weight tuning |
| **Greedy RP** | ❌ | ❌ | Any | Baselines | Fast | Stagnates |
| **LinearUCB** | Confidence | ✅ | Linear | Contextual ads | Theory-backed | Linear assumption |
| **LinearBandit** | ❌ | ✅ | Linear | Simple systems | Interpretable | Weak exploration |
| **Linear TS** | Bayesian | ✅ | Linear | Contextual CTR | Excellent balance | Complex |
| **Mixture** | Configurable | Depends | Hybrid | Safe rollout | Robust | Hard to tune |
| **NeuralLinUCB** | Confidence | ✅ | Hybrid | Complex ads | Powerful | Heavy |
| **Ranking** | Depends | ✅ | ML | Search & Ads | UX aligned | Expensive |
| **RP Base** | ❌ | Depends | Any | Framework | Extensible | Not standalone |

---

## 🎯 Decision Flowchart

```
START
  │
  ├─ Binary rewards (click/no-click)?
  │    ├─ YES + No context → BernoulliThompsonSampling
  │    └─ YES + With context → LinearThompsonSampling
  │
  ├─ Need multiple objectives?
  │    └─ YES → GreedyMultiObjectiveNeural
  │
  ├─ Need ranking (not single action)?
  │    └─ YES → RankingPolicy
  │
  ├─ Have contextual features?
  │    ├─ Linear relationship → LinearUCB or LinearTS
  │    └─ Non-linear → NeuralLinUCB
  │
  ├─ Production system (no exploration needed)?
  │    └─ YES → Falcon or GreedyRewardPrediction
  │
  └─ Want safe experimentation?
       └─ YES → MixturePolicy
```
---

## 1. BernoulliThompsonSamplingPolicy

### 💡 Layman Idea
Each action is a coin (success/failure). You guess how biased each coin is, sample from that belief, and pick the best.

### 📌 When to Use
- Rewards are binary (click / no click)
- You want probabilistic exploration

### 🔢 Math
```
Reward ~ Bernoulli(θ)
Prior: θ ~ Beta(α, β)
Sample θ̂ from Beta, choose action with max θ̂
```

### 📝 Example
- Ad A: 7 clicks / 10 views
- Ad B: 2 clicks / 3 views
- Sampling may still pick B sometimes → exploration

| ✅ Pros | ❌ Cons |
|---------|---------|
| Excellent exploration | Binary rewards only |
| Simple | No context unless extended |
| Fast convergence | |

---

## 2. BoltzmannRewardPredictionPolicy (Softmax)

### 💡 Layman Idea
Give probability to each action proportional to how good it looks. Better actions → higher chance, but never 0.

### 📌 When to Use
- You have reward predictions
- You want smooth exploration

### 🔢 Math
```
P(a) = exp(Q(a)/τ) / Σ exp(Q(i)/τ)

τ = temperature
  High τ → more exploration
  Low τ  → greedy
```

### 📝 Example
Action scores: `[10, 8, 3]` → Still sometimes pick 8 or 3

| ✅ Pros | ❌ Cons |
|---------|---------|
| Simple | Sensitive to temperature |
| Tunable exploration | Can over-explore bad actions |

---

## 3. CategoricalPolicy

### 💡 Layman Idea
Policy directly outputs probability distribution: "Pick A with 60%, B with 30%, C with 10%"

### 📌 When to Use
- Policy-gradient methods
- Discrete action spaces

### 🔢 Math
```
π(a|s) = Categorical(p₁, p₂, ..., pₙ)
```

### 📝 Example
Neural network outputs logits → softmax → action

| ✅ Pros | ❌ Cons |
|---------|---------|
| Very flexible | Needs lots of data |
| Works with deep RL | Can be unstable |

---

## 4. FalconRewardPredictionPolicy

### 💡 Layman Idea
Production-grade reward prediction + greedy choice. Used in large-scale ad systems.

### 📌 When to Use
- High traffic
- Stable reward models

### 🔢 Math
```
a* = argmax R̂(a|x)
```

### 📝 Example
Pick ad with highest predicted revenue

| ✅ Pros | ❌ Cons |
|---------|---------|
| Scalable | No exploration |
| Deterministic | Can get stuck in local optima |
| Easy to debug | |

---

## 5. GreedyMultiObjectiveNeuralPolicy

### 💡 Layman Idea
Optimize multiple goals at once: Revenue, User experience, Fairness

### 📌 When to Use
- Trade-offs matter
- Business constraints exist

### 🔢 Math
```
R = w₁·R₁ + w₂·R₂ + ...
```

### 📝 Example
70% revenue + 30% CTR

| ✅ Pros | ❌ Cons |
|---------|---------|
| Handles business constraints | Hard to tune weights |
| Flexible objectives | Greedy (no exploration) |

---

## 6. GreedyRewardPredictionPolicy

### 💡 Layman Idea
Always pick best predicted reward

### 📌 When to Use
- You already trust your model
- Low risk tolerance

### 🔢 Math
```
a* = argmax R̂(a)
```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Simple | Zero exploration |
| Fast | Can degrade over time |

---

## 7. LinearUCBPolicy

### 💡 Layman Idea
"This option looks good AND I'm uncertain → try it"

### 📌 When to Use
- Contextual bandits
- Linear reward relationship

### 🔢 Math
```
a* = argmax [R̂(a) + α · uncertainty]
```

### 📝 Example
New ad with few samples but high uncertainty gets explored

| ✅ Pros | ❌ Cons |
|---------|---------|
| Principled exploration | Assumes linearity |
| Strong theoretical guarantees | Heavy matrix ops |

---

## 8. LinearBanditPolicy

### 💡 Layman Idea
Reward = linear function of features. Choose action with highest expected reward.

### 🔢 Math
```
R = θᵀx
```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Interpretable | No uncertainty handling |
| Efficient | Weak exploration |

---

## 9. LinearThompsonSamplingPolicy

### 💡 Layman Idea
Thompson Sampling with context

### 📌 When to Use
- Contextual bandits
- Binary or continuous rewards

### 🔢 Math
```
Sample θ from posterior → maximize θᵀx
```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Better than LinearUCB in practice | More complex |
| Natural exploration | Posterior approximation needed |

---

## 10. MixturePolicy

### 💡 Layman Idea
Combine multiple policies: "70% greedy + 30% exploratory"

### 📝 Example
- 50% Thompson Sampling
- 50% Greedy

| ✅ Pros | ❌ Cons |
|---------|---------|
| Robust | Hard to optimize |
| Easy experimentation | Debugging complexity |

---

## 11. NeuralLinUCBPolicy

### 💡 Layman Idea
Deep network learns features, Linear UCB on top for exploration

### 🔢 Math
```
Input → NN → feature vector → LinUCB → action
```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Handles non-linearity | Heavy computation |
| Principled exploration | Complex tuning |

---

## 12. RankingPolicy

### 💡 Layman Idea
Not one action → ordered list. Optimize entire ranking.

### 📌 When to Use
- Ads
- Recommendations
- Search results

### 🔢 Math
```
Listwise / pairwise loss (NDCG, MRR)
```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Matches real UX | Complex training |
| Powerful | Expensive inference |

---

## 13. RewardPredictionBasePolicy

### 💡 Layman Idea
Base class: predict reward → choose. No exploration logic by default.

| ✅ Pros | ❌ Cons |
|---------|---------|
| Reusable | Needs extension |
| Clean architecture | Not a learning policy itself |

