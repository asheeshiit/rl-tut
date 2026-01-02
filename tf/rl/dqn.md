# Deep Q-Network (DQN) — Reinforcement Learning Explained

---

## 1. DQN in Layman Terms

Imagine teaching a dog to play a video game:

- The dog **sees the screen**
- It **presses a button**
- It either:
  - Gets a **reward** (good)
  - Or **nothing / penalty** (bad)

Over time, the dog learns:
> "In this situation, pressing this button usually works best."

**DQN is exactly this:**
- Learn which action is best in each situation
- By trial and error
- Using a neural network as memory

---

## 2. Core Idea in One Sentence

> DQN learns a function that tells you how good each action is in a given state, and always picks the best one.

---

## 3. What Problem DQN Solves

Classic Q-learning uses a table:

| State | Action | Value |
|-------|--------|-------|
| s₁    | a₁     | 5.2   |
| s₁    | a₂     | 3.1   |
| ...   | ...    | ...   |

**But:**
- Real states are huge (images, sensors)
- Table becomes impossible to store

➡️ **DQN replaces the table with a neural network**

---

## 4. Reinforcement Learning Setup

### Environment Components

| Symbol | Meaning     |
|--------|-------------|
| $s$    | State       |
| $a$    | Action      |
| $r$    | Reward      |
| $s'$   | Next state  |

### Goal

**Maximize total future reward.**

---

## 5. What is a Q-value?

### Definition

$$Q(s, a) = \text{expected total future reward if we take action } a \text{ in state } s$$

This includes:
- Immediate reward
- All future rewards

---

## 6. Bellman Equation (Core Math)

This is the foundation of DQN:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

**Where:**
| Symbol | Meaning |
|--------|---------|
| $r$ | Immediate reward |
| $\gamma \in [0, 1]$ | Discount factor |
| $s'$ | Next state |
| $a'$ | Next action |

**Meaning:**
> "Value now = reward now + best possible future value"

---

## 7. What DQN Approximates

Instead of true $Q^*(s, a)$, we learn:

$$Q_\theta(s, a)$$

Where $\theta$ = neural network parameters

---

## 8. DQN Algorithm (High-Level Steps)

1. **Observe** current state $s$
2. **Choose** action $a$ (explore or exploit)
3. **Execute** action → get $r, s'$
4. **Store** transition $(s, a, r, s')$
5. **Train** neural network to satisfy Bellman equation
6. **Repeat**

---

## 9. Action Selection — ε-Greedy

To balance learning vs trying new things:

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s, a) & \text{with probability } 1 - \epsilon \end{cases}$$

- High $\epsilon$ → more **exploration**
- Low $\epsilon$ → more **exploitation**

---

## 10. Experience Replay (Important)

Instead of learning from the latest step only:

1. **Store** experiences in memory
2. **Sample** random mini-batches

**Why?**
- Breaks correlation between consecutive samples
- Stabilizes learning

**Replay buffer:**

$$D = \{(s, a, r, s')\}$$

---

## 11. Target Network (Important)

Using the same network for:
- Predicting values
- Computing targets

...causes **instability**.

**Solution:**
- Maintain a separate **target network** $Q_{\theta^-}$
- Updated slowly from $\theta$

---

## 12. DQN Loss Function (Math)

### Target Value

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

### Prediction

$$\hat{y} = Q_\theta(s, a)$$

### Loss (Mean Squared Error)

$$L(\theta) = \mathbb{E}\left[ (y - Q_\theta(s, a))^2 \right]$$

---

## 13. Gradient Update

$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

This pushes Q-values toward **Bellman consistency**.

---

## 14. Full DQN Algorithm (Formal)

```
Initialize Q-network Q_θ
Initialize target network Q_θ⁻ = Q_θ
Initialize replay buffer D

for each episode:
    observe initial state s

    for each step:
        choose action a using ε-greedy
        execute a → observe r, s'
        store (s, a, r, s') in D

        sample minibatch from D
        compute targets:
            y = r + γ max_a' Q_θ⁻(s', a')

        minimize (y − Q_θ(s, a))²

        periodically update θ⁻ ← θ
```

---

## 15. Worked Example (Simple Numbers)

### Environment

- **State:** position on a line
- **Actions:** Left (L), Right (R)
- **Goal:** reach position 3
- **Reward:**
  - +10 at goal
  - −1 otherwise

### Step Example

**Current state:** $s = 2$

**Network predicts:**

| Action | Q-value |
|--------|---------|
| L      | 3       |
| R      | 6       |

**Choose action:** $a = R$ (since $Q(2, R) > Q(2, L)$)

### Observe Outcome

- **Reward:** $r = +10$
- **Next state:** $s' = 3$ (terminal)

### Compute Target

$$y = 10 + \gamma \cdot 0 = 10$$

(Terminal state has 0 future value)

### Compute Loss

$$(10 - 6)^2 = 16$$

**Gradient update increases** $Q(2, R)$ toward 10.

---

## 16. Example with Future Rewards

**Given:**
- $r = 1$
- $\gamma = 0.9$
- $\max Q(s') = 5$

**Target:**

$$y = 1 + 0.9 \times 5 = 5.5$$

**Prediction:**

$$Q(s, a) = 4$$

**Loss:**

$$(5.5 - 4)^2 = 2.25$$

---

## 17. Why DQN Works

| Component | Purpose |
|-----------|---------|
| Neural network | Generalizes across states |
| Replay buffer | Stabilizes learning |
| Target network | Prevents oscillations |
| Bellman equation | Enforces optimality |

---

## 18. Limitations of DQN

- ❌ **Discrete actions only** (can't handle continuous action spaces)
- ❌ **Overestimation bias** (tends to overestimate Q-values)
- ❌ **Sample inefficient** (needs many interactions)

### Extensions that address these:
- **Double DQN** — reduces overestimation
- **Dueling DQN** — separates state value and advantage
- **Rainbow** — combines multiple improvements

---

## 19. Intuition Summary

> DQN learns by repeatedly guessing how good an action is, checking reality, and correcting itself so that present value equals reward plus best future value.

---

## 20. One-Line Takeaway

> **DQN is Q-learning with a neural network trained to satisfy the Bellman equation using replay memory and target networks.**
