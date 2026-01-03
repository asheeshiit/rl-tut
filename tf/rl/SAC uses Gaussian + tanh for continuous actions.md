# SAC: Gaussian + Tanh for Continuous Actions

## 1. Why SAC Uses Gaussian + Tanh

In many control problems (robot torque, steering, throttle), actions are **continuous and bounded**, e.g., each action dimension must be in $[-1, 1]$ or $[-\text{max}, \text{max}]$.

A plain Gaussian $\mathcal{N}(\mu, \sigma)$ is **unbounded** (can produce any real number), so SAC does:

1. **Sample** an unbounded action $u \in \mathbb{R}^d$ from a Gaussian
2. **Squash** it with $\tanh$ to force it into $(-1, 1)$

This yields a **squashed Gaussian** (often called a *tanh-normal policy*).

---

## 2. The Continuous SAC Policy Parameterization

### Actor Network Outputs

Given state $s$, the actor outputs per action dimension:

- **Mean:** $\mu_\theta(s) \in \mathbb{R}^d$
- **Log std:** $\log \sigma_\theta(s) \in \mathbb{R}^d$

Convert std:

$$
\sigma_\theta(s) = \exp(\log \sigma_\theta(s))
$$

> In practice, $\log \sigma$ is often clipped for stability.

### Reparameterization Trick (Low-Variance Gradients)

1. **Sample noise:**

$$
\varepsilon \sim \mathcal{N}(0, I)
$$

2. **Construct the pre-squash action:**

$$
u = \mu_\theta(s) + \sigma_\theta(s) \odot \varepsilon
$$

3. **Squash:**

$$
a = \tanh(u)
$$

If the environment action bounds are $[\text{low}, \text{high}]$, commonly you map:

$$
a_{\text{env}} = a_{\text{scale}} \odot a + a_{\text{bias}}
$$

where (per dimension):

$$
a_{\text{scale}} = \frac{\text{high} - \text{low}}{2}, \quad a_{\text{bias}} = \frac{\text{high} + \text{low}}{2}
$$

---

## 3. The Key Question: How Do We Compute $\log \pi(a \mid s)$?

### Important: $a$ is NOT Gaussian

Because $a = \tanh(u)$, the distribution of $a$ is **not Normal** anymore.

So you **cannot** do $\log \mathcal{N}(a; \mu, \sigma)$.

**You must use the change-of-variables rule.**

---

## 4. Change-of-Variables Derivation (Core Math)

We start with a base density for $u$:

$$
u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)
$$

And transform:

$$
a = \tanh(u)
$$

### Change-of-Variables Formula

For an invertible transform $a = f(u)$:

$$
\pi(a \mid s) = p(u \mid s) \left| \det\left(\frac{du}{da}\right) \right|
$$

Equivalently (more commonly used):

$$
\log \pi(a \mid s) = \log p(u \mid s) - \log \left| \det\left(\frac{da}{du}\right) \right|
$$

### Jacobian for $\tanh$

Elementwise:

$$
\frac{da_i}{du_i} = 1 - \tanh(u_i)^2
$$

Since the transform is elementwise, the Jacobian is diagonal and:

$$
\log \left| \det\left(\frac{da}{du}\right) \right| = \sum_{i=1}^{d} \log(1 - \tanh(u_i)^2)
$$

But $\tanh(u_i) = a_i$, so:

$$
\log \left| \det\left(\frac{da}{du}\right) \right| = \sum_{i=1}^{d} \log(1 - a_i^2)
$$

### Final Log-Probability (Squashed Gaussian)

Let $\log p(u \mid s)$ be the diagonal Gaussian log-prob:

$$
\log p(u \mid s) = \sum_{i=1}^{d} \left( -\frac{1}{2}\left(\frac{u_i - \mu_i}{\sigma_i}\right)^2 - \log \sigma_i - \frac{1}{2}\log(2\pi) \right)
$$

Then:

$$
\log \pi(a \mid s) = \log p(u \mid s) - \sum_{i=1}^{d} \log(1 - \tanh(u_i)^2)
$$

Or equivalently:

$$
\log \pi(a \mid s) = \log p(u \mid s) - \sum_{i=1}^{d} \log(1 - a_i^2)
$$

---

## 5. Scaling to Environment Bounds

If:

$$
a_{\text{env}} = a_{\text{scale}} \odot a + a_{\text{bias}}
$$

then $\frac{da_{\text{env}}}{da} = a_{\text{scale}}$ (diagonal), so:

$$
\log \pi(a_{\text{env}} \mid s) = \log \pi(a \mid s) - \sum_{i=1}^{d} \log(a_{\text{scale},i})
$$

**In practice:**

$$
\log \pi(a_{\text{env}} \mid s) = \log p(u \mid s) - \sum_{i} \log(1 - \tanh(u_i)^2) - \sum_{i} \log(a_{\text{scale},i})
$$

---

## 6. Numerical Stability

### Problem

When $u$ is large, $a = \tanh(u) \approx \pm 1$, so $1 - a^2 \approx 0$ and $\log(1 - a^2) \to -\infty$.

### Solution A: Add a Tiny Epsilon

$$
\log(1 - a^2 + \epsilon)
$$

with $\epsilon \approx 10^{-6}$.

### Solution B: Use a Stable Identity (Common in SAC)

A numerically stable form:

$$
\log(1 - \tanh(u)^2) = 2(\log 2 - u - \text{softplus}(-2u))
$$

where:

$$
\text{softplus}(x) = \log(1 + e^x)
$$

This avoids catastrophic precision issues when $u$ is large.

---

## 7. Worked 1D Example (Full Numbers)

Let's do a single action dimension.

### Setup

Assume actor outputs:
- $\mu = 0.5$
- $\sigma = 0.2$

Sample $\varepsilon = 1.0$:

$$
u = \mu + \sigma \varepsilon = 0.5 + 0.2(1.0) = 0.7
$$

Squash:

$$
a = \tanh(0.7) \approx 0.6043678
$$

### Step 1: Gaussian Log-Prob of $u$

$$
\log p(u) = -\frac{1}{2}\left(\frac{u - \mu}{\sigma}\right)^2 - \log \sigma - \frac{1}{2}\log(2\pi)
$$

Compute:
- $(u - \mu) / \sigma = (0.7 - 0.5) / 0.2 = 1$
- First term: $-0.5$
- $-\log(0.2) = 1.6094379$
- $-\frac{1}{2}\log(2\pi) \approx -0.9189385$

So:

$$
\log p(u) \approx -0.5 + 1.6094379 - 0.9189385 = 0.1904994
$$

### Step 2: Jacobian Correction

$$
1 - \tanh(0.7)^2 \approx 1 - 0.6043678^2 \approx 0.6347396
$$

$$
\log(1 - \tanh(0.7)^2) \approx \log(0.6347396) \approx -0.4545405
$$

### Step 3: Final Squashed Log-Prob

$$
\log \pi(a \mid s) = \log p(u) - \log(1 - \tanh(u)^2)
$$

$$
\log \pi(a \mid s) \approx 0.1904994 - (-0.4545405) = 0.6450398
$$

So $\log \pi(a \mid s) \approx 0.6450$ for the squashed action $a \approx 0.6044$.

### With Environment Bounds $[-2, 2]$

Then $a_{\text{scale}} = 2$, $a_{\text{env}} = 2a$. Add:

$$
\log \pi(a_{\text{env}} \mid s) = \log \pi(a \mid s) - \log 2 \approx 0.6450398 - 0.6931472 = -0.0481074
$$

---

## 8. Where SAC Uses $\log \pi(a \mid s)$

You compute this log-prob for:

### Critic Target

$$
y = r + \gamma \left( \min(Q_1, Q_2)(s', a') - \alpha \log \pi(a' \mid s') \right)
$$

### Actor Loss

$$
L_\pi = \mathbb{E}\left[ \alpha \log \pi(a \mid s) - \min(Q_1, Q_2)(s, a) \right]
$$

---

> **Key Takeaway:** Computing $\log \pi$ correctly (with the tanh Jacobian correction) is essential for SAC to work properly.
