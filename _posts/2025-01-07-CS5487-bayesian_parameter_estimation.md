---
layout: post
title: "机器学习 (CS5487) - 贝叶斯参数估计"
subtitle: "Machine Learning"
date: 2025-01-07 13:35:00 +0800
categories: [AI]
---

# 贝叶斯参数估计 (Bayesian Parameter Estimation)

## 贝叶斯方法基础

贝叶斯方法与我们已知的根据统计的方法来推测概率的方式不同，贝叶斯理论是以“反方向”的推导来对未知的概率进行计算。

在贝叶斯方法中，存在多个概率分布。例如先验分布 $p(\theta)$，此分布表示我们已知的的目标参数的概率分布，或者说是我们已经知道的知识。同时，存在后验分布 $p(\theta \| D)$，表示存在观测值 $D$ 时的目标参数的概率是多少。显然，我们要计算的目标值就是此后验概率 $p(\theta \| D)$。

> 与之前的方法不同，贝叶斯方法的结果是一个概率分布，而不是一个确定的数值

根据条件概率公式，我们可以推导出：

$$
p(\theta | D) = \frac{p(D | \theta)p(\theta)}{p(D)}
$$

其中，$p(D \| \theta)$ 可以理解为已知参数 $\theta$ 时出现观测值 $D$ 的概率，也称为似然 (likelihood)

$p(D)$ 表示为出现观测值 $D$ 的概率，$p(D)$ 一般又可以表示为各种不同的前提条件下出现 $D$ 的概率的和，也就是 $\sum_k^K p(D \| \theta_k) p(\theta_k)$ (离散值)，$\int p(D \| \theta)p(\theta) \mathrm{d}\theta$ (连续值)

> 例如，在抛硬币例子中，$p(D)$ 就可以用 $p(D\|\theta_1)p(\theta_1) + p(D\|\theta_0)p(\theta_0)$ 来计算

### 示例：高斯分布

假设方差 $\sigma^2$ 是已知的，那么似然 $p(x_i \| \theta)$ 就是 $\mathcal{N}(x_i \| \mu, \sigma^2)$

并且假设先验 $p(\mu) = \mathcal{N}(\mu \| \mu_0, \sigma_0^2)$，也就是我们假设未知的参数 $\mu$ 也服从高斯分布

若假设所有的样本值都是独立同分布的，那么就有：

$$
p(\mu | D) = \frac{p(D | \mu)p(\mu)}{\int p(D|\mu)p(\mu) \mathrm{d}\mu}
$$

- $p(D \| \mu) = \prod_{i=1}^N p(x_i \| \mu)$

显然，$p(D)$ 是个常数，因此就有：

$$
p(\mu | D) \propto p(D | \mu)p(\mu)
$$

若对两边取对数：

$$
\begin{align*}
\log p(\mu | D) & \propto \log p(D | \mu) + \log p(\mu) \\
& = \sum_{i=1}^N \log p(x_i|\mu) + \log p(\mu) \\
& = \sum_{i=1}^N \log \mathcal{N} (\mu, \sigma^2) + \log \mathcal{N} (\mu_0, \sigma_0^2) \\
& = -\frac{1}{2 \hat\sigma_n^2}(\mu - \hat\mu_n)^2 + \text{const}
\end{align*}
$$

其中 $\hat\sigma_n^2 = \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}}$, $\hat\mu_n = \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\sigma^2}}(\frac{n}{\sigma^2}\hat\mu_{ML} + \frac{1}{\sigma_0^2}\mu)$
- $\hat\mu_{ML}$ 表示在这之后利用 Maximum Likelihood 的方法求出的值

根据此结果，我们可以得出以下结论：
- 当 $n \to 0$ 时， $\hat\mu_n \to \mu_0, \hat\sigma_n^2 \to \sigma_0^2$
- 当 $n \to \infty$ 时，$\hat\mu_n \to \hat\mu_{ML}, \hat\sigma_n^2 \to 0$

也就是说，当观测值的数量非常少时，我们更趋向于相信我们的先验知识；当观测值足够多时，可以有很高信心去相信我们预估值

## 预测分布 (Predictive Distribution)

> 预测分布是指，当我们手上有一组观测值 $D$ 时，出现新样本 $x_*$ 的概率分布

- 假设后验概率：$p(\mu \| D) = \mathcal{N}(\hat\mu_n, \hat\sigma_n^2)$
- 似然：$p(x \| \mu) = \mathcal{N}(\mu, \sigma^2)$

那么预测分布就可以写成：

$$
\begin{align*}
p(x_* | D) & = \int p(x_*|\mu)p(\mu|D) \mathrm{d}\mu \\
& = \int \mathcal{N}(\hat\mu_n, \hat\sigma_n^2) \mathcal{N}(\mu, \sigma^2) \mathrm{d}\mu \\
& = \int \mathcal{N}(\hat\mu_n, \sigma^2 + \hat\sigma_n^2)...\mathrm{d}\mu
\end{align*}
$$

> '...' 部分不会对最终结果产生影响

因此最终结论是：$p(x_*\|D) = \mathcal{N}(\hat\mu_n, \sigma^2 + \hat\sigma_n^2)$

## 最大后验概率 (Maximize A Posterior (MAP))

> 通常，直接计算 $\int p(D\|\theta)p(\theta) \mathrm{d}\theta$ 是比较困难的，因此我们可以利用其他方法去近似此计算的结果

> 解决方法之一就是去选择能使后验概率最大化的一组参数 $\theta$

也就是说：

$$
\begin{align*}
\hat\theta_{MAP} & = \mathop{\arg\max}_\theta p(\theta | D) \\
& = \mathop{\arg\max}_{\theta} p(D|\theta)p(\theta) \\
& = \mathop{\arg\max}_{\theta}\log p(D|\theta) + \log p(\theta)
\end{align*}
$$

此时我们可以得出：
- 后验分布 $p(\theta \| D) = \delta(\theta - \hat\theta_{MAP})$
    - 狄拉克函数 $\delta(x - \mu)$ 被定义为在除了 0 以外的所有点的值都为 0， 但是积分为 1，或者说概率分布中的所有质量都集中在一个点上，在 $x = \mu$ 处有无限窄也无限高的峰值的概率质量
- 预测分布 $p(x_* \| D) = p(x_* \| \hat\theta_{MAP})$

### 示例：高斯分布

$$
\begin{align*}
\hat\mu_{MAP} & = \mathop{\arg\max}_{\mu} p(\mu | D) \\
& = \mathop{\arg\max}_\mu \mathcal{N} (\mu | \hat\mu_n, \hat\sigma_n^2) \\
& = \hat\mu_n
\end{align*}
$$

## 贝叶斯回归 (Bayesian Regression)

该方法的 setup，并且概率分布与之前的示例相同：

$$
\begin{align*}
& x \in \mathbb{R}, \phi(x) \in \mathbb{R}^d \\
& f(x, \theta) = \phi(x)^T\theta, \theta \in \mathbb{R}^d \\
& y = f(x, \theta) + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2) \\
& \text{new condition} \rightarrow p(y|x, \theta) = \mathcal{N}(\theta|0, \alpha I)
\end{align*}
$$

- $\phi(x)$ 表示特征转换，$\epsilon$ 是服从高斯分布的自然噪声

利用最大后验概率：

$$
\begin{align*}
\hat\theta_{MAP} & = \mathop{\arg\max}_{\theta} \log p(D|\theta)p(\theta) \\
& = \mathop{\arg\max}_{\theta}-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i - \phi(x_i)^T\theta)^2 - \frac{1}{2\alpha}\left\|\theta\right\|^2
\end{align*}
$$

令 $\lambda = \frac{\sigma^2}{\alpha}$，并称之为超参数 (hyperparameter)：

$$
\hat\theta_{MAP} = \mathop{\arg\min}_{\theta}\left\|y - \Phi^T\theta\right\|^2 + \lambda \left\|\theta\right\|^2
$$

此时，该问题就变成了正则化最小二乘法问题 (Regularized Least Square, RLS)，或者也被称为岭回归 (Ridge Regression, RR) 等等

最后，对上式进行求解，就可以获得最优解：

$$
\hat\theta = (\Phi\Phi^T + \lambda I)^{-1}\Phi y
$$

- 此时 $\lambda I$ 的作用就是对协方差矩阵进行正则化，防止一些极端情况下的异常结果
