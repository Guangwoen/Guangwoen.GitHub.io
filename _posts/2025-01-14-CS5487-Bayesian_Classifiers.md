---
layout: post
title: "机器学习 (CS5487) - 贝叶斯分类器"
subtitle: "Machine Learning"
date: 2025-01-14 16:10:00 +0800
categories: [AI]
---

# 贝叶斯分类器 (Bayesian Classifiers)

## 贝叶斯决策论 (Bayesian Decision Theory)

> 对于一个分类决策问题，贝叶斯决策论是用来做出一些有选择性的决策的框架，是用统计学的方法进行的模式分类

### 方法框架

1. World 具有状态 (States)，或者类（Classes），生成于随机变量 $Y$
    - 先验 $p(Y)$ 是该状态发生的概率
2. Observer 从随机变量 $X$ 测量观测值（Observations）或者特征（Features）
    - 类条件概率密度（Class Conditional Density (CCD)） 
    - $p(X|Y)$ 是特定于一个类或者状态的特征
3. Decision Function：用特征来推断状态
    - $g(X): X \to Y$
4. Loss Function: 选择错误的 $Y$ 时会产生的惩罚（错误的决策）

我们的目标就是在上述条件下（损失函数，类条件概率密度，先验）去寻找一个最优决策函数 $g^*(x)$

## 贝叶斯决策法则 (Bayesian Decision Rule)

风险 Risk 就是损失函数的期望值，可以写成：

$$
\begin{align*}
\mathbb{E}_{X, Y} [L(g(X), Y)] & = \sum_{y \in Y} \int_x p(x, y)L(g(x), y)\mathrm{d}x \\
& = \int_x \sum_y p(y|x)p(x)L(g(x), y) \mathrm{d}x \\
& = \int_x p(x) [\sum_y p(y|x)L(g(x), y)] \mathrm{d}x \\
& = \mathbb{E}_x[R(x)]
\end{align*}
$$

- 其中，在已知 $x$ 的情况下，$R(x) = \sum_y p(y|x)L(g(x), y)$ 被称为条件风险 (conditional risk)

> 因为损失函数永远大于等于 0，因此最小化风险函数等同于最小化 $x$ 条件的条件风险 $R(x)$

也就是说，当已知 $x$ 时：

$$
\begin{align*}
g^*(x) = y^* & = \mathop{\arg\min}_{j \in Y} R(x) \\
& = \mathop{\arg\min}_{j \in Y} \sum_y p(y|x)L(j, y) \\
& = \mathop{\arg\min}_{j \in Y} \mathbb{E}_{y|x}[L(j, y)] \rightarrow \text{Bayesian Decision Rule}
\end{align*}
$$

### 0-1 损失函数和分类

对于 $Y \in \{1, ..., c\}$，定义0-1 损失函数：

$$
L(g(x), y) = \begin{cases}
1 & \text{if } g(x) \neq y \\
0 & \text{otherwise}
\end{cases}
$$

此时，条件损失就变成了：

$$
\begin{align*}
R(x) & = \mathbb{E}_{y|x}[L(g(x), y)] \\
& = p(g(x) \neq y |x) \\
& = 1 - p(g(x) = y|x)
\end{align*}
$$

此时 BDR 就是：

$$
\begin{align*}
g^*(x) = y^* & = \mathop{\arg\min}_{j \in Y} (1 - p(y=j|x)) \\
& = \mathop{\arg\max}_{j \in y} p(y=j|x)
\end{align*}
$$

此时，根据 MAP 法则，我们应当选择能最大化后验的 j，那么上式就等同于：

$$
\begin{align*}
g^*(x) & = \mathop{\arg\max}_j \frac{p(x|y=j)p(y=j)}{p(x)} \\
& = \mathop{\arg\max}_j p(x|y=j)p(y=j) \\
& = \mathop{\arg\max}_j \log p(x|y=j) + \log p(y=j)
\end{align*}
$$

#### 示例：2 分类问题

根据我们目前的结论，在仅有 2 个分类的问题中，我们应当选择具有更大后验的那一类，也就是当 $p(x|0)p(0) > p(x|1)p(1)$ 时选择 0 ，其他情况选择 1

此时：

$$
p(x|0)p(0) > p(x|1)p(1) \Rightarrow \frac{p(x|0)}{p(x|1)} > \frac{p(1)}{p(0)}
$$

- $\frac{p(x|0)}{p(x|1)}$ 被称为似然比率 (likelihood ratio)
- $\frac{p(1)}{p(0)}$ 被称为阈值 (threshold)

对于 0-1 分类问题的总结：

- BDR 服从 MAP 规则
- 风险 Risk 是错误 error 的概率分布
- BDR 会最小化风险

## 示例：含有噪声的 Channel

> 我们需要在含有噪声的 channel 中传输一些 bit

假设我们的决策阈值是 T：

$$
Y = \begin{cases}
0 & x < T \\
1 & x \geq T
\end{cases}
$$

- 求出最佳的 T

求解：

假设比特的传输概率是 $p(Y=1) = p(Y=0) = \frac{1}{2}$，并且假设噪声是高斯噪声：$\epsilon \sim \mathcal{N}(0, \sigma^2)$，那么传输完成之后的比特就是 $x = \mu y + \epsilon$

$$
\begin{cases}
p(x|0) = \mathcal{N}(x|\mu_0, \sigma^2) \\
p(x|1) = \mathcal{N}(x|\mu_1, \sigma^2)
\end{cases}
$$

根据 基于 0-1 loss 的 BDR：

$$
\begin{align*}
y^* & = \mathop{\arg\max}_j \log p(x|j) + \log p(j) \\
& = \mathop{\arg\max}_j -\frac{1}{2 \sigma^2}(x-\mu_j)^2 - \frac{1}{2}\log 2\pi - \frac{1}{2}\log \sigma^2 + \log \frac{1}{2} \\
& = \mathop{\arg\min}_j (x-\mu_j)^2 \\
& = \mathop{\arg\min}_j \mu_j^2 - 2x\mu_j
\end{align*}
$$

因此，当 $\mu_0^2 - 2x\mu_0 < \mu_1^2 - 2x\mu_1$ 时选择 0， 也就是当 $x < \frac{\mu_1 + \mu_0}{2}$ 时

我们在这里的假设是先验分布是均匀的，也就是 $p(Y=0) = p(Y=1) = \frac{1}{2}$；当此概率不是均匀分布时：

当 $x < \frac{\mu_0 + \mu_1}{2} + \frac{\sigma^2}{\mu_1 - \mu_0} \log \frac{p(y=0)}{p(y=1)}$ 时选择 0

- $\frac{\sigma^2}{\mu_1 - \mu_0}$ 项正则化了两个 mean 之间的距离，根据此项，我们还可以得出以下结论
    - 当 mean 之间的距离很大时，无视先验
    - 当 mean 之间的距离很小时，使用先验

## 高斯分类器 (Gaussian Classifier)

假设我们有：$Y \in \{1, ..., c\}, p(y=j) = \pi_j, x \in \mathbb{R}^d$，并且 CCD 是高斯分布，也就是：$p(x|y=j) = \mathcal{N}(x|\mu_j, \Sigma_j)$

基于 0-1 损失函数的 BDR：

$$
\begin{align*}
g^*(x) & = \mathop{\arg\max}_j \log p(x|y=j) + \log p(y=j) \\
& = \mathop{\arg\max}_j -\frac{1}{2} \left\|x-\mu_j\right\|^2 - \frac{1}{2} \log |\Sigma_j| + \log \pi_j
\end{align*}
$$

特别地，当 $\Sigma_j = \sigma^2 I$ 时：

$$
\begin{align*}
g_j(x) & = -\frac{1}{2\sigma^2} \left\|x-\mu_j\right\|^2 - \frac{1}{2} \log |\sigma^2 I| + \log \pi_j \\
& = -\frac{1}{2\sigma^2} (-2x^T\mu_j + \mu_j^T \mu_j) + \log \pi_j + \text{const}
\end{align*}
$$

若将此式写成 $g_j(x) = w_j^T x + b_j$ （线性辨别函数）的形式，那么就有：$w_j = \frac{1}{\sigma^2}\mu_j$, $b_j = -\frac{1}{2\sigma^2}\mu_j^T \mu_j + \log \pi_j$
