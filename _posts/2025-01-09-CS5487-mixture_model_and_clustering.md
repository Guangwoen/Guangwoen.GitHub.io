---
layout: post
title: "机器学习 (CS5487) - 混合模型与聚类"
subtitle: "Machine Learning"
date: 2025-01-09 13:32:00 +0800
categories: [AI]
---

# 混合模型与聚类 (Mixture Models and Clustering)

> 在前两章的方法中，我们都是先假设观测数据 $D$ 服从一个概率分布模型，并根据最大化后验或者最大化似然的方式去计算出最优的一组参数 $\theta$

> 然而，很多情况下单一模型并不能很好的去拟合真实数据，因此我们需要组合多个模型来解决这个问题

## 高斯混合模型 (Gaussian Mixture Model (GMM))

GMM 模型一般由多个高斯模型的加权和作为其密度函数，也就是：

$$
p(x) = \sum_{j=1}^k p(x|z=j) p(z=j) = \sum_{i=1}^k \pi_j \mathcal{N}(x | \mu_j, \sigma_j^2)
$$

- $z$ 是表示类型的随机变量，也被称为隐变量 (Hidden Variable)
- $k$ 是单高斯模型的个数，也就是我们的混合模型是由 $k$ 个高斯模型组成的
- $\pi_j$ 表示第 $j$ 个模型的权重，同理 $\mu_j$ 和 $\sigma_j^2$ 表示第 $j$ 个高斯模型的均值和方差

理论上，GMM 可以拟合出任意类型的分布，并且我们并不能直接观察到类型 $z$，只能观察到 $x$

> 注意，GMM 模型并不是高斯随机变量的加权和 (Gaussian Random Variables)，而是高斯概率密度函数的加权和 (Gaussian PDF)

## GMM 聚类

> GMM 是一个生成模型 (Generative Model)

假设有样本 $D = \{x_1, x_2, ..., x_N\}$ 和 $k$ 个分类

从样本 $D$ 生成 GMM：

1. 第 $j$ 个高斯分量 (Gaussian Component)：位置 $\mu_j$, 宽度 $\sigma_j^2$
2. 第 $j$ 个分量的权重：$\pi_j$
3. 每个样本 $x_i$ 在每个分量中的安排 $z_i$

### 详细步骤

令 $z_i \in \{1, 2, ..., k\}$，表示 $x_i$ 被安排在哪一个分量中

- $z_i$ 将会是一个参数，我们的目标就是去优化这个参数
- 或者说，我们的目标是去最大化联合似然 (Joint Likelihood)

#### 联合似然 (Joint Likelihood) 和数据似然 (Data Likelihood) 之间的区别

联合似然是关于所有观测数据和模型参数的联合分布的似然函数，表示数据和参数同时发生的概率，通常写为：

$$
L_{joint}(\theta; X) = p(X, \theta)
$$

联合似然通常又可以被分解为先验概率和数据似然的乘积：

$$
L_{joint}(\theta; X) = p(X|\theta)p(\theta)
$$

- $p(X\|\theta)$ 是条件分布，称为数据似然
- $p(\theta)$ 是参数的先验分布

因此，在 GMM 中最大化联合似然的目标就可以写为：

$$
\begin{align*}
\hat\theta & = \mathop{\arg\max}_{\theta, z}\sum_{i=1}^N \log p(x_i, z_i) \\
& = \mathop{\arg\max}_{\theta, z} \sum_{i=1}^N \log p(x_i|z_i) + \log p(z_i) \\
& = \mathop{\arg\max}_{\theta, z} \sum_{i=1}^N \log \mathcal{N} (x_i|\mu_{z_i}, \sigma_{z_i}^2) + \log \pi_{z_i}
\end{align*}
$$

此时，为了方便，我们可以对 $z_i$ 使用 indicator variable 的方法，也就是用 $z_{i, j} = 1$ 来表示 $z_i = j$，其他情况则是 $z_{i, j} = 0$

此时上式中的先验分布和数据似然就会变成：

- $\log p(z_i) = \sum_{j=1}^k z_{i, j} \log \pi_j$
- $\log p(x_i\|z_i) = \sum_{j=1}^k z_{i, j}\log \mathcal{N}(x_i\|\mu_j, \sigma_j^2)$

将此变化代入到我们的目标函数中：

$$
\begin{equation}
\hat\theta = \mathop{\arg\max}_{\theta, z} \sum_{i=1}^N \sum_{j=1}^k z_{i, j}[\log \pi_j + \log \mathcal{N}(x_i|\mu_j, \sigma_j^2)]
\end{equation}
$$

## Expectation Maximization - EM 算法

### 交替极大化算法 (Alternating Maximization)

显然，因为有两个随机变量，而且这两个随机变量又是相互依赖的，因此我们不能一次性地来算出最优解析解，而是需要固定其中一个变量，算出另一个变量的最优解，然后固定此最优解，算出之前 step 中固定的变量的最优解，交替进行。

具体来说，对于 GMM 模型：

1. 固定参数 $\theta = \{\mu_j, \sigma_j^2, \pi_j\}$，找出最优 $z_i$

此时的目标就是：$\mathop{\arg\max}_{z_{i, j}} \sum_{j=1}^k z_{i, j}[\log \pi_j + \log \mathcal{N}(x_i\|\mu_j, \sigma_j^2)]$，但是我们可以发现，对于每个样本，只能被分配到一个分量中，所以在此目标的 $k$ 个分量中只有一个值是有效的，也就是：

$$
z_i = \mathop{\arg\max}_{j} \log \pi_j + \log \mathcal{N}(x_i|\mu_j, \sigma_j^2)
$$

2. 固定 $z_i$，找出最优 $\theta = \{\mu_j, \sigma_j^2, \pi_j\}$

此时，我们就可以对目标函数(1)的每一个参数，求出最优解：
- $\hat\mu_j = \frac{1}{\sum_i^N z_{i, j}} \sum_i^N z_{i, j}x_i$
- $\hat\sigma_j^2 = \sum_i^N z_{i, j} (x_i - \hat\mu_j)^2$
- $\hat\pi_j = \frac{1}{N}\sum_i^N z_{i, j}$

3. 迭代进行步骤 1, 2，直到收敛

若此时 $\hat\sigma^2_j = constant$，并且 $\hat\pi_j = \frac{1}{k}$，那么就会变成 KMeans 算法
- $\hat z_j = \mathop{\arg\min}_j (x_i - \mu_j)^2$
- $\hat\mu_j = \frac{1}{\sum z} \sum zx$

> 显然，对于这种算法，参数的初始值可能对结果产生非常大的影响

### EM 算法

EM算法是一种从存在隐变量的数据集中求解概率模型参数的最大似然估计的方法

> 我们在概率的角度来处理隐变量 $z$

大致步骤可以写成：

1. 对于隐变量 $z$ 求解期望值 $\hat z$
2. 最大化 $\log p(x, \hat z)$ 并求解最优模型
3. 重复

#### EM 算法步骤

选择参数初始值 $\hat \theta^{(old)}$

1. E-Step

$$
Q(\theta; \hat \theta^{(old)}) = \mathbb{E}_{z|x, \hat\theta^{(old)}}[\log p(x, z|\theta)]
$$

2. M-Step

$$
\hat \theta^{(new)} = \mathop{\arg\max}_{\theta} Q(\theta; \theta^{(old)})
$$

3. $\hat\theta^{(old)} \gets \hat\theta^{(new)}$, 并重复 1, 2 直到收敛

#### EM for GMM

1. E-Step

$$
\begin{align*}
Q(\theta; \hat\theta^{(old)}) & = \mathbb{E}_{z|x, \hat\theta^{(old)}}[\log p(x, z|\theta)] \\
& = \sum_i^N \sum_j^k \hat z_{i, j} [\log \pi_j + \log \mathcal{N}(x_i|\mu_j, \sigma_j^2)]
\end{align*}
$$

其中

$$
\begin{align*}
\hat z_{i, j} & = \mathbb{E}_{z|x, \hat\theta^{(old)}} [z_{i, j}] = p(z_i = j|X, \hat\theta^{(old)}) \\
& = \frac{p(X|z_i = j, \hat\theta^{(old)})p(z_i = j)}{p(X)} \\
& = \frac{p(X_{\not i}) p(x_i|z_j, \hat\theta^{(old)})p(z_i=j)}{p(X_{\not i})p(x_i)} \\
& = \frac{\pi_j p(x_i|z_i=j, \hat\theta^{(old)})}{\sum_k^K \pi_k p(x_i|z_i = k)} \\
& = p(z_i = j|x_i, \hat\theta^{(old)})
\end{align*}
$$

2. M-Step

M-Step 与交替极大化算法的第2步相同，但是其中的 $z_{i, j}$ 将会被替换成 $\hat z_{i, j}$

****

总结：
- EM 是求解含有隐变量的 MLE 模型的通用算法
- EM 算法的每次迭代将会增大数据似然，最终会收敛到局部最大值
- 对参数 $\theta$ 的不同的初始化将会导致不同的结果，我们应该选出能使数据似然最大的初始化值
- EM 算法的解释
    - E-Step 构建了函数的下限
    - M-Step 最大化了函数的下限
