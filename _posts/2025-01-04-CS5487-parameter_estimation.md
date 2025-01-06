---
layout: post
title: "机器学习 (CS5487) - 参数估计"
subtitle: "Machine Learning"
date: 2025-01-04 13:23:00 +0800
categories: [AI]
---

> 机器学习 CS5487 系列是本人在 CityUHK 的 CS5487 课程的笔记，将会涉及到机器学习各种算法的数学推导以及最本质的问题。

# 参数估计 (Parameter Estimation)

> 参数估计要解决的问题是我们应该如何去拟合一个随机变量 $X$ 的概率分布 (probability distribution)

参数估计的过程一般会涉及到三个步骤：

1. 选择一个概率分布，例如高斯分布 (Gaussian Distribution)，然后令分布的参数为 $\theta$
2. 从 $X$ 中选择一组样本（观测值）: $D = \{x_1, x_2, ..., x_N\}$，此时我们假设 $x_i$ 是独立同分布的 (i.i.d)
3. 最后利用最大似然原则 (Maximum Likelihood Principle)：“最佳的一组参数 $\theta^*$ 是最大化似然 (maximizes likelihood) 的或者说是能最大化训练集的概率的一组参数 (maximizes the probability of the training data)”

## 最大似然估计 (Maximum Likelihood Estimation)

> 哪一组参数能使数据集 $D$ 出现的概率最大？

利用数学式来表示 MLE 的过程：

$$
\begin{align*}
\theta^* & = \mathop{\arg\max}_\theta p(D|\theta) \\
& = \mathop{\arg\max}_\theta \log p(D|\theta) \\
& = \mathop{\arg\min}_\theta -\log p(D|\theta)
\end{align*}
$$

- 为了方便计算，一般会使用 log 来消除一些指数操作并简化一些乘法或者除法计算，此时取 log 的似然函数将称为对数似然函数 (Log-Likelihood Function)
- 如果取负号，此时被称为负对数似然函数 (Negative Log-Likelihood Function)

Note: 此时的数据集 $D$ 是已知的，所以概率 $p(D|\theta)$ 是关于 $\theta$ 的函数，这并不是概率密度函数 (Probability Density Function)，并且不同于 $p(x)$ 的形状

[关于 PDF, PMF 和 CDF](https://blog.csdn.net/wzgbm/article/details/51680540):

> - 概率密度函数 (Probability Density Function): 在数学中，连续型随机变量的概率密度函数是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数
> - 概率质量函数 (Probability Mass Functino): 在概率论中，概率质量函数是离散随机变量在各特定取值上的概率
> - 累积分布函数 (Cumulative Distribution Function): 又叫分布函数，是概率密度函数的积分，能完整描述一个实随机变量 $X$ 的概率分布

## 似然函数 (Data Likelihood Function)

进一步展开似然函数：

$$
\begin{align*}
l(\theta) & = \log p(D|\theta) \\ 
& = \log \prod_{i=1}^N p(x_i|\theta) \\
& = \sum_{i=1}^N \log p(x_i|\theta)
\end{align*}
$$

### 取得最大似然解的条件

#### 如果 $\theta$ 是标量

1. $\frac{\partial l}{\partial \theta} = 0 \rightarrow $ derivitive is zero at $\theta^*$
2. $\frac{\partial^2 l}{\partial \theta^2} < 0 \rightarrow $ concave at $\theta^*$
3. 检查 $\theta$ 的 value conditions

#### 如果 $\theta$ 是向量

1. $\nabla_\theta l(\theta) = 0$
2. $\nabla^2_\theta l(\theta) \prec 0 \rightarrow \theta^T H \theta \prec 0, \forall \theta$, that is, every direction will decrease the gradient
    - $H$ 是 Hessian 矩阵：$H = \frac{\partial^2 l}{\partial \theta \partial \theta^T}$  

******

# 示例

## 伯努利分布

假设我们有 $\theta = \pi, 0 \leq \pi < 1$, 和数据集 $D = \{x_1, x_2, ..., x_N\}$，那么此时的对数似然函数为：

$$
\begin{align*}
L(\theta) & = \sum_{i=1}^N \log p(x_i|\theta) \\
& = \sum_{i=1}^N \log \pi^x_i (1 - \pi)^{1-x_i} \\
& = \sum_{i=1}^N x_i\log\pi + (1-x_i)\log(1-\pi)
\end{align*}
$$

- 此时数据集中 1 的个数和 0 的个数将会被称为充分统计量 (sufficient statistic)，因为最后的结果仅与此充分统计量相关

因此若此时令 1 的个数为 $m = \sum_{i=1}^N x_i$:

$$
\frac{\partial}{\partial \pi} L(\pi) = \frac{m}{\pi} + \frac{N - m}{1 - \pi}
$$

令对参数 $\pi$ 的求导为 0，此时我们就能获得参数 $\pi$ 的最优解：

$$
\begin{align*}
\frac{m}{\pi} + \frac{N - m}{1 - \pi} & = 0 \\
\pi^* & = \frac{1}{N} \sum_{i=1}^N x_i
\end{align*}
$$

## 高斯分布

对于高斯分布的两个参数 $\mu$ 和 $\sigma^2$，和数据集 $D = \{x_1, x_2, ..., x_N\}$

那么此时的对数似然函数为：

$$
\begin{align*}
L(\theta) & = \sum_{i=1}^N \log p(x_i|\theta) \\
& = \sum_{i=1}^N \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(x-\mu)^2}{2\sigma^2}) \\
& = -\frac{N}{2}\log 2\pi - \frac{N}{2} \log \sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2
\end{align*}
$$

- 此时的充分统计量为 $\sum_{i=1}^N x_i^2$ 和 $\sum_{i=1}^N x_i$

为了求出最佳参数，我们提取与每个参数相关的项并对其求偏导，并令其值为 0：

对于 $\mu$:

$$
\begin{align*}
\frac{\partial}{\partial \mu} L(\theta) & = \frac{\partial}{\partial \mu} -\frac{1}{2\sigma^2} \sum_{i=1}^N (x_i - \mu)^2 = 0 \\
& \Rightarrow \frac{1}{\sigma^2} \sum_{i=1}^N (x_i - \mu) = 0 \\
& \Rightarrow \hat\mu = \frac{1}{N} \sum_{i=1}^N x_i
\end{align*}
$$

- 可以看出估计的 $\hat\mu$ 就是样本的平均值

对于 $\sigma^2$

$$
\begin{align*}
\frac{\partial}{\partial \sigma^2} L(\theta) & = \frac{\partial}{\partial \sigma^2} -\frac{N}{2} \log \sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2 = 0\\
& \Rightarrow \hat{\sigma^2} = \frac{1}{N}\sum_{i=1}^N (x_i - \hat\mu)^2
\end{align*}
$$

- 同样，我们可以看出估计的 $\hat{\sigma^2}$ 就是样本的方差

***

# 估计量 (Estimator)

> 若数据集是随机的，那么我们的估计量也将会是个随机变量，或者说是一个映射样本值和估计值的函数，那么就存在 mean 和 variance 以评估我们的估计的好坏

> 估计量，估计值等的概念可以参考此[文章](https://juejin.cn/post/7318445620011286565)

## Bias 和 Variance

对于我们的估计量 $\hat\theta = f(x_1, x_2, ..., x_N)$，我们可以计算 Bias 和 Variance 来评估估计量的收敛性和收敛速度

### Bias

> 评估是否会收敛到真实值 $\theta$

$$
\begin{align*}
Bias(\hat\theta) & = \mathbb{E}_{x_1, x_2, ..., x_N}[\hat\theta - \theta] \\
& = \mathbb{E}[\hat\theta] - \theta
\end{align*}
$$

- 若 Bias 为非 0，那么此时不管我们有多少个样本，最终还是不能收敛到真实值

### Variance

> 评估收敛需要多少样本？

$$
Var(\hat\theta) = \mathbb{E}_{x_1, x_2, ..., x_N} [(\hat\theta - \mathbb{E}[\hat\theta])^2]
$$
