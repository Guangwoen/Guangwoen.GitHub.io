---
layout: post
title: "机器学习 (CS5487) - 维度"
subtitle: "Machine Learning"
date: 2025-01-18 23:37:00 +0800
categories: [AI]
---

# 维度 (Dimensionality)

## 线性降维 (Linear Dimensionality Reduction)

> 我们的目标是减少相互关联的多个特征，这些相互关联的特征在原空间的线性低维子空间中被称为是 "live" 的

### 主成分分析 (Principle Component Analysis (PCA))

主要思想：如果数据在子空间中 live，那么这些数据将会在原空间中看起来是平坦的 (flat)，例如如果数据服从高斯分布，那么概率分布函数将会是一个扁椭圆形

令 $(\lambda_i, v_i)$ 为协方差矩阵 $\Sigma$ 的特征值 (eigenvalue) 和特征向量 (eigenvector)：

- $\Sigma = V \Lambda V^T, V = [v_1, ..., v_d], \Lambda = \begin{bmatrix} \lambda_1 & ... & 0 \\ ... & & ... \\0 & ... & \lambda_d \end{bmatrix}$
- 每个 $\lambda_i$ 代表椭圆的轴线
- $v_i$ 定义了其宽度

协方差矩阵 $\Sigma$ 的特征值表明了在哪个轴线上是平坦的

- 那我们就可以选择具有最大的特征值的轴线作为主成分 (principle component)

#### 算法步骤

给定数据 $D = \{x_1, ..., x_N\}$ 和维度 $k$

1. 估计高斯分布 (Estimate Gaussian)

$$
\begin{align*}
& \mu = \frac{1}{N} \sum_{i=1}^N x_i \\
& \Sigma = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
\end{align*}
$$

2. 特征值分解 (Eigen Decomposition)

$$
\Sigma = V \Lambda V^T
$$

3. 特征值排序 (Order the Eigenvalues)

$$
\lambda_1 \geq \lambda_2 ... > 0
$$

4. 选择 top-k 特征向量 (Select Top-K Eigenvectors)

$$
\Phi = [v_1, ..., v_k]
$$

5. 映射 (Project X onto $\Sigma$)

$$
z = \Phi^T (x - \mu)
$$

- 这也被称为 PCA 参数 (PCA Coefficients)

6. 将 $z$ 作为新的 feature vector

#### 注意点

- 映射后的方差，需要最大化：$\left\|z_i\right\|^2, z_i = \Phi^T(x_i - \mu)$
- 映射时的 reconstruction error，需要最小化：$\sum_{i=1}^N \left\|x_i - \Phi^T(z_i + \mu)\right\|^2$
- 如何选择 k?
    - 选择在下游任务中有效的 k
    - 选择能保留 p% 的方差的 k: $p = \frac{\sum_i^k \lambda_i}{\sum_j^d \lambda_j}$

> PCA 方法在方差的表示中是最优的，但是在分类任务中是次优的

### Fisher 线性判别分析 (Fisher's Linear Discriminant Analysis (LDA))

主要思想：找到可以最大化区别每个类的映射：$z = w^Tx$

| |原空间|映射后的一维空间|
|---|---|---|
|均值 (Mean)| $\mu_j = \frac{1}{N_j} \sum_{x_i \in C_j} x_i$ | $m_j = w^T \mu_j$ |
|散度 (Scatter)| $S_j \sum_{x_i \in C_j} (x_i - \mu_j)(x_i - \mu_j)^T$ | $S_j = w^TS_jw$ |

- 我们的目标就是最大化映射后每个 class 的均值之间的距离：$(m_1 - m_2)^2 = (w^T(\mu_1 - \mu_2))^2$

Fisher's Idiom:

$$
\begin{align*}
w^* & = \mathop{\arg\max}_w \frac{(m_1 - m_2)^2}{S_1+S_2} \\
& = \mathop{\arg\max}_w \frac{w^TS_Bw}{w^TS_ww}
\end{align*}
$$

其中
- $S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T, S_w = S_1 + S_2$
- $(m_1 - m_2)^2$ 被称为类间散度 (between-class scatter)，$S_1+S_2$ 被称为类内散度 (within-class scatter)
- 那么就有：$w^* = (S_1 + S_2)^{-1}(\mu_1 - \mu_2)$

> 此时分开两个类的超平面的方差就是 $\Sigma = \frac{1}{N}(S_1+S_2)$
> -> FLD 在 2 个类的高斯具有相同方差时会有最优解
