---
layout: post
title: "机器学习 (CS5487) - 核密度估计与均值漂移"
subtitle: "Machine Learning"
date: 2025-01-11 14:08:00 +0800
categories: [AI]
---

# 核密度估计与均值漂移 (KDE and MeanShift)

> 在 GMM 的方法中，我们假设了数据是由若干个高斯模型的加权和组成的，我们的目标是去求解每个模型的权重和每个模型的参数

核密度估计 (Kernel Density Estimation (KDE)) 方法的原理与 GMM 不同，我们不会假设数据服从一个固定的模型分布，也就是无参数 (non-parametric) 的方法

## 基本思想

高斯混合模型是基于多个参数的混合模型，但是核密度估计并不会假设一个模型，不利用有关数据分布的先验知识，对数据分布不附家任何规定，而是通过直接拟合数据的方法来对整体分布进行估计

例如，直方图就是一个很好的例子，通过直方图，我们可以很直观地看出目标数据的分布情况：

- 矩形的面积为该区间的频率
- 矩形的高度为该区间的平均频率密度

极限思维：我们使用微分的思想，若将直方图的组距减小，当这个宽度足够小时，我们的直方图就会变成一条曲线，这个曲线也就自然是我们要求的概率密度曲线

对于一个特定的区间 $R$，随机变量 $x$ 落入该区间的概率 $\pi$ 可以写成：

$$
\begin{align*}
\pi & = p(x \in R) \\
& = \int_{x \in R} p(x) \mathrm{d}x
\end{align*}
$$

存在一组样本 $D = \{x_1, x_2, ..., x_N\}$ 时，若区间 $R$ 内的样本个数为 $k_R$，那么对于 $\pi$ 的最大似然估计就是：

$$
\hat\pi = \frac{k_R}{N}
$$

当 $R$ 足够小时，$p(x)$ 就可以被视为多个矩形的组合，假设区间 $R$ 的大小为 $V_R$，那么此时就有：

$$
\hat\pi = p(x) V_R = \frac{k_R}{N} \Rightarrow \hat p(x) = \frac{k_R}{NV_R}, x \in R
$$

- 显然，区间 $R$ 的大小选择是至关重要的

> - 如果固定 $V_R$，变化 $k_R$，此时就变成了核密度估计方法 (或者被称为 Parzen 窗方法 (Parzen Window))
> - 如果固定 $k_R$，变化 $V_R$，此时就变成了 K-邻近算法 (K-Nearest Neighbors Estimation)

## 核密度估计 (Kernel Density Estimation)

> 核密度估计，就是通过核函数将每个数据点的数据 + 带宽作为核函数的参数，得到 N 个核函数，再线性叠加就形成了核密度估计的函数，归一化后就是核密度概率函数

核密度估计的方法用公式来表示就是：

$$
\hat p(x) = \frac{1}{N} \frac{k_R}{V_R} = \frac{1}{Nh^d} \sum_{i=1}^N k(\frac{x - x_i}{h})
$$

- 其中 $h$ 表示的就是带宽，是用于函数平滑的参数
- $d$ 表示随机变量的维度
- $k(.)$ 表示核函数

### 核函数

并不是所有的函数都能作为核函数来使用，因此核函数存在一些条件：

$$
\begin{cases}
k(x) \geq 0 \\
\int k(x) \mathrm{d} x = 1
\end{cases}
$$

- 理论上，所有平滑的峰值函数均可作为 KDE 的核函数来使用，只要对归一化后的 KDE 而言，该函数的曲线下方的面积和等于 1 即可

#### 矩形

直方图就是使用矩形的核函数的方法，该核函数可以用以下方式来表达：

$$
k(x) = \begin{cases}
\frac{1}{2} & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

#### Epanechnikov 曲线

$$
k(x) = \begin{cases}
\frac{3}{4} (1 - x^2) & \text{if } |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

#### 高斯曲线

$$
k(x) = \frac{1}{\sqrt{2\pi}} \exp{(-\frac{1}{2}x^2)}
$$

- 此时，概率密度函数就是 $\hat p(x) = \frac{1}{N} \sum_{i=1}^N \mathcal{N}(x|x_i, h^2I)$

### 带宽

> 核密度估计的结果不依赖于区间的划分，而是依赖于核的形状和宽度（即带宽），带宽过小可能会导致过拟合，即在数据点位置有过高的峰值，而在数据点之间几乎为 0；带宽过大则可能会导致估计过于平滑，无法捕捉到数据的真实分布

也就是说：

$$
\begin{cases}
h \to \infty \Rightarrow bias \to 0, var \to \infty \\
h \to 0 \Rightarrow bias \to \infty, var \to 0
\end{cases}
$$

- 因此，在实际应用算法时，应该去选择合适的带宽，以确保算法的性能

## 均值漂移算法 (Mean-Shift Algorithm)

> 均值漂移算法的目标是去找出一个 mode (peak)

方法步骤：

1. 选择任意一点 $\hat x$ ($\hat x$ 可以是一个样本点)
2. 对 $\hat x$ 的 peak 进行梯度上升 (gradient ascending): $\hat x^{(k+1)} \gets \hat x^{(k)} + \lambda \hat p(\hat x^{(k)})$
3. 重复直到 $\hat x$ 收敛到一个 mode
4. 对所有样本值 $x_i \in D$ 执行以上步骤，直到找出所有的 mode

### 聚类

定义径向对称函数 (radially symmetric kernels)：

$$
k(x) = \alpha \bar k(\left\|x\right\|^2)
$$

其中

- $\alpha$ 是个常数
- $\bar k$ 是 kernel profile
- $\left\|x\right\|^2$ 是距离

例如，对于高斯核：

$$
k(x) = \frac{1}{(2\pi)^{\frac{d}{2}}}\exp{(-\frac{1}{2} \left\|x\right\|^2)}
$$

- $\bar k(x) = e^{-\frac{1}{2} r}$
    - $r = \left\|x\right\|^2$
- $\alpha = \frac{1}{(2\pi)^{\frac{d}{2}}}$

#### 密度估计

此时，密度估计就可以被写为：

$$
\hat p(x) = \frac{\alpha}{Nh^d}\sum_{i=1}^N \bar k(\left\|\frac{x-x_i}{h}\right\|^2)
$$

对于高斯核，定义 $\bar g(r) = - \bar k'(r)$，则梯度上升的步骤就可以被写为：

$$
\hat x^{(k+1)} = \frac{\sum_i x_i \bar g(\left \| \frac{\hat x^{(k)}-x_i}{h} \right \|^2)}{\sum_i \bar g(\left \| \frac{\hat x^{(k)}-x_i}{h} \right \|^2)}
$$
