---
layout: post
title: "两数相乘算法"
subtitle: "Big-O"
date: 2022-09-19 19:35:23 +0800
categories: [daily]
---

# ⌨️ 算法设计与分析

***

## 两数相乘算法 & Big-O

对于n位数的相乘，按照一般的方法，最坏的情况下需要![[Pasted image 20220915185546.png]]次的乘法计算和![[Pasted image 20220915185552.png]]次的加法计算

我们可以利用 Big-O 表达式表达某个算法的近似运行时间复杂度; 例如一般的两数相乘方法的复杂度是 ![[Pasted image 20220915192729.png]]

***

## 分而治之

范例算法：
	1. 将一个问题切分为小的子问题
	2. **递归地**解决各个子问题
	3. 组合各个子问题的解得出最终的大问题的解

**两数相乘的分治法（两个4位数的相乘）：**
- *1234 • 5678*
= ![[Pasted image 20220915193822.png]]

- 这样子问题就变成了4个2位数相乘的问题

- 伪代码:
	```
	MULTIPLY(x, y):
		if(n = 1):
			return x * y;
		write x as a * 10^n/2 + b
		write y as c * 10^n/2 + d
		ac = MULTIPLY(a, c)
		ad = MULTIPLY(a, d)
		bc = MULTIPLY(b, c)
		bd = MULTIPLY(b, d)
		return ac*10^n + (ad + bc)*10^n/2 + bd
	```

- 根据递归调用树，此时的代价仍然是![[Pasted image 20220915192729.png]]

**KARATSUBA 算法**
- 因为存在这样的等式：(a + b)(c + d) - ac - bd = ad + bc, 上述的一般的分治法的4个子问题转换为3个子问题
- 根据递归调用树, KARATSUBA 算法的代价是![[Pasted image 20220915201355.png]]

***

## 渐进分析

![[截屏2022-09-15 20.23.56.png]]