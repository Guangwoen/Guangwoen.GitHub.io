---
layout: post
title: "머신러닝 배우기 - 선형 회귀(1)"
subtitle: "Machine Learning"
date: 2022-04-10 14:58:31 +0800
categories: [machine_learning]
---

# 머신러닝 배우기 - 선형 회귀(Linear Regression)

> 자동차의 자율주행에서의 컴퓨터비전 알고리즘의 학습을 목표로하여 머신러닝에 대한 학습을 시작한다.

***

* 선형회귀란? 우리가 수학공부를 하면서 회귀직선에 대한 학습을 진행한적이 있었다. 그중에서도 통계학에서 회귀직선을 자주 사용한다. 머신러닝에서 선형회귀란 지도학습(Supervised Learning)의 한가지로서 종속변수 Y 와 한개 이상의 독립변수 X 와의 선형 상관관계를 모델링하는 분석방법이다.

* 례를 들면 사과의 값이 알의 크기와 관계된다고 가정할때 여러조의 수치를 제공하여 알의 크기와 사과의 값과의 관계를 추리하기위해 회귀방정식을 가정한후 그 회귀방정식의 미지수들을 계산하여주는 과정이다. 그러면 우리가 사과알의 크기만 알고있다면 대략적인 값을 추리할수있다. 간단하게 말하여 데이터 속에서 가장 적합한 최적선을 찾는 행위를 말한다. 

***

## 단변량 선형 회귀 - Simple Linear Regression

먼저 독립변수가 하나만 존재할때의 경우를 생각해보자.

* 수학지식을 리용해 우리는 쉽게 이런 모델 방정식을 쓸수있다.
  
  > y = W * x + b

* 이중 W 와 b 는 수학에서 각각 Slope 와 Intercept 를 뜻하는데 머신러닝 선형 회귀에서는 각각 Weight 와 Bias 라고 부른다.

* 우리의 목표는 바로 주어진 수치들에 의하여 Weight 와 Bias 의 값을 구하는것이다.

* 주어진수치에 의하여 그릴수있는 회귀직선은 무한개가 존재하는데 그중에서 우리는 최적의 직선을 선택하여야한다. 그래서 우리는 주어진 점과 회귀직선사이의 Euclidean Distance 를 표준으로 삼아 최적의 회귀직선을 그려낼수있다. 즉 잔차(residual) 가 가장 작은 회귀직선을 선택한다.

    ![residual_loss](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2015.58.50.png)

* 이공식에 의해 우리는 잔차들의 합을 구할수 있지만 주어진 점들이 회귀직선의 두쪽에 모두 분포될수있기에 정의 값과 부의 값들이 서로 소거될수가 있다.

* 그래서 우리는 이렇게 잔차의 평방의 합을 쓰는 손실함수(Square Loss)라는 공식을 쓰게될수있다.

  * Square Loss  ![Square_loss](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2016.08.27.png)

  * Mean Square Loss ![Mean_square_loss](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2016.12.02.png)
  
* 위의 평균 제곱 오차 함수(Mean Square Loss) 공식에 의하여 우리는 최소이승법을 사용할수있다. 이 함수가 최소치를 가질때의 W 와 b 의 값이 바로 우리가 원하는 최적의 회귀직선의 W 와 b 인것이다.

* 최소치를 구하기 위해 우리는 이 방정식에 대하여 편도함수를 구해 편도함수가 각각 0의 값을취하는곳을 구해준다.

  * 편도함수: ![partial_derivative](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2017.18.13.png)

  * 해결식: ![solution](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2017.27.09.png)

* 위의 해결식은 Analytical solution(근사한 값으로 대체하여 얻은 결과) 인데 Closed-form solutions(엄격한 계산과 추리를 통해 얻은 결과) 으로도 해결식을 세울수 있지만 코딩에 편리하기위해 Analytical solution을 선택하는것이다.

* 해결식을 python 코드로 적어본다면:
  
    ```python
    import tensorflow as tf
    import numpy as np

    x = tf.constant(np.random.rand(10))
    y = tf.constant(np.random.rand(10))

    meanX = tf.reduce_mean(x)
    meanY = tf.reduce_mean(y)

    sumXY = tf.reduce_sum((x - meanX) * (y - meanY))
    sumX = tf.reduce_sum((x - meanX) * (x - meanX))

    w = sumXY / sumX
    b = meanY - meanX * w

    print("대응된 회귀방적식은: y=", round(w.numpy(), 2), "* x +", round(b.numpy(), 2))
    ```

 * 위의 계산은 python 의 list 로도 할수 있을뿐만 아니라 numpy array 를 사용하여 계산할수도 있다.

 이러면 우리가 어떠한 x 에 대응되는 y 의 값을 구할때 이 회귀방정식에 대입하면 쉽게 그 근사치를 알아낼수가 있다.

***

## 다변량 선형 회귀 - Multivariate Linear Regression

단변량 회귀와 달리 다변량 회귀에서는 독립변수가 1개 이상 존재한다. 따라서 대응되는 weight 를 구하는절차도 조금씩 달라지게 되지만 우리의 목표는 여전히 변량앞의 weight 와 bias 를 구해주는것이다.

* n개의변량의 회귀방정식에 대해 우리는 이렇게 모델링을 할수있다.
    
    * ![modeling](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2018.10.59.png)

* m개의 표본에 대하여 우린 이런 련립방정식들을 세울수 있다.

    * ![modeling_](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.03.49.png)

* 행렬로 전환시킨다.

    * ![array](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.06.20.png)

    * 즉 이렇게 볼수 있다. ![re](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.23.11.png)

* 이런 다변량인 경우에도 위에서의 손실함수를 쓸수있는데 다만 행렬에 대한 계산들이 포함되여있다.

    * 손실함수: ![loss](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.24.10.png)

    * ![loss_sim](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.25.08.png)

 * 손실함수가 최소치를 가지는 W의 값들, 즉 W 행렬을 구하면 되는데 이것도 손실함수가 W에 관한 편도함수를 계산하여준다. 그럼 마지막 우리가 계산해줄 행렬은 이렇게 된다:
    
    * ![resu](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/%E6%88%AA%E5%B1%8F2022-04-10%2020.28.30.png)

* 코드로 알아보자

    ```python
    import numpy as np
    x1 = np.random.rand(10)
    x2 = np.random.rand(10)
    y = np.random.rand(10)

    x0 = np.ones(len(x1))
    X = np.stack((x0, x1, x2), axis=1) # axis=1 의 방향에 따라 push 해준다
    Y = np.array(y).reshape(-1, 1)

    Xt = np.transpose(X) # 전치행렬(T)
    XtX_1 = np.linalg.inv(np.matmul(Xt, X)) # 역행렬(^-1)
    XtX_1_Xt = np.matmul(XtX_1, Xt) # 행렬곱하기
    W = np.matmul(XtX_1_Xt, Y) 

    W = W.reshape(-1)

    print("대응된 회귀 방정식은: Y=", round(W[1], 2), "*x1+", round(W[2], 2), "*x2+", round(W[0], 2))
    ```
