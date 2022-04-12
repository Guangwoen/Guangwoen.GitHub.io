---
layouts: post
title: "머신러닝 배우기 - 선형 회귀(2)"
subtitle: "Machine Learning"
date: 2022-04-12 21:21:32 +0800
categories: [machine_learning]
---

# 머신러닝 배우기 - 선형 회귀 - 경사하강법

오늘에는 선형 회귀 방법중에서의 경사하강법을 배워보려한다.

* 경사하강법은 search에 기반한 최적의 방법이면서 오차가 가장 작은 방법중의 하나다.

* 우리가 최적의 회귀 직선을 찾으려 할때 손실함수를 기준으로 손실함수가 가장 작은 값을 취하는 곳을 찾는것을 목표로 한다.

* 이 손실함수의 최고 차항은 보통 2차 이상인데 경사하강법이란 함수 값이 낮아지는 쪽으로 독립변수의 값을 변경 시켜 최종으로 함수가 최소의 값을 취하게 하는 독립변수를 찾는 과정이다. 이것을 실현하기 위해 우리는 손실함수중의 미지수들에 대하여 미분을 진행 시키는데 미분의 값이 작을수록 경사도가 작다는걸 뜻하기에 미분의 값을 조금씩 감소시키면 결국 함수의 최소의 값을 찾을수가 있다.

***

# 단변량

* 단변량일 경우, 우리의 모델링은 y = W * x + b 인데 손실함수는 미지수 W 와 b 에 관한 함수가 된다. W 와 b 에 관한 편도함수를 계산해본다면:
  
  * ![grad](041201.png)

* 이때 우리는 이런 공식으로 다음 과정에서의 W 와 b 의 값을 이렇게 업데이트 해줄수 있다.

  * ![upd](041202.png)

* 평균 손실함수에 대해서는 계수부분에 1/n을 추가해주면 된다.

* 바로 이 공식을 통해 우리는 미분의 값이 작아지는 방향따라 W 와 b 를 업데이트 시키면서 손실함수가 최소의 값을 취하는곳을 찾아줄수 있다.

* 이중에서 계수 에타는 학습률(Learning rate)를 뜻하는데 이 변수는 머신러닝에서 엄청 중요한 변수이다. 경사하강법의 결과에 큰 영향을 줄만큼 학습률의 선택은 아주 중요하다. 학습률은 모델링을 해주는 사람이 경험에 의해 임의로 선택해주는 값이다.

* 학습률을 너무 작게 선택했을때 W 와 b 의 변화가 너무 작아 최적치를 계산해주는데까지 비교적 오랜 시간이 걸린다. 학습률을 너무 크게 선택했을때 한번에 너무 큰 폭을 뛰여넘어 최적치를 놓칠수있을 뿐만 아니라 이 손실함수 사이에서 진동이 생길수 있다. 합리한 학습률을 선택하기위해 우리는 여러번의 실험을 거쳐 최적의 값을 얻어낼수 있다.

* 학습률의 선택을 마친후 우리는 W 와 b 의 값을 랜덤으로 선택해준다. 랜덤의 값으로부터 손실함수가 최소의 값을 취하는곳까지 변화하는 과정인것이다.

코드로 알아보자(tensorflow 실현):

  ```python
  import tensorflow as tf
  import numpy as np

  x = np.random.rand(10)
  y = np.random.rand(10)

  learn_rate = 0.0001  # 학습률
  iter_count = 100  # 반복회수

  np.random.seed(612)
  w = tf.Variable(np.random.randn())  # w 와 b 의 초기화
  b = tf.Variable(np.random.randn())

  mse = []

  for i in range(0, iter_count+1):
    with tf.GradientTape() as tape:  # tensorflow 가 제공한 미분계산 API
      pred = x * w + b
      Loss = 0.5 * tf.reduce_mean(tf.square(y - pred))
    mse.append(Loss)

    dL_dw, dL_db = tape.gradient(Loss, [w, b])  # Loss 함수 미분계산

    w.assign_sub(learn_rate * dL_dw)  # w, b의 업데이트
    b.assign_sub(learn_rate * dL_db)
  ```

***

# 다변량

다변량인경우 행렬계산이 포함된다.

  ```python
  import tensorflow as tf
  import numpy as np

  x1_ = np.random.rand(10)
  x2_ = np.random.rand(10)
  y_ = np.random.rand(10)
  size = len(x1)
  x0 = np.ones(size)
  x1 = (x1_ - x1_.min()) / (x1_.max() - x1_.min())  # 정규화
  x2 = (x2_ - x2_.min()) / (x2_.max() - x2_.min())
  X = np.stack((x0, x1, x2), axis=1)
  Y = y_.reshape(-1, 1)

  learn_rate = 0.01
  iter = 100

  np.random.seed(612)
  W = tf.Variable(np.random.rand(3, 1))  # 3행 1렬

  mse = []

  for i in range(0, iter+1):
    with tf.GradientTape as tape:
      PRED = tf.matmul(X, W)
      Loss = 0.5 * tf.reduce_mean(tf.square(Y - PRED))
    mse.append(Loss)

    dL_dW = tape.gradient(Loss, W)  # W 에 관한 미분
    W.assign_sub(learn_rate * dL_dW)
  ```