---
layout: post
title: "머신러닝 배우기 - 로지스틱 회귀"
subtitle: "Machine Learning"
date: 2022-05-03 15:15:44 +0800
categories: [AI]
---

# 머신러닝 배우기 - 로지스틱 회귀(Logistic Regression)

선형회귀에 이어 로지스틱 회귀에 대해 알아보자.

***

## 선형회귀와 로지스틱 회귀

* 선형회귀는 종속변수의 평균이 독립변수와 회귀계수들의 선형결합으로 된 회귀모형을 말하며 회귀계수를 선형 결합으로 표현할수 있는 모형을 뜻한다.

* 하지만 현실생활에서는 선형 모델보다는 비선형인 모델들이 더 많이 존재한다. 이때 더 일반적인 선형회귀가 바로 일반화선형회귀이다(Generalized Linear Regression). 일반화선형회귀는 종속변수를 적절한 함수로 변화시킨 f(y) (link function)를 독립변수와 회귀계수의 선형결합으로 모형화 한것이다. 그중 가장 대표적인 일반화선형회귀가 바로 로지스틱 회귀와 Cox의 비례위험회귀 이다.

***

## 로지스틱 회귀

로지스틱 회귀는 분류문제에서 많이 사용하게 된다. 즉 사건 혹은 사물의 특징을 입력으로 하여 이산값을 수출로 하는것이다.

* 분류하는 과정에서 일반적인 unit-step function 을 사용할수도 있지만 이러한 함수는 한가지 문제가 존재한다. 바로 z 가 0 의 값을 취하는곳에서 함수는 련속적이지 않는것이다. 이런 성질은 회귀분석에서 많은 문제들을 초래할수 있다.

* 로지스틱에서 사용하는 link function 은 바로 logistic function 이다: y = 1 / (1 + e^-z). 이 함수를 변환시키면: z = ln(y / (1 - y)) 이러한 등식을 얻을수도 있다. 이중에서 바로 y / (1 - y) 는 odds 를 대표한다.

* 이함수는 부의 무한대에서 0의 값에 접근하고 정의 무한대에서는 1의 값에 접근하여 이분형문제(성공/실패)에서 사용하게 된다. 뿐만아니라 이 함수는 x 가 어떤 값을 취하든지 함수는 항상 도함수를 가질수 있어 아주좋은 수학성질을 가지고 있다. 함수의 형태가 'S'형을 나타내여 Sigmoid 함수라고도 부른다.

* 그래서 우리는 함수값을 분류값에 사상하여 로지스틱 회귀를 이렇게 모델링할수 있다: y = 1 / (1 + e^-(wx + b)).

## 손실함수

* 선형회귀에서 우리는 평균제곱 손실함수로 모델의 좋고나쁨을 판단하였는데 로지스틱회귀에서도 이러한 손실함수를 리용할수있다. 하지만 우리가 선형회귀에서의 평균제곱 손실함수를 직접 로지스틱회귀에 사용한다면 경사하강법을 사용할때 최적의 값에 이르는데 시간이 오래걸릴뿐만 아니라 국부 최소치에 머무를수 있다.

* 그래서 우리는 로지스틱 회귀에서 교차 엔트로피 손실함수(Cross Entropy Loss)를 사용하게 된다.

    > ![cee](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/050301.png)

    > ![mse](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/050302.png)

* 교차 엔트로피 손실함수를 사용할때 로지스틱함수에 e 와 손실함수에서의 ln 이 서로 소거되여 없어지기에 sigmoid 함수에 대해 도함수를 계산할 필요가 없기에 계산이 빨라질수있다.

* 핵심코드를 알아보자:
    
    ```python
    learn_rate = 0.005  # 학습률
    iter = 100  # 반복회수

    # w 와 b 의 초기화
    np.random.seed(612)
    w = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())

    cross_train = []  # 교차 엔트로피 손실 저장
    acc_train = []  # 정확률 저장

    for i in range(0, iter + 1):
        with tf.GradientTape() as tape:
            pred_train = 1 / (1 + tf.exp(-(w * x + b)))
            Loss_train = -tf.reduce_mean(y_train * tf.math.log(pred_train) + (1 - y_train) * tf.math.log(1 - pred_train))
            Accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train < 0.5, 0, 1), y_train), tf.float32))

        cross_train.append(Loss_train)
        acc_train.append(Accuracy_train)

        dL_dw, dL_db = tape.gradient(Loss_train, [w, b])

        # 경사하강법
        w.assign_sub(learn_rate * dL_dw)
        b.assign_sub(learn_rate * dL_db)
    ```

* 다변량일 경우 선형회귀처럼 행렬에 대해 계산을 해주면 된다.

***

## 다중 분류(Multi-class classification)

* 이상의 로지스틱 회귀방법으로 우리는 쉽게 선형 분류를 실현할수 있었다. 하지만 현실생활에서는 종종 여러가지의 종류의 물건 혹은 사건들을 분류해야하는 경우가 더 많기에 다중 분류 처리방법이 필요하다.

* 일반의 이중 분류를 진행할때 태그는 1 과 0 이였지만 다중 분류를 실현할때에도 이러한 자연순서의 태그를 사용한다면 예측한 결과가 완전히 다른 종류로 나타나게 된다. 그래서 우리가 다중 분류를 실현할때 One-Hot Encoding 의 방법으로 이 태그들을 표시해준다.

    > One-Hot Encoding 이란 0 과 1 로 이루어진 벡터로 표시하는 방법인데 각각 부동한 자리에 1 을 표시해주기에 이런 방식을 One-Hot 라고 지은것이다.
    > 례를 들면 3가지 종류가 있을때 이 3가지 종류들을 각각 (0, 0, 1), (0, 1, 0), (1, 0, 0) 로 표시할수 있다.

    ### Softmax()

    * 다중 분류에서 우리는 또 softmax()란 함수를 사용하게된다. 일반적인 max()함수는 hard한 함수인데 이 hard는 몇개의 수치중에서 가장큰것만 골라내는 것이다. 하지만 softmax() 함수는 가장 큰것을 골라주는 것이 아닌 최대치를 가질수 있는 가능성들을 계산해주는 함수이다.

        > softmax()함수의 공식은 이렇다:

        ![softmax](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/050303.png)

    * Softmax 함수는 Logistic 함수가 다중분류에서의 전환이다.

* 다중 분류에서의 교차 엔트로피 손실 함수는 이렇다:
  
    ![mentro](https://raw.githubusercontent.com/Guangwoen/Guangwoen.GitHub.io/main/pics/050304.png)

* 핵심코드를 알아보자:
    
    ```python
    # Data
    x0_train = np.ones(num_train).reshape(-1, 1)
    X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf, tf.float32)
    Y_train = tf.one_hot(tf.constant(y_train, dtype=tf.int32), 3)

    # 초기화
    learn_rate = 0.2
    iter = 500

    np.random.seed(612)
    W = tf.Variable(np.random.randn(3, 3), dtype=tf.float32)

    acc = []  # 정확률
    cce = []  # 다중 뷴류 교차 엔트로피 손실

    for i in range(0, iter + 1):
        with tf.GradientTape() as tape:
            PRED_train = tf.nn.softmax(tf.matmul(X_train, W))
            Loss_train = -tf.reduce_sum(Y_train * tf.math.log(PRED_train)) / num_train

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1), y_train), tf.float32))

        acc.append(accuracy)
        cce.append(Loss_train)

        dL_dW = tape.gradient(Loss_train, W)
        W.assign_sub(learn_rate * dL_dW)
    ```
