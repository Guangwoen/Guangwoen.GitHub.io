---
layout: post
title: "React 개발중 axios 의 사용"
subtitle: "frontEnd"
date: 2022-06-05 19:47:30 +0800
categories: [tools]
---

# 프론트엔드 개발 - React and Axios

학업임무가 너무 많아 블로그 업뎃이 조금 느려졌다...

***

기말 프로젝트의 웹개발을 진행하면서 생긴 문제였다. 프론트엔드 개발이 거의 마무리를 지고 있었고 마지막 백엔드와의 연결만 남았었는데 axios 방법을 리용해 연결을 진행하려고 했었다.

## Axios

* Axios 는 Promise 에 기반한 HTTP 라이브러리 인데 React 프로젝트 혹은 Vue 프로젝트에서 많이 사용되고 있다.

## 생긴 문제

* axios 의 사용이 쉬워보여 직접 사용을 하려했었다. axios의 콜백함수에서 state 변경을 해주는 코드가 있었는데 마지막 실행결과를 보면 state 의 변경이 실현되지 않고 원래의 값을 유지하고 있었다.

* 콜백함수에서 console.log 를하여 수출해본결과 axios 신청을 보낸후 받아온 데이타들은 또 정상적이게 수출이 되였다.

* axios 의 post 방법과 get 방법의 차이에서 생긴문제인가 했더니 post 와 get 방법의 실행결과는 똑같았고 state의 변경은 여전히 진행되지 않았다.

***

## 원인분석

* axios 방법의 진행은 asynchronize인 방법이였다. 즉 axios방법을 사용한후 그아래의 코드를 계속 읽어주며 실행해주는데 axios방법의 외부에서 state를 관찰해본다면 axios방법을 사용하기 전의 state와 똑같았었던것이다.

* 이때 우리는 이함수의 실행을 기다려주는 await문구를 사용해주면 된다. await문구는 async문구가 들어있는 fucntion속에서만 사용할수 있는데 await뒤의 코드의 진행을 기다려주는 문구였다.

* 만약 이 async의 function이 하나의 Component였다면 await를 사용하는 방법은 또 async rendering의 문제를 초래하게되는데 결국 우리의 앱이 점점 어려워지게 한다.

***

## 해결방법

* React class에서 componentDidMount()의 방법에서 axios방법을 사용하는것이 가장 적절한 방법이다.

* componentDidMount()방법은 해당한 class가 mount되였을때, 즉 DOM tree에 추가되였을때 즉시 사용되는 방법인데 이방법은 data request 를 진행하기 가장좋은 곳이다.

* 렌더링을 해줄 class를 써준후 먼저 constructor() 함수에서 이 class의 state들을 초기화 해준다. 그후 componentDidMount()함수에서 axios방법을 사용한후 콜백함수에서 class의 state를 받아온 data들로 변경을해준다면 그후 render()함수에서 보게될 state는 axios신청에서 받아온 데이터일것이다. 

