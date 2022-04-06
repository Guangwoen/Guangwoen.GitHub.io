---
layout: post
title: "UML set name direction"
subtitle: "starUML"
date: 2022-04-06 17:47:01 +0800
categories: [tools]
---

# Add direction of association name

* UML 앱으로 모델링을 진행할때 association 관계를 더 명확하게 표현하기 위해 삼각형으로 방향을 가리켜 줘야할 때가 있었다.

* 세주일전 starUML 에서 모델링을 진행하였는데 Enterprise Architect 와 조작이 조금씩 달랐다.

* 두개의 class 를 만들고 association 으로 련결시킨후 association 이름에 방향을 달아주려고 했지만 어느 옵션에도 존재하지않았다.

* 그래서 그냥 방향표기를 달지않은채 계속 모델링을 진행하였는데 오늘 starUML공식 웹사이트에 들어가보니 3월 26일 업데이트된 최신버전인 starUML 5.0에서는 association 관계의 이름에 방향을 달아주는 공능이 추가되였다.

# 조작방법

* association 에 이름을 달아준후 이름을 두번 클릭하면 왼쪽에 constraint, memo등이 있는 공능란에서 새로 추가된 "order of ends" 라는 옵션에서 우리가 원하는 방향을 달아줄수 있다.