---
layout: post
title: "MacOS에서 OpenMP 환경 설치"
subtitle: "Multi-processing"
date: 2022-04-02 15:12:11 +0800
categories: [tools]
---

# 공유 메모리 다중 처리 프로그래밍 API - OpenMP

### OpenMP(Open Multi-Processing, 오픈MP)는 공유 메모리 다중 처리 프로그래밍 API로, C, C++, 포트란 언어와, 유닉스 및 마이크로소프트 윈도우 플랫폼을 비롯한 여러 플랫폼을 지원한다.
***

* Multi-Processing 을 지지하는 API 는 여러가지가 존재한다. 그럼 그중에서 C언어에도 적응되는 OpenMP 의 MacOS 환경에서의 설정을 알아보자.

* 과당시간에서 교수님이 제공한 자료들은 모두 Windows 에 기반하여 CodeBlocks 컴파일러에서 설정하는 절차들이였다. 물론 이절차들은 맥북에서 CodeBlocks를 설치하여 하는것도 적용되지만 MacOS 버전의 CodeBlocks 는 언제부턴가 업데이트를 멈춰버려서 코딩하는것이 아주 불편했다.

* 생각해보니 C/C++ 코딩은 원래부터 `터미널 + gcc(g++)명령 + sublime 에디터` 해왔는지라 간단하게 OpenMP 라이브러리만 임포트하여서 하는것이 더욱 쉬운 방법일듯 했다.

***

## OpenMP 의 다운로드

* 터미널에 들어가 `homebrew` 도구를 리용해 `brew install libomp` 를 입력하면 저절로 알아서 필요한 문건들을 임포트 해준다.
    > Homebrew 도구는 MacOS 를 리용하는 개발자들에게 있어서 아주 유용한 도구인데 구글에서 검색하면 관련된 자료들을 쉽게 찾을수있다.

## GCC 의 reinstall

* Install 이 끝난후 바로 코딩을 해보니 `Undefined symbols for architecture x86_64...` 이런 에러가 나타낫다. 

* 처음보는 에러였는데 검색해보니 gcc 컴파일러 버전에 관한 문제였던것같았다.

* 그래서 다시 brew 명령을 통해 최신버전의 gcc 로 reinstall 하였다. `brew reinstall gcc`
    > 독자들의 gcc 버전이 어떤 버전인지 모르겠다면 그냥 reinstall 하는것을 추천함.

## 정확한 실행 절차

이상의 절차들이 완성이 되였지만 아직 끝난것이 아니다.
* 코딩할때 OpenMP 의 격식처럼 <omp.h> 헤더를 include 한후 `#pragma omp ...` 등 알맞게 편집을 완성하면 된다.

* 에디트가 완성된후 터미널에 들어가 컴파일을 진행할때 그냥 `gcc target.c -o target` 하면 무조건 에러가 뜬다.

* 이때 우리가 입력해야할 명령은 `gcc-11 -fopenmp target.c -o target` , 이렇게하면 성공적으로 컴파일이 완성된다. `-11` 은 gcc의 버전을 뜻하고 `-fopenmp` 는 우리가 쓴 C 문건에 OpenMP 를 사용한다고 컴파일러한테 알려주는 문법이다.


이상 MacOS 에서 OpenMP 를 리용한 Multi-Processing 이다.

