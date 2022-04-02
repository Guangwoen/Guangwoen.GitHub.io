---
layout: post
title: "jekyll 셋팅에서 생긴문제"
subtitle: "Timezone"
date: 2022-04-01 08:44:21 +0800
categories: [bugs]
---

# jekyll timezone

* 금방 블로그를 시작하면서 github page 기초에 jekyll 랑 같이 사용하는걸 선택햇다.

환경: MacOS Monterey

***

## 새 프로젝트 생성
Github 에서 먼저 새 프로젝트를 생성한후 Github pages 옵션을 선택한다. 

* jekyll 은 ruby에 기반하기에 터미널에서 명령  `brew install ruby` 통해 ruby 환경을 만들어준다.
  
* 그후 ruby 명령 `gem` 에기반하여 `gem install jekyll bundler` 을통해 jekyll 을 설치하여준다.
  
    >*만약 이때 "You don't have write permissions for the..." 에러나 나타난다면 맨앞에 'sudo' 를 넣어 슈퍼유저권한으로 실행하면 된다.*

* 마지막으로 github 에서 프로젝트를 로컬에 클론하여 `jekyll new . --force` 를 쳐넣으면 블로그를 쓸준비가 다된것이다. 터미널에서 `jekyll serve` 입력한후 '127.0.0.1:4000' 를 통하여 실시간으로 웹페이지 변경을 볼수있다.

블로그 쓸준비가 다되였다.

***

## 블로그 쓰기
jekyll 가 생성한 파일에서 우리가 변경해야할것들은 config.yml, _posts, _data, categories 들이 있다.

* config.yml
  >전체 사이트의 정보를 저장하는곳이라 여기에서 블로그 주인, 타임존, 마크다운 설정 등 여러가지 옵션들을 추가할수있다.

* _posts
  >우리가 쓴 블로그를 저장하는 파일, 하지만 블로그파일명은 무조건 jekyll 형식 - 'yyyy-mm-dd-title.markdown' 대로 써넣어야한다.
  
  >블로그 헤더에는 YAML헤더 정보가 있어야한다. 상세한것은 례로 생성한 블로그문건이 있기에 그 파일에서 보면된다.

* _data
  >우리가 새로운 카테고리를 생성할때 이파일하에 categories.yml 에서 추가하면된다. 새로운 카테고리는 같은 목록하에 'categories' 에서 html 문건도 추가하여야 실행이된다.

* categories
  >새로운 카테고리를 생성할때 대응된 html 을 넣어줘야한다. 내용은 YAML 헤더만 써넣으면된다.

* 로컬에서 수정을 마친후 git 명령을통해 업데이트 시키면 바로 웹페이지에서 변경을 볼수있다.

***

## 타임존 문제

* 새로운 블로그를 생성한후 로컬에서 실행을 해보니 터미널에서 `Skipping: _posts/2022-04-01-jekyll-problem.markdown has a future date` 에러가 생겼다.

* 'future date' 라고 한것을 봐서는 블로그 YAML 헤더의 date 변량에서 문제가 생긴것 같았는데 검사를 여러번 해보았지만 아무런 문제가 없는것 같앗다.

* 구글해보니 최신버전의 jekyll 2.0.0 은 tzinfo 에대한 issue 가 존재하였다.
    >jekyll 은 config.yml 에서 타임존 설정을 식별하지 못하였다면 자동으로 UTC 시간으로 변경하는것이였다.

    >구체적인 해결방법은 config.yml 문건에서 타임존 설정을 해준다: 위치는 중국에 있기에 `timezone: Asia/Shanghai` 을 한마디 넣어주면된다. Beijing시간으로 고치는줄로 알았었는데 여기서는 Shanghai 시간을 표준으로 하였다.

    >그후 우리가 쓴 블로그 문건 헤더에서 date 변량의 맨뒤에 `+0800` 을넣어 UTC 시간과의 시차를 표시하여주면 에러는 사라지고 웹페이지에서 성공적으로 우리가 쓴 블로그가 나타나였다.
    
***

이전의 버그들을 첫 블로그로 쓰려고했지만 오늘의 버그를 해결하기위해 꽤나 오랜 시간을 팔았기에 블로그를 시작하면서생긴 첫 issue 를 블로그의 시작으로 삼아봤다.
