---
layouts: post
title: "React-router-dom v6와 v5의 차이"
subtitle: "frontEnd React"
date: 2022-04-18 11:19:25 +0800
categories: [tools]
---

# React-router-dom 의 새로운 사용법

프론트엔드 개발을 진행하면서 Router 설정에서 새로운 사용법이 생겼다.

* V5에서 우리는 Router 의 설정에서 이렇게 사용했다.

    ```TypeScript
    React.render(
        <BrowserRouter history={ createBrowserHistory() }>
            <Switch>
                <Route exact path="/" component={ Index }>
                <Route path="/app" component={ App }>
            </Switch>
        </BrowserRouter>,
        document.getElementById("root")
    )
    ```

* 하지만 이런 사용법은 V6에서는 컴파일에 실패했다. 터미널에 뜨는 에러는 Switch not found in 'react-router-dom' 이였지만 IDEA에서는 정확한 사용이라고 알려지고 있었다.

***

V6에서의 사용법은 이렇다

* Switch는 Routes라는 이름으로 변경되였고 Route속의 component는 element로 대체됨과 동시에 파라미터는 `<Component/>` 이런식으로 써주면 된다.

    ```TypeScript
    React.render(
        <BrowserRouter history={ createBrowserHistory() }>
            <Switch>
                <Route exact path="/" element={ <Index/> }>
                <Route path="/app" element={ <App/> }>
            </Switch>
        </BrowserRouter>,
        document.getElementByID("root")
    )
    ```

* 이렇게 변경해주었더니 에러는 사라지고 웹은 정상적으로 작동하였다.

***

* useHitory() 함수도 useNavigate()로 대체되였고 history.push(path)/history.replace(path) 조작들도 navigate(path)/navigate(path, {replace: true}) 로 변경되였다.

React Router V6은 더욱 편리한 사용법이 있을뿐만 아니라 package의 크기도 절반으로 줄어들었다.