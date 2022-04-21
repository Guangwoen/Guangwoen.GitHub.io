---
layout: post
title: "Tree 의 직경 구하기"
subtitle: "Tree algorithm"
date: 2022-04-21 15:32:39 +0800
categories: [algorithm]
---

# Tree 와 Graph 의 직경 구하기

Tree 와 Graph 의 직경구하기란 간단하게말해서 제일 큰 길이를 구하는것이다.

* 이 알고리즘은 BFS 혹은 DFS 의 기초하에서 진행된다.

* 아래의 설명은 BFS 로 실현한다.

* 먼저 임의의 Node 에서부터 시작된다. 이 Node 로부터 시작해 우리는 BFS를 통해 가장 멀리 갈수있는 Node 까지 기록해준다. 그다음 우리는 이 Node 를 시작점으로 하여 다시 BFS로 이 Node 에서부터 가장 멀리갈수있는 Node를 기록한다. 즉 BFS 를 두번 진행 해주는 것이다. 그러면 첫번째 BFS의 끝점으로 부터 두번째 BFS의 끝점까지의 거리가 바로 이 Tree 혹은 Graph 의 직경인것이다.

* 코드로 알아보자:

    ```cpp
    #include <bits/stdc++.h>

    const int N = 10005;

    int begin, end; // 매번의 시작점과 끝점을 기록
    std::vector<int> edge[N]; // 그라프의 모양
    int distance[N]; // 거리 기록;
    int n; // Node 의 개수
    int furthest; // 최종 답안

    void bfs(int x) { // BFS
        std::queue<int> q;
        q.push(x);
        while(!q.empty()) {
            int head = q.front();
            q.pop();
            for(int i = 0; i < edge[head].size(); i++) {
                if(distance[edge[head][i]] == 0 && edge[head][i] != begin) {
                    distance[edge[head][i]] = distance[head] + 1;
                    q.push(edge[head][i]);
                    furthest = distance[edge[head][i]];
                    end = edge[head][i];
                }
            }
        }
    }

    int main() {
        std::cin >> n;
        for(int i = 0; i < 0; i++) {
            int a, b;
            std::cin >> a >> b;
            begin = a; // 시작점을 정해준다
            edge[a].push_back(b);
            edge[b].push_back(a);
        }
        bfs(begin);
        for(int i = 1; i <= n; i++) { // refresh
            distance[i] = 0;
        }
        begin = end; // 두번째의 bfs 에서는 끝점이 시작점이된다
        bfs(end);
        std::cout << furthest << std::endl;
        return 0; 
    }

    ```