---
layout: post
title: "배낭 채우기 문제"
subtitle: "Dynamic programming - Knapsack problem"
date: 2022-04-03 10:02:06 +0800
categories: [algorithm]
---

# Dynamic Programming - 0-1 배낭채우기 문제

*다이나믹 프로그래밍이란: 문제에 대한 해결의 알고리즘 설계 기법 중의 하나. 순차적으로 된 의사 결정의 최적화 문제를 정식화함으로써 얻어지는 문제 취급 이론 및 수법 이라고 볼수있다.*

***

코딩문제를 하면서 DFS 나 BFS, backtrace 등등 에 관한 문제들은 이해가 너무 힘들진 않았지만 유독 DP 에 관한 문제라면 조금 풀기 어려웠던것 같아서 DP 에서 가장 기본적인문제, 0-1 배낭 채우기 문제에 대해서 적어보려 한다.

## 문제 해석

>0-1 배낭채우기 문제란 n 개 보석이 있는데 이 n 개 보석들은 각각 부동한 무게와 부동한 가격들을 가지는데 이보석들을 최대용량이 W 인 배낭에 넣어 이 배낭속에 넣은 보석들이 최고의 가격을 가지게 만드는 문제다.

## 문제 분석

* 배낭문제를 처음 봤을땐 Greedy 알고리즘을 선택할수 있는데 Greedy 알고리즘은 이 문제에서 최악의 경우와 최적의 경우 들이 모두 나타날수 있다.

* 진일보로 생각하면 일상에서 진짜 보석을 사는것처럼 킬로그램당 가격을 계산하여 이가격이 높은순으로 배낭속에 넣는것도 가능한 문제풀이 방법이 될듯한데 이것 역시 풀리지가 않았다.

    ### 그럼 우리가 이러한 Greedy 알고리즘 대신 모든 경우들을 enum 해주는 enumeration method 를 선택하여 보자.

    * 보석을 배낭속에 넣는 모든 경우들을 몽땅 한번씩 계산해주는 식이였는데 이 모든 가능성중 무조건 한가지가 최적의 답안이 될것이다.

        ```cpp
        #include <bits/stdc++.h>

        int main(int argc, char* argv[]) {
            int max, n;
            std::cin >> max >> n;
            int list[31];
            memset(list, 0, sizeof(list));
            int ans = max;
            for(int i = 0; i < n; i++) {
                std::cin >> list[i];
            }
            for(int i = 0; i < (1 << n); i++) {
                int cur = 0;
                for(int j = 0; j < n; j++) {
                    if(i & (1 << j)) {
                        cur += list[j];
                    }
                }
                if(cur > max) continue;
                ans = std::min(ans, max - cur);
            }
            std::cout << ans << std::endl;
            return 0;
        }
        ```

    * 보석의 개수가 20개보다 적을땐 시간제한내에 모두 정확한 답안을 얻었지만 보석개수가 더 많은 경우에서는 결국 time limit 를 벗어났다.

***

## 정확한 풀이

* 여기서 우리는 dp 어레이를 리용하게 된다.

* int dp[n][W] 이렇게 만들어 주는데 이 어레이의 함의는 이렇다:

    > dp[i][j] 는 보석 리스트에서 앞의 i 개 보석을 용량이 j 인 배낭에 넣었을때 최적의 답안을 기록한다.
    
    > 이러면 우리가 새로운 경우의 최적의 답안을 찾으려면 그 전의 최적의 답안을 리용하여 현재의 보석을 넣을것인가 아닌가 를 결정할수 있다.

    > 이게 무슨말이냐면 ... 코드로 알아보자:

    ```cpp
    #include <bits/stdc++.h>

    int main(int argc, char* argv[]) {
        int V, n;
        std::cin >> V >> n;
        int weight[105], value[105];
        int dp[105][1005];
        memset(weight, 0, sizeof(weight));
        memset(value, 0, sizeof(value));
        memset(dp, 0, sizeof(dp));
        for(int i = 1; i <= n; i++) {
            std::cin >> weight[i] >> value[i];
        }
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= V; j++) {
                if(weight[i] > j) dp[i][j] = dp[i - 1][j]; // 현재 배낭용량을 벗어남.
                else dp[i][j] = std::max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
            }
        }
        std::cout << dp[n][V] << std::endl;
        return 0;
    }
    ```

* 이 알고리즘이 최적의 답안을 찾을수 있는것은 k번째 계산에서는 항상 k - 1번째의 최적의 답안의 기초하에서 계산해주는 방식이다. 이 배낭문제에서는 이렇게 표시할수 있다: `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])` 보다싶이 용량이 j인 배낭에 i번째 물건을 넣을때 최적의 답안은 i - 1번째의 답안에 기초한것이다. dp[i - 1][j] 와 dp[i - 1][j - weight[i]] + value[i] 에서 최대값을 선택하는것은 dp[i - 1][j] 는 현재 i번째 물건을 넣지 않을때의 가격 과 이번의 i번째 물건을 넣을때 배낭을 i번째 물건만큼의 중량을 비워낸 최적의 가격과 i번째 물건의 값을 더한 가격에서 최대값을 선택하여 최적화의 답안을 찾아주는것이다.

* 이중 for 순환에 i번째 물건이 현재의 배낭크기를 넘쳐나는지도 검사해줘야한다.

*** 
설명이 간단했지만 직접 코딩을 하면 인차 이해가 될것이다.
