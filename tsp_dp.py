import numpy as np
import pandas as pd
import math
from tqdm import tqdm

NOW_WEIGHT = 557
# calculate the distance
df = pd.read_csv('./Appendix.csv')
df = np.array(df)
distance = np.zeros((df.shape[0], df.shape[0]))
need_list = []
for i in range(df.shape[0]):
    need_list.append(df[i, 3])
need_list[8] = 0
for i in range(df.shape[0]):
    for j in range(df.shape[0]):
        jing1 = df[i, 1]
        jing2 = df[j, 1]
        d_wei = (df[i, 2] - df[j, 2])
        distance[i, j] = 2 * math.asin(math.sqrt(
            pow(math.sin(d_wei / 2), 2) + math.cos(jing1) * math.cos(jing2) * pow(math.sin(jing1 - jing2),
                                                                                  2))) * 6378.137

INF = 0x3f3f3f3f

# idi[i] represents the corresponding point in the storage structure
idi = [0] * 17
# vb[i] indicates the required material quantity of the ith point in the storage structure
vb = [30, 30, 30, 40, 55, 55, 40, 32, 0, 40, 50, 30, 30, 30, 30, 35]


def tsp(start, choose, motion):
    # initialize the information for tsp
    N = len(choose)
    M = 1 << (N - 1)
    V = 0
    for i in range(N):
        V += vb[choose[i]]

    for i in range(N):
        idi[i] = choose[i]
    for i in range(N):
        if choose[i] == start:
            idi[i] = idi[0]  # set zero point as the start point
            idi[0] = start

    g = []  # distance in tsp
    v = []  # required material in tsp

    for i in range(N):
        g.append([])
        for j in range(N):
            g[i].append(distance[idi[i]][idi[j]])
    for i in range(N):
        v.append(vb[idi[i]])
    v[0] = 0
    V -= vb[start]

    dp = np.zeros((N, M, V + 1))
    path = np.zeros(N)

    for i in range(N):
        for k in range(V):
            dp[i][0][k] = g[i][0] * k

    # dp
    for j in tqdm(range(1, M)):
        for i in range(N):
            for rest in range(V + 1):
                dp[i][j][rest] = INF
                if i != 0:
                    if (j >> (i - 1)) & 1 == 1:
                        continue
                for k in range(1, N):
                    if ((j >> (k - 1)) & 1) == 0:
                        continue
                    dp[i][j][rest] = min(dp[i][j][rest], g[i][k] * rest + dp[k][j ^ (1 << (k - 1))][rest - v[k]])

    if motion == 0:
        return [dp[0][M - 1][V], start]

    print("min:")
    print(dp[0][M - 1][V])

    visited = np.zeros(N)
    pioneer = 0
    mn = INF
    S = M - 1
    path[0] = 0
    rest = V
    cnt = 1

    # record the path
    while 1:
        if cnt == N - 1:
            for i in range(1, N):
                if visited[i] == 0:
                    path[cnt] = i
            break

        temp = -1
        for i in range(1, N):
            if (visited[i] == 0) and ((S & (1 << (i - 1))) != 0):
                p = g[pioneer][i] * rest + dp[i][(S ^ (1 << (i - 1)))][rest - v[i]]
                if mn > p:
                    mn = p
                    temp = i
        pioneer = temp
        path[cnt] = pioneer
        visited[pioneer] = 1
        S = S ^ (1 << (pioneer - 1))
        rest -= v[pioneer]
        mn = INF
        cnt += 1

    print("path:")
    print(start, end="->")
    for i in range(1, N):
        print(idi[int(path[i])], end="->")
    print(start)

    return [dp[0][M - 1][V], start]


def main():
    # final_ans = []
    # ans0 = tsp(8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [0, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans0:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [1, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans1:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [2, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans2:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [3, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans3:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    # # ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # # tsp(8, ls, 1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [4, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans4:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [5, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans5:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [6, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans6:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [7, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans7:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [8, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans8:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [9, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans9:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0)[0]
    # ans1 = INF
    # pos = -1
    # list = [10, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 0)
    # print("---")
    # print("ans10:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)
    #
    # cnt = 0
    # for tt in final_ans:
    #     print(cnt, ":", tt)
    #     cnt += 1

    # ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10], 1)[0]
    # ans1 = INF
    # pos = -1
    # list = [7, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 1)
    # print("---")
    # print("ans7:", ans0 + ans1)
    # final_ans.append(ans0 + ans1)

    # ans0 = tsp(8, [0, 3, 4, 5, 6, 7, 8, 9, 10], 1)[0]
    # ans1 = INF
    # pos = -1
    # list = [1, 2, 11, 12, 13, 14, 15]
    # for i in list:
    #     xx = tsp(i, list, 0)
    #     if ans1 > xx[0] or pos == -1:
    #         ans1 = xx[0]
    #         pos = xx[1]
    # tsp(pos, list, 1)
    # print("---")
    # print("ans1_2:", ans0 + ans1)

    # tsp(start,set of chosen points,motion)
    # motion 1 means show answer and 0 means not show answer
    ans0 = tsp(8, [0, 1, 2, 3, 4, 5, 6, 7, 8], 1)[0]
    ans1 = tsp(8, [8, 9, 10, 11, 12, 13, 14, 15], 1)[0]
    print("ans_two:", ans0 + ans1)

    return


if __name__ == '__main__':
    main()
