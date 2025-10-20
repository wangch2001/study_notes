n1, m = map(int, input().split())
graph = [[] for _ in range(n1 + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# 图（邻接矩阵）
n2 = int(input())
matrix = [list(map(int, input().split())) for _ in range(n2)]

# 输出图（邻接表）
print("邻接表表示：")
for u in range(1, n1 + 1):
    print(u, ":", *graph[u])

# 输出图（邻接矩阵）
print("\n邻接矩阵表示：")
for row in matrix:
    print(*row)