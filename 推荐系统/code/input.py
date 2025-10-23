# 基本输入
# 01 单个整数
n = int(input())

# 02 单行多个整数
a, b, c = map(int, input().split())

# 03 数组（单行）
arr = list(map(int, input().split()))

# 04 多行数组/矩阵
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]

# 05 多行单数
n = int(input())
arr = [int(input()) for _ in range(n)]

# 06 不定行输入（知道EOF）：
import sys
for line in sys.stdin:
    nums = list(map(int, line.split()))

# 07 快速输入（大数据量）
import sys
input = sys.stdin.readline()
n = int(input.strip())

# 输出
# 01 输出单个数/单个变量
print(n)

# 02 输出多个数（同一行，空格分隔）
a, b, c = 1, 2, 3
print(a, b, c)

# 03 输出数组（空格分隔）
arr = [1, 2, 3, 4]
print(*arr)

# 04 输出数组（换行分隔）
arr = [1, 2, 3, 4]
for x in arr:
    print(x)

# 05 输出矩阵
matrix = [[1, 2, 3], [4, 5, 6]]
for row in matrix:
    print(*row)

# 06 格式化输出
name, score = "Alice", 95
print(f"{name} scored: {score}")
print("{} scored {}".format(name, score))

# 07 指定小数位
pi = 3.1415926535
print(f"{pi:.2f}")

# 08 不换行输出
print("hello", end = "")
print("word")

# 09 输出到同一行（循环）
for i  in range(5):
    print(i, end = "")

# 10 输出布尔/特殊要求
print("Yes" if True else "No")


# 链表
# 常见输入：一行数字，表示链表的节点值
# 1 2 3 4 5
class ListNode:
    def __init__(self, val = 0, nxt = None):
        self.val = val
        self.next = nxt

def build_linked_list(self):
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next

arr = list(map(int, input().split()))
head = build_linked_list()

# 输出链表：
def print_linked_list(head):
    cur = head
    while cur:
        print(cur.val, end = " ")
        cur = cur.next
    print()


# 二叉树（层序输入）：
# 常见输入：空节点用null或None表示
# 1 2 3 null 4 null 5

from collections import deque
class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(values):
    if not values or values[0] == "null":
        return None
    root = TreeNode(int(values[0]))
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if values[i] != "null":
            node.left = TreeNode(int(values[i]))
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] != "null":
            node.right = TreeNode(int(values[i]))
            q.append(node.right)
        i += 1
    return root

values = input().split()
root = build_tree(values)

# 输出二叉树（层序遍历）
from collections import deque
def print_tree(root):
    if not root:
        print("None")
        return
    q = deque([root])
    res = []
    while q:
        node = q.popleft()
        if node:
            res.append(str(node.val))
            q.append(node.left)
            q.append(node.right)
        else:
            res.append("null")
    print(" ".join(res))





# 图（邻接表）
# 4 4
# 1 2
# 2 3
n, m = map(int, input.split())
graph = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# 图（邻接矩阵）
# 3
# 0 1 0
# 1 0 1
# 0 1 0
n = int(input())
matrix = [list(map(int, input.split())) for _ in range(n)]

# 13 输出图(邻接表)
for u in range(1, n + 1):
    print(u, ":", *graph[u])










