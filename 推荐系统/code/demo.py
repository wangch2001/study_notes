from collections import deque


class TreeNode():
    def __init__(self, val=0, left=None, right=None):
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
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] != "null":
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1
    return root


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

values = input().split()
root = build_tree(values)
print_tree(root)