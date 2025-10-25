class ListNode():
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

def build_ListNode(arr):
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next

def print_ListNode(head):
    cur = head
    while cur:
        print(cur.val, end = " ")
        cur = cur.next
    print()

arr = list(map(int, input().split()))
head = build_ListNode(arr)
print_ListNode(head)