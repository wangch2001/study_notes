import bisect

arr = [1, 2, 3, 4, 5, 7]

pos = bisect.bisect_left(arr, 3)
print(pos)
print(arr[pos])

pos2 = bisect.bisect_right(arr, 3)
print(pos2)
print(arr[pos2])