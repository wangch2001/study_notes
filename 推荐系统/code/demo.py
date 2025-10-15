def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = quicksort([x for x in arr if x < pivot])
    mid = [x for x in arr if x == pivot]
    right = quicksort([x for x in arr if x > pivot])
    return left + mid + right

num = [3,6,1,5,7]
print(num)
print(quicksort(num))