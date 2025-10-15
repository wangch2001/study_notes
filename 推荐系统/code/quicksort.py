
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = quicksort([x for x in arr if x < pivot])
    mid = [x for x in arr if x == pivot]
    right = quicksort([x for x in arr if x > pivot])
    return left + mid + right

nums = [3,6,8,10,1,2,1]
print(nums)
print(quicksort(nums))