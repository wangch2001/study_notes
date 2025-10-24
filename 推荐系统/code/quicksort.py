import random


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



# 原地版快速排序（更接近算法教科书）
def partition(arr, low, high):
    i = random.randint(low, high)
    pivot = arr[i]
    arr[i], arr[low] = arr[low], arr[i]
    i = low + 1
    j = high
    while True:
        while i <= j and arr[i] < pivot:
            i += 1
        while i <= j and arr[j] > pivot:
            j -= 1
        if i >= j:
            break
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1
    nums[low], nums[j] = nums[j], nums[low]
    return j


def quicksort_inplace(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort_inplace(arr, low, pi-1)
        quicksort_inplace(arr, pi+1, high)

# 测试
nums = [3,6,8,10,1,2,1]
print("排序前:", nums)
quicksort_inplace(nums, 0, len(nums)-1)
print("排序后:", nums)