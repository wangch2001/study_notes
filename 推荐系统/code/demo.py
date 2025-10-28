import random

def partition(nums, low, high):
    i = random.randint(low, high)
    pivot = nums[i]
    nums[i], pivot = pivot, nums[i]
    i = low + 1
    j = high
    while True:
        while i <= j and nums[i] < pivot:
            i += 1
        while i <= j and nums[j] > pivot:
            j -= 1
        if i >= j:
            break
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1
    nums[low], nums[j] = nums[j], nums[low]
    return j

def quicksort(nums, low, high):
    if low < high:
        pi = partition(nums, low, high)
        quicksort(nums, low, pi - 1)
        quicksort(nums, pi + 1, high)

# 测试
nums = [3,6,8,10,1,2,1]
print("排序前:", nums)
quicksort(nums, 0, len(nums)-1)
print("排序后:", nums)