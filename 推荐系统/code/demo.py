import random
nums = [1, 2, 3, 4, 5, 6, 7]
k = 6 # 4 = nums[5 - 2]

def partition(nums, low, high):
    i = random.randint(low, high)
    pivot = nums[i]
    nums[low], nums[i] = nums[i], nums[low]
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


def find_k(nums, k):
    n = len(nums)
    low = 0
    high = n - 1
    while low < high:
        pi = partition(nums, low, high)
        if pi == n - k:
            return nums[pi]
        elif pi > n - k:
            high = pi - 1
        elif pi < n - k:
            low = pi + 1

print(find_k(nums, k))

