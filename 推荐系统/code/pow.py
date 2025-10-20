def my_pow(x, n):
    if n == 0:
        return 1
    elif n < 0:
        return 1 / my_pow(x, -n)

    result = 1
    base = x
    exp = n
    while exp > 0:
        if exp % 2 == 1:
            result *= base
        base *= base
        exp //= 2
    return result

print(my_pow(2, 5))