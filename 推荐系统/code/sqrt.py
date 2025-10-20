def sqrt_newton(x, tolerance=1e-10):
    if x < 0:
        raise ValueError("不能对负数开平方")
    guess = x / 2.0
    while abs(guess * guess - x) > tolerance:
        guess = (guess + x / guess) / 2
    return guess

print(sqrt_newton(16))  # 输出约为 4.0