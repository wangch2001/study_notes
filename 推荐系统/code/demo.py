def sqrt3(
        x = 1.0,
        lr = 0.01,
        max_steps = 10000,
        tol = 1e-12
):

    for i in range(max_steps):
        grad = 4 * x * (x ** 2 - 3)
        x_new = x - grad * lr


        if abs(x_new - x) < tol:
            return x_new
        x = x_new

ans = sqrt3(
    x=2.0,   # 初值给 2.0，接近 √3 ≈ 1.732...
    lr=0.05   # 学习率 0.05 在这个问题上挺稳
)

print("\nApprox sqrt(3) =", ans)
print("Check: ans^2    =", ans * ans)

