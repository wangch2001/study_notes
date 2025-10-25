def sqrt3_gradient_descent(
    x0=1.0,       # 初始猜测
    lr=0.01,      # 学习率 alpha
    max_steps=10000,
    tol=1e-12     # 收敛阈值
):
    x = x0
    for step in range(max_steps):
        grad = 4 * x * (x * x - 3)
        x_new = x - lr * grad
        loss = (x * x - 3) ** 2

        if step < 10 or step % 500 == 0:
            print(f"step {step:5d} | x = {x:.12f} | loss = {loss:.12e} | grad = {grad:.12e}")

        if abs(x_new - x) < tol:
            print(f"Converged at step {step}")
            x = x_new
            break
        x = x_new
    return x

ans = sqrt3_gradient_descent(
    x0=2.0,   # 初值给 2.0，接近 √3 ≈ 1.732...
    lr=0.05   # 学习率 0.05 在这个问题上挺稳
)

print("\nApprox sqrt(3) =", ans)
print("Check: ans^2    =", ans * ans)
