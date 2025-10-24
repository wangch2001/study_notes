def sqrt3_gradient_descent(
    x0=1.0,       # 初始猜测
    lr=0.01,      # 学习率 alpha
    max_steps=10000,
    tol=1e-12     # 收敛阈值
):
    x = x0
    for step in range(max_steps):
        # 1. 计算梯度 dL/dx = 4x(x^2 - 3)
        grad = 4 * x * (x * x - 3)

        # 2. 参数更新 x <- x - lr * grad
        x_new = x - lr * grad

        # 3. 看看 loss 有多大
        loss = (x * x - 3) ** 2

        # 4. 打印前几步 & 最后一步，方便看收敛
        if step < 10 or step % 500 == 0:
            print(f"step {step:5d} | x = {x:.12f} | loss = {loss:.12e} | grad = {grad:.12e}")

        # 5. 收敛判断：如果变化已经非常小就停
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
