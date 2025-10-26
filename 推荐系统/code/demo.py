def sqrt3(
        x0 = 1.0,
        lr = 0.01,
        max_steps = 10000,
        tol = 1e-12
):
    x = x0
    for step in range(max_steps):
        grad = 4 * x * (x ** 2 - 3)
        x_new = x - lr * grad
        loss = (x * x - 3) ** 2

        if step < 10 or step % 500 == 0:
            print(f"step:{step} | x = {x:.12f} | loss = {loss:.12e} | grad = {grad:.12e}")
        if abs(x_new - x) < tol:
            print(f"Converged at step {step}")
            x = x_new
            break
        x = x_new
    return x

print(sqrt3())