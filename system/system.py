def vanderpol(X, mu = 2):
    x, y = X[0], X[1]
    dx__dt = mu * (x - 1/3 * x**3 - y)
    dy__dt = 1/mu * x
    return [dx__dt, dy__dt]

def duffing_oscillator(X, f, gamma = 0.002, omega = 1, beta = 1, alpha = 1, delta = 0.3):
    x0, x1 = X[0], X[1]
    d_x0__dt = x1
    d_x1__dt = -gamma * x1 - alpha * x0 - beta * x0**3 + f
    return [d_x0__dt, d_x1__dt]
