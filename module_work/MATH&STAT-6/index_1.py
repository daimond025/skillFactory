def f(x, y, z):
    return x **3 - 2 * x ** 2 + y ** 2 + z ** 2 - 2 * x * y + x * z  - y * z  + 3 * z
def grad(x, y):
    dx = 4 * x - 4 * y
    dy = - 4 * x  + 4 * y **3
    return (dx, dy)
def dist(x1, x2):
    return (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2

from scipy import optimize
def f_for_scipy(x):
    return f(x[0], x[1], x[2])
print( optimize.minimize(f_for_scipy, x0=(0, 0, 0)))
exit()

x0 = (0, 2)
gamma = 1e-4
x_cur = x0


vals = []
coords = []
i = 0


while True:
    x_new = (x_cur[0] - gamma * grad(*x_cur)[0],
            x_cur[1] - gamma * grad(*x_cur)[1])
    # if dist(x_new, x_cur) < 1e-9:
    #     break
    x_cur = x_new
    vals.append(f(*x_cur))
    coords.append(x_cur)
    i += 1
    print(f"iter={i}; x=({x_cur[0]:.9f}, {x_cur[1]:.9f});"
          f" f(x)={f(*x_cur):.9f}; grad f(x)=({grad(*x_cur)[0]:.9f}, {grad(*x_cur)[1]:.9f})")