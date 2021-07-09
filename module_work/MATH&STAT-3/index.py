import math
import numpy as np


def rosen(x):
# Функция Розенброка
    print(x[:-1])
    print(x[1:])
    exit()
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Настраиваем 3D график
fig = plt.figure(figsize=[15, 10])
ax = fig.gca(projection='3d')

# Задаем угол обзора
ax.view_init(45, 30)

# Создаем данные для графика
X = np.arange(-2, 2, 0.5)
Y = np.arange(-1, 3, 0.5)

X, Y = np.meshgrid(X, Y)
Z = rosen(np.array([X,Y]))

# Рисуем поверхность
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()

f = np.array([8, 2, 8, 3, 5, 6, 5, 15])
print(np.gradient(f))
def sigmoid(x):
    return 1 / (1 + math.e ** -2*(0.1+0.3+0.6))
