from scipy import optimize
import numpy as np
from numpy.random import rand
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance_matrix

from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

# 7.3
x0 = np.array([0])
def function_to_minimize(x):
    return np.cos(x[0]**5) - x[0]**4 + x[0]**3 - np.sin(x[0]**2)

bounds = Bounds(5,42)
res = minimize(function_to_minimize, x0, method='bfgs', bounds=bounds, options={ 'verbose': 1})
print(res)
exit()

# bounds = Bounds ([-10, 0], [10, 100.0])
# linear_constraint = LinearConstraint ([[1, 1]], [42], [42])
#
# def func(x):
#     return x[0]**3 - 5*x[1]**2 -5
#
#
# x0 = np.array([0.5, 0])
# res = minimize(func, x0, method='trust-constr', constraints=[linear_constraint],
#         options={'verbose': 1}, bounds=bounds)
# print(res)

# def function_to_minimize(x):
#     return x[0]**3 + 4 * x[0]**2 + 10 + 30*x[1]**2
#
# def linear(x):
#     return x[0]
#
# x0 = np.array([1, 3])
#
# res = minimize(linear, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
# res = minimize(linear, x0, method='bfgs', options={'xatol': 1e-8, 'disp': True})

