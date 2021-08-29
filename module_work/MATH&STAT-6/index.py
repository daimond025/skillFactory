import numpy as np
import cvxpy
from numpy import integer
from scipy.optimize import linprog

# values = [4, 2, 1, 7, 3, 6]
# weights = [5, 9, 8, 2, 6, 5]
# C = 15
# n = 6
#
# c = -1 * np.array(values)
# A = np.array(weights)
# A = np.expand_dims(A, 0)
# b = np.array([C])

# print(linprog(c=c, A_ub=A, b_ub=b))

# x = cvxpy.Variable(shape=n, integer =True)
# constraint = A @ x <= b
# x_positive = x >= 0
# total_value = c * x
# problem = cvxpy.Problem(cvxpy.Minimize(total_value), constraints=[constraint, x_positive])
# problem.solve()


c = np.array([
    [1000,12,10,19,8],
    [12,1000,3,7,2],
    [10,3,1000,6,20],
    [19,7,6,1000,4],
    [25,2,20,4,1000],
])
x = cvxpy.Variable(shape=(5, 5), boolean=True)

constraint = [cvxpy.sum(x[0]) == 1,
              cvxpy.sum(x[1]) == 1,
              cvxpy.sum(x[2]) == 1,
              cvxpy.sum(x[3]) == 1,
              cvxpy.sum(x[4]) == 1,
              cvxpy.sum(x[:, 0]) == 1,
              cvxpy.sum(x[:, 1]) == 1,
              cvxpy.sum(x[:, 2]) == 1,
              cvxpy.sum(x[:, 3]) == 1,
              cvxpy.sum(x[:, 4]) == 1]
total_value = cvxpy.sum(cvxpy.multiply(c, x))

problem = cvxpy.Problem(cvxpy.Minimize(total_value), constraints=constraint)
problem.solve(solver='ECOS_BB')
print(problem.solve(solver='ECOS_BB'))

