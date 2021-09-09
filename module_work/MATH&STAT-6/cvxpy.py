import cvxpy as cp
import numpy as np

m = 1000
n = 1000
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)




# Define and solve the CVXPY problem.
x = cp.Variable(shape=n)
print(A)
exit()
cost = cp.sum_squares(A @ x - b)
x_positive = (x >= 0)
prob = cp.Problem(cp.Minimize(cost), constraints=[x_positive ])
prob.solve()
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)