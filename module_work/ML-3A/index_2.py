import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
data = load_boston()


# def linreg_linear(X, y):
#     theta = np.linalg.inv(X.T @ X) @ X.T @ y
#     return theta
def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')
# X, y = data['data'], data['target']
#
# X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
#
# theta = linreg_linear(X, y)
#
# y_pred = X.dot(theta)
#
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
# theta = linreg_linear(X_train, y_train)
# y_pred = X_valid.dot(theta)
# y_train_pred = X_train.dot(theta)
#
#
# lr = LinearRegression()
# lr.fit(X,y)
# y_pred = lr.predict(X)


def calc_mse_gradient(X, y, theta):
    n = X.shape[0]
    grad = 1. / n * X.transpose().dot(X.dot(theta) - y)
    return grad

def gradient_step(theta, theta_grad, alpha):
    return theta - alpha * theta_grad


def optimize(X, y, grad_func, start_theta, alpha, n_iters):
    theta = start_theta.copy()

    for i in range(n_iters):
        theta_grad = grad_func(X, y, theta)
        theta = gradient_step(theta, theta_grad, alpha)
    return theta
X, y = data['data'], data['target']
X = (X - X.mean(axis=0)) / X.std(axis=0)

X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
m = X.shape[1]
theta = optimize(X, y, calc_mse_gradient, np.ones(m), 0.01, 5000)

y_pred = X.dot(theta)
print_regression_metrics(y, y_pred)
exit()
