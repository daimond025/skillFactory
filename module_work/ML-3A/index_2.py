import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
data = load_boston()

def linreg_linear(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta
X, y = data['data'], data['target']

X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
print([np.ones(X.shape[0])[:, np.newaxis], X])
# print(np.ones(X.shape[0])[:, np.newaxis])