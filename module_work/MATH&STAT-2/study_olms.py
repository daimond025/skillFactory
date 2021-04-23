import numpy as np # для работы с массивами
import pandas as pd # для работы с DataFrame
from sklearn import datasets # для импорта данных
import seaborn as sns # библиотека для визуализации статистических данных
import matplotlib.pyplot as plt # для построения графиков


# x = np.arange(1,11)
# y = 2 * x + np.random.randn(10)*2
# X = np.vstack((x,y))
#
# Xcentered = (X[0] - x.mean(), X[1] - y.mean())
# m = (x.mean(), y.mean())
#
# covmat = np.cov(Xcentered)
# _, vecs = np.linalg.eig(covmat)
# v = -vecs[:,1]
#
# Xnew = v@Xcentered

X = np.array([[1,2 , 1,1]])
Y = np.array([70,130 , 65,60])

X_centre = X - X.mean()
Y_centre = Y - Y.mean()

X_st = X_centre/np.linalg.norm(X_centre)
Y_st = Y_centre/np.linalg.norm(Y_centre)

A = np.array([[1, 0.9922],[0.9922, 1]])
c = np.array([[1, 0.9922],[0.9922, 1]])
lam, vecs = np.linalg.eig(A)



x_new = (X_st * np.linalg.eig(c)[1][0,0]+Y_st*np.linalg.eig(c)[1][1,0])
# x_new /= np.linalg.norm(x_new)
c = 0.70710678  * X_st +0.70710678  * Y_st

c = c/np.linalg.norm(c)

print(x_new)
print(c)
exit()







