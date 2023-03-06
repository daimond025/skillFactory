from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
c
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt



iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_2d = X[:, :2]
X_2d = X_2d[y > 0]

y_2d = y[y > 0]
y_2d -= 1

# C_range = np.logspace(-2, 10, 13)
# param_grid = dict(C=C_range)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(C=C_range, gamma=gamma_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
model = SVC(random_state=42)
clf = GridSearchCV(model, param_grid=param_grid, cv=cv)
clf.fit(X_2d, y_2d)
print(clf.best_params_)
print(clf.best_score_)
exit()

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))


plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # оценим функцию принятия решения в сетке
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # визуализируем функцию принятия решения для этих параметров
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")

    # визуализируем влияние параметра на функцию принятия решения
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis("tight")
plt.show(block=False)

# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# model = LinearSVC(random_state=42)
# clf = GridSearchCV(model, param_grid=param_grid, cv=cv)
# clf.fit(X, y)
# print(clf.best_params_)
# print(clf.best_score_)

# C_2d_range = [1e-2, 1, 1e2]
# classifiers = []
# for C in C_2d_range:
#     clf = LinearSVC(C=C)
#     clf.fit(X_2d, y_2d)
#     classifiers.append((C, clf))
#
# plt.figure(figsize=(3, 6))
# xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
# for (k, (C, clf)) in enumerate(classifiers):
#     # оценим функцию принятия решения в сетке
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     # визуализируем функцию принятия решения для этих параметров
#     plt.subplot(len(C_2d_range), 1, k + 1)
#     plt.title(" C=10^%d" % (np.log10(C)), size="medium")
#
#     # визуализируем влияние параметра на функцию принятия решения
#     plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
#     plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
#     plt.xticks(())
#     plt.yticks(())
#     plt.axis("tight")
# plt.show(block=False)
exit()

