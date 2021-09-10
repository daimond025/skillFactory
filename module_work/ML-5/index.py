from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree

data = pd.read_csv('./data/petrol_consumption.csv')

X = data.iloc[:, :-1].to_numpy()
Y = data.iloc[:, -1].to_numpy()

RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=RANDOM_SEED)
clf_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
clf_tree.fit(X_train, y_train)

y_pred = clf_tree.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(rmse)
# print(clf_tree.get_depth())

# data = pd.read_csv('./data/bill_authentication.csv')
#
# X = data.iloc[:, :-1].to_numpy()
# Y = data.iloc[:, -1].to_numpy()
#
# RANDOM_SEED = 17
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=17)
#
# clf_tree = DecisionTreeClassifier(max_depth=3, max_features=2, random_state=RANDOM_SEED,)
# clf_tree.fit(X_train, y_train)
# Y_predicted = clf_tree.predict(X_test)
# # print(f1_score(y_test,Y_predicted))
# print( clf_tree.predict([[2.04378,-0.38422,1.437292,0.76421]]))
