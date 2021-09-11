from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from pprint import pprint

weather=pd.read_csv('./data/temps_extended.csv')
y = weather['actual']
X = weather.drop(['actual','weekday','month','day','year'],axis =1)

X_train, X_val, Y_train, Y_val=train_test_split(X,y,test_size=0.3, random_state=42)


rf = RandomForestRegressor(bootstrap=True, max_depth= 10, max_features= 'sqrt', min_samples_leaf=2,
                            min_samples_split=5, n_estimators=1000,  random_state=42)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_val)
print(mean_squared_error(Y_val, y_pred))
print(np.sqrt(mean_squared_error(Y_val, y_pred)))
exit()

rf = RandomForestRegressor(random_state=42)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
                               cv=3, verbose=2, random_state=42, n_jobs=10)
rf_random.fit(X_train, Y_train)
pprint(rf_random.best_params_)

#TODO 5.5.2
# data = pd.read_csv('./data/winequality-red.csv', sep=';')
# data['quality'] = (data.quality >= 6).astype(int)
#
# X = data.drop('quality', axis=1)
# Y = data['quality']
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# log_reg = LogisticRegression()
# dec_tree = DecisionTreeClassifier(random_state=42, max_depth=10)
#
# log_reg.fit(X_train, y_train)
# dec_tree.fit(X_train, y_train)
#
# f1_score_reg = f1_score(y_test, log_reg.predict(X_test))
# f1_score_tree = f1_score(y_test, dec_tree.predict(X_test))
# print(f1_score_reg)
# print(f1_score_tree)
#
# bag_clf = BaggingClassifier(dec_tree, n_estimators=1500, random_state=42)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# f1_bagging = f1_score(y_test, y_pred)
# print(f1_bagging)
# exit()
#

# data = pd.read_csv('./data/petrol_consumption.csv')
#
# X = data.iloc[:, :-1].to_numpy()
# Y = data.iloc[:, -1].to_numpy()
#
# RANDOM_SEED = 42
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=RANDOM_SEED)
# clf_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
# clf_tree.fit(X_train, y_train)
# y_pred = clf_tree.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
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
