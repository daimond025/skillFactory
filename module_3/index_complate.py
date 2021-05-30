import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

pd.set_option('display.max_rows', None)

date = pd.read_csv('main_task.csv')
date['Number of Reviews'].fillna(date['Number of Reviews'].median(), inplace=True)

y = date['Rating'].copy()
X = date.drop(['Rating'], axis=1).copy()



# date = pd.read_csv('main_task.xls')
# date['Number of Reviews'].fillna(date['Number of Reviews'].median(), inplace=True)
# X = date[['Number of Reviews', 'Ranking']].copy()
# y = date['Rating'].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
# округление
y_pred = np.round(y_pred,1)


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

