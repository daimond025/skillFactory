
import  pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

date = pd.read_csv('main_task.xls')

X = date.drop(['Restaurant_id', 'Rating'], axis=1)
y = date['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

