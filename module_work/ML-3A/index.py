import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('data_flats.csv', sep =';')
flats2 = data.drop(['id', 'life_sq','kindergarten_km', 'park_km', 'kremlin_km', 'preschool_education_centers_raion'], axis=1)
flats2.dropna(inplace=True)

flats2['price_doc'] = flats2['price_doc'].apply(lambda w: np.log(w + 1))
X = flats2.iloc[:, :-1]
Y = flats2.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=77)

scaler = RobustScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

myModel = LinearRegression()
myModel.fit(X_train_transformed,Y_train)
y_pred = myModel.predict(X_test_transformed)

MSE  = metrics.mean_squared_error( np.exp(Y_test) - 1, np.exp(y_pred) - 1)
print(round(MSE))

# y_happy = [2,3,-1,4]
# y_happy_pred = [1,3,2,5]
# MAE = metrics.mean_absolute_error(y_happy, y_happy_pred)
# MSE  = metrics.mean_squared_error(y_happy, y_happy_pred)
R_2 = metrics.r2_score(y_happy, y_happy_pred)
# print(MAE)
# print(MSE )
# print(R_2 )

# myData = pd.read_csv('mycar.csv', sep=',')
# myData = myData['Speed,Stopping_dist'].str.split(',',expand=True)
# myData.columns=['Speed','Stopping_dist']
#
# X = myData['Speed'].astype('int32').to_numpy()
# Y = myData['Stopping_dist'].astype('int32').to_numpy()
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
#
# x_train= X_train.reshape(-1, 1)
# x_test= X_test.reshape(-1, 1)
# y_train= Y_train.reshape(-1, 1)
# y_test= Y_test.reshape(-1, 1)
#
# myModel = LinearRegression()
# myModel.fit(x_train,y_train)
#
# y_pred = myModel.predict(x_test)
# print(myModel.coef_)
# print(myModel.intercept_)
# print(y_pred)
# exit()




