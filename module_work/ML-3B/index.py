from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def logloss(y_true, y_pred ):
    return  - y_true * np.log(y_pred) - ( 1 - y_true) * np.log( 1 - y_pred)
def mult_logloss(y_true, y_pred):
    n, m = y_true.shape[0], y_true.shape[1]
    sum_ = []
    for i in range(n):
        for j in range(m):
            if y_true[i][j] != 0 and y_pred[i][j] != 0:
                sum_.append(y_true[i][j] * np.log(y_pred[i][j]))
    sum_ = np.nansum(sum_)
    loss = -(sum_ / n)
    return loss

y_pred = np.array([[0.2, 0.3, 0.5], [0, 0, 1], [0.1, 0, 0.9]])
y_true = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])

print(mult_logloss(y_true, y_pred))
exit()

import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
# print(data.corr()['price_range'].sort_values())
data = data[['touch_screen','px_height', 'px_width', 'battery_power', 'ram', 'price_range']].copy()

X = data.iloc[:, :-1].to_numpy()
Y = data.iloc[:, -1].to_numpy()


X_train, X_val, Y_train, Y_val = train_test_split(X, Y,random_state=31,  test_size = 0.2)
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_predicted = model.predict(X_val)

print(model.coef_)
print(accuracy_score(Y_val,Y_predicted))
print(precision_score(Y_val,Y_predicted))
print(recall_score(Y_val,Y_predicted))
print(f1_score(Y_val,Y_predicted))
