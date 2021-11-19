import tensorflow as tf  # пока что используем этот пакет только для скачки данных :)
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
import sklearn
import matplotlib.pyplot as plt

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x_train_flat = x_train.reshape(-1, 28 * 28).astype(float)
x_val_flat = x_val.reshape(-1, 28 * 28).astype(float)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_flat = scaler.fit_transform(x_train_flat)
x_val_flat = scaler.transform(x_val_flat)

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
# clf.fit(x_train_flat, y_train)

# x_train_flat_pre = clf.predict_proba(x_train_flat)
# print(y_train[1])
# print(x_train_flat_pre[1])
# print(clf.coef_.shape)
# print('#0 ' + str(x_train_flat[1] @ clf.coef_[0]))  #0
# print('#1 ' + str(x_train_flat[1] @ clf.coef_[1]))  #1
# print('#2 ' + str(x_train_flat[1] @ clf.coef_[2]))  #2
# print('#3 ' + str(x_train_flat[1] @ clf.coef_[3]) ) #3
# print('#4 ' + str(x_train_flat[1] @ clf.coef_[4])  )#4
# print('#5 ' + str(x_train_flat[1] @ clf.coef_[5]))  #5
# print('#6 ' + str(x_train_flat[1] @ clf.coef_[6]))  #6
# print('#7 ' + str(x_train_flat[1] @ clf.coef_[7]))  #7
# print('#8 ' + str(x_train_flat[1] @ clf.coef_[8]))  #8
# print('#9 ' + str(x_train_flat[1] @ clf.coef_[9]) ) #9

# # не так плохо работает!
from sklearn.metrics import accuracy_score
# accuracy_score(y_val, clf.predict(x_val_flat))
# print(accuracy_score(y_val, clf.predict(x_val_flat)))

from sklearn.neural_network import MLPClassifier  # многослойный персептрон (просто много полносвязных слоев)
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

# random_grid = {
#                 'activation': ['identity', 'logistic', 'tanh', 'relu'],
#                 'learning_rate': ['constant', 'invscaling', 'adaptive'],
#                'max_iter': [int(x) for x in np.linspace(start=20, stop=50, num=5)],
#                'alpha': [0.0001, 0.0002, 0.0003, 0.0005],
#                'beta_1': [0.88, 0.89, 0.90, 0.91, 0.92],
#                'beta_2': [0.999, 0.998, 0.997, 0.9999],
#                'hidden_layer_sizes': [(200,), (150,), (250,), (300,)  ],
#                }
#
# model = MLPClassifier(random_state=0)
# etr_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3,
#                                 verbose=10, random_state=0, n_jobs=3)
# etr_random.fit(x_train_flat, y_train)
# print(etr_random.best_params_)
# exit()

clf = MLPClassifier(max_iter=50, learning_rate='constant', random_state=0,
                    hidden_layer_sizes=(300,),beta_2= 0.999, alpha=0.0003, beta_1=0.89, activation='relu' )
clf.fit(x_train_flat, y_train)
print(accuracy_score(y_val, clf.predict(x_val_flat)))
