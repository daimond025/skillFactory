import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

from flask import Flask, request

app = Flask(__name__)

model_ = None

# генерация +  сохранение модели / перед запуском
def model_save():
    X, y = load_diabetes(return_X_y=True)
    X = X[:, 0].reshape(-1, 1)  # Берём только один признак
    regressor = LinearRegression()
    regressor.fit(X, y)

    with open('./myfile.pkl', 'wb') as output:
        pickle.dump(regressor, output)  # Сохраняем


# чтение модели из файла / после запуска сервера
def model_read():
    with open('./myfile.pkl', 'rb') as pkl_file:
        return pickle.load(pkl_file)  # Загружаем


def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


def model_predict(value):
    print(isfloat(value))
    if isfloat(value):
        value = float(value)
    elif isint(value):
        value = int(value)
    else:
        return None

    value = np.array([value]).reshape(-1, 1)
    predict = model_.predict(value)
    return predict[0]


@app.route('/predict')
def hello_func():
    value = request.args.get('value')
    prediction = model_predict(value)
    if prediction is None:
        return f'value is not float type !'
    else:
        return f'the result is {prediction}!'


if __name__ == '__main__':
    model_save()
    model_ = model_read()
    app.run('localhost', 5000)
