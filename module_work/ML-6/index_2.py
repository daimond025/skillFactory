import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./data/spam7.csv')
mydict = {
    "y": 1,
    "n": 0,
}

df['spam'] = df['yesno'].apply(mydict.get)
df.drop(['yesno', 'Unnamed: 0'], axis=1, inplace=True)

X = df.drop(['spam'], axis=1)
y = df['spam']

coll = X.columns

pf = PolynomialFeatures(interaction_only=True, include_bias=False)
poly_data = pf.fit_transform(X)
poly_cols = pf.get_feature_names(X.columns)
poly_cols = [x.replace(' ', '_') for x in poly_cols]

poly_X = pd.DataFrame(poly_data, columns=poly_cols)


random_state=42
X_train, X_test, y_train, y_test = train_test_split(poly_X, y, test_size=0.20, random_state=random_state)

# model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2,
#                     min_samples_leaf=1, subsample=1,max_features=None, random_state=random_state)
# model.fit(X_train, y_train)
# Y_predicted = model.predict(X_test)
# print(accuracy_score(y_test,Y_predicted))


# imp_f = pd.Series(model.feature_importances_)
# imp_f.index = poly_cols
# imp_f.sort_values(inplace = True)
# imp_f.plot(kind = 'barh')



# TODO GridSearchCV
#
# param_grid = {'learning_rate':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
#               'n_estimators':[100, 250, 500, 750, 1000, 1250, 1500, 1750]}
# model =  GradientBoostingClassifier(random_state=random_state)
# clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# print(clf.best_score_)

# model_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1250, random_state=random_state)
# print( cross_val_score(model_2, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean())


# param_grid = {'max_depth': list(range(5, 16))}
# model_3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1250, random_state=random_state)
# clf = GridSearchCV(model_3, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# print(clf.best_score_)

def AdaBoost_scratch(X, y, M=10, learning_rate=1):
    # инициалиазция служебных переменных
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []

    # инициализация весов
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())

    # цикл по длине М
    for m in range(M):
        # обучим базовую модель и получим предсказание
        estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        # Маска для ошибок классификации
        incorrect = (y_predict != y)

        # Оцениваем ошибку
        estimator_error = np.average(incorrect, weights = sample_weight, axis = 0)

        # Вычисляем вес нового алгоритма
        estimator_weight = learning_rate * np.log ((1-estimator_error)/estimator_error)

        # # Получаем новые веса объектов
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Сохраяем результаты данной итерации
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Для удобства переведем в numpy.array
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Получим предсказания
    preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    print('Accuracy = ', (preds == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list


estimator_list, estimator_weight_list, sample_weight_list = AdaBoost_scratch(poly_X, y, M=10, learning_rate=0.001)