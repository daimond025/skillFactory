from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score, confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
import  pandas as pd
import  numpy as np

def print_logisitc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'acc = {acc:.2f} F1-score = {f1:.2f}')

def convert_county(row, list_):
    if row['native-country'] is None or row['native-country'] not in list_:
        row['native-country'] = 'other'
    return row
def prepare_adult_data():
    adult = pd.read_csv('adult.csv',
                        names=['age', 'workclass', 'fnlwgt', 'education',
                               'education-num', 'marital-status', 'occupation',
                               'relationship', 'race', 'sex', 'capital-gain',
                               'capital-loss', 'hours-per-week', 'native-country', 'salary'])

    country = adult['native-country'].value_counts().rename_axis('native-country').reset_index(name='counts')

    country = country[country['counts'] >= 100]['native-country'].to_list()
    country.remove(' ?')

    adult= adult.apply(lambda row:  convert_county(row, country) , axis=1)

    # Сконвертировать целевой столбец в бинарные значения
    adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')
    # Сделать one-hot encoding для некоторых признаков
    adult = pd.get_dummies(adult,
                           columns=['native-country', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                    'sex'])

    # Нормализовать нуждающиеся в этом признаки
    a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
    norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
    adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features

    # Разбить таблицу данных на матрицы X и y
    X = adult[list(set(adult.columns) - set(['salary']))].values
    y = adult['salary'].values

    # Добавить фиктивный столбец единиц (bias линейной модели)
    X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

    return X, y
X, y = prepare_adult_data()
model = LogisticRegression()
model.fit(X, y)
y_predicted = model.predict(X)
print(f1_score(y, y_predicted))
# C = np.arange(0.01,1.01,0.01)
# max_f1 = -1
# max_c = -1
# for c in C:
#     model = LogisticRegression(C=c, penalty="l2",)
#     model.fit(X, y)
#     y_predicted = model.predict(X)
#     f2 = f1_score(y, y_predicted)
#     if f2 > max_f1:
#         max_c = c
#         max_f1 = f2
# print(max_f1)
# print(max_c)
# exit()



# print_logisitc_metrics(y,Y_predicted )
# print(confusion_matrix(y, Y_predicted))

# y_pred_proba_pack = model.predict_proba(X)
# print(np.round(roc_auc_score(y, y_pred_proba_pack[:, 1]), 2))
