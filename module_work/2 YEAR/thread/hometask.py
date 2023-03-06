import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split,  cross_validate

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats.stats import pearsonr
from scipy.stats import rankdata




data = pd.read_csv('./data/winequality-red.csv', sep=';')

data.columns = data.columns.str.replace(' ', '_')

data['target'] =  data.apply(
            lambda row: 0 if (row['quality'] < 6.5)
            else 1, axis=1)

Y_n= data['target'].copy()
X = data.drop(['target', 'quality'], axis=1).copy()

scaler = StandardScaler()
X_n = scaler.fit_transform(X)

#  TODO подбор признаков
col = list(data.columns)
col.remove('target')
col.remove('quality')
x_data = pd.DataFrame(data=X_n, columns=col )

trans = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
paramsTransform = trans.fit_transform(x_data)
paramsPoly = pd.DataFrame(data=paramsTransform, columns=trans.get_feature_names(x_data.columns))

fs = SelectKBest(score_func=f_classif, k=20)
X_selected = fs.fit_transform(paramsPoly, Y_n)
cols = fs.get_support(indices=True)

features_df_new = paramsPoly.iloc[:,cols]
print(features_df_new)
exit()


# TODO кроссвалидация
# model_test = SVC(kernel='rbf', C=46.415, probability=True)
# scores = cross_val_score(model_test, X_n, Y_n, cv=cv, scoring='accuracy')


# def confusion_matrix_scorer(clf, X, y):
#     y_pred = clf.predict(X)
#     y_pred_probs = clf.predict_proba(X)
#
#     roc = roc_auc_score(y, y_pred_probs[:, 1])
#     cm = confusion_matrix(y, y_pred)
#
#     return {
#         'tn': cm[0, 0],
#         'tn': cm[0, 0],
#         'fp': cm[0, 1],
#         'fn': cm[1, 0],
#         'tp': cm[1, 1],
#         'roc': roc
#     }
# cv_results = cross_validate(model_test, X_n, Y_n, cv=cv, scoring=confusion_matrix_scorer)
# print(cv_results)
exit()





#  TODO Подбор параметров модели
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': np.logspace(-1, 3, num=10),
#     # 'coef0': np.linspace(1, 100, num=10),
# }
# model = SVC()
# cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, train_size=0.8, random_state=45)
# clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy',cv=cv)
# clf.fit(X_train, Y_train)
#
# best_params = clf.best_params_
# best_score = clf.best_score_
#
# print(best_params)
# print(best_score)
# exit()

# TODO кросвалидация перебором
#  TODO кросвалидация
# loop_scores = []
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=45)
# # for idx_train, idx_test in cv.split(X_n, Y_n):
# #     X_train, y_train, X_test, y_test = X_n[idx_train], Y_n[idx_train], X_n[idx_test], Y_n[idx_test]
# #
# #     model_test = SVC(kernel='rbf', C=46.415, probability=True)
# #     model_test.fit(X_train, y_train)
# #
# #     probs_ = model_test.predict(X_test)
# #     pred_probs = model_test.predict_proba(X_test)
# #
# #     loop_scores.append(
# #         roc_auc_score(y_test, pred_probs[:, 1])
# #     )


def graphdistrib(data, column, band = 0.3):
    X = data[column].values
    X = X.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(X)
    parcelDensity = kde.score_samples(X)

    X = X.reshape(1, -1)[0]
    Y = np.exp(parcelDensity.reshape(1, -1)[0])

    fig, ax = plt.subplots(figsize=(20, 15))
    plt.bar(X, Y)
    plt.title('Optimal estimate feature ' + column +  ' with Gaussian kernel')
    plt.show()
def graphdistribSns(data,column ):
    sns.kdeplot(data=data, x=column)
