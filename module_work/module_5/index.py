from pandas import Series
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve

def replaceValue(data, columns):
    for column in columns:
        max_ = data[column].value_counts().idxmax()
        data[column] = data[column].apply(lambda x: max_ if pd.isnull(x) else ( max_ if (x == 'nan' or x.strip() == '')  else x  ))
    return data

data = pd.read_csv("./data/train.csv", encoding = 'ISO-8859-1', low_memory = False)
#  зачена на п
data = replaceValue(data, ['education'])


# for i in ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']:
#     plt.figure()
#     sns.distplot(data[i][data[i] > 0].dropna(), kde = False, rug=False)
#     plt.title(i)
#     plt.show()

num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']
number_log_column = ['age',  'income']
def logNumberValue(data, columns):
    for column in columns:
        data[column] = data[column].apply(lambda w: np.log(w + 1 ))
    return data
data = logNumberValue(data, num_cols)



# imp_num = pd.Series(f_classif(data[num_cols], data['default'])[0], index = num_cols)
# imp_num.sort_values(inplace = True)
# imp_num.plot(kind = 'barh')


cat_cols =  ['education', 'home_address', 'work_address']
bin_cols = ['sex', 'car', 'car_type', 'foreign_passport', 'good_work']
label_encoder = LabelEncoder()
for column in bin_cols:
    data[column] = label_encoder.fit_transform(data[column])

# imp_cat = Series(mutual_info_classif(data[bin_cols + cat_cols], data['default'],
#                                      discrete_features =True), index = bin_cols + cat_cols)
# imp_cat.sort_values(inplace = True)
# imp_cat.plot(kind = 'barh')

X_cat = OneHotEncoder(sparse = False).fit_transform(data[cat_cols].values)
X_num = StandardScaler().fit_transform(data[num_cols].values)
X = np.hstack([X_num, data[bin_cols].values, X_cat])
Y = data['default'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# probs = model.predict_proba(X_test)
# probs = probs[:,1]
# fpr, tpr, threshold = roc_curve(y_test, probs)
# roc_auc = roc_auc_score(y_test, probs)

penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression()
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X_train, y_train)
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])




# def get_boxplot(date, column):
#     fig, ax = plt.subplots(figsize=(15, 15))
#     sns.boxplot(x='default', y=data[col], data=data, orient='v', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title('Boxplot for ' + column)
#     plt.show()
#
# for col in num_cols:
#     get_boxplot(data, col)