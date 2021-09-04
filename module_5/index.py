from pandas import Series
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler, \
    PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA
import calendar
# pd.set_option('display.max_columns', None)
# pd.set_option("max_rows", None)

#  Функция фильтраци выбросов
def outliersData(data, column):
    quantile_3 = data.query('sample == 1')[column].quantile(0.75)
    quantile_1 = data.query('sample == 1')[column].quantile(0.25)
    if quantile_3 == quantile_1:
        return data

    IQR = quantile_3 - quantile_1
    column_min = quantile_1 - 1.5 * IQR
    column_max = quantile_3 + 1.5 * IQR

    count_element_out = data.query('sample == 1')[(data[column] >= column_max) & (data[column] <= column_min)][
        column].count()


    count_element_out_tmp = data[(data[column] >= column_max) & (data[column] <= column_min)][
        column].count()

    if count_element_out_tmp> 0 :
        print(column)
        exit()



    print('Q1 - {} . Q3 - {}. Нижняя граница: {}. Вверхняя граница: {}. Количество элементов за границами: {}'.format(
        round(quantile_1, 2), round(quantile_3, 2), round(column_min, 2), round(column_max, 2), count_element_out
    ))

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].set_title(' Распределение параметра ' + column)
    # sns.distplot(data[column], ax=axes[0])
    # axes[1].set_title(' Параметр ' + column + ' от целевой переменной')
    # sns.boxplot(x='default', y=data[column], data=data, orient='v', ax=axes[1])

    data = data[data[column].between(column_min, column_max)]
    return data



# Функция замены на самое частое значение
def replaceValue(data, columns):
    for column in columns:
        max_ = data[column].value_counts().idxmax()
        data[column] = data[column].apply(
            lambda x: max_ if pd.isnull(x) else (max_ if (x == 'nan' or x.strip() == '') else x))
    return data


# Функция масштабирования признака от 0 до 1
def Max_min_scalar(data, columns, default='default'):
    scaler = MinMaxScaler()
    cor_before = data.query('sample == 1')[columns].corr(data[default])
    data[columns] = scaler.fit_transform(data[[columns]])
    cor_after = data.query('sample == 1')[columns].corr(data[default])
    print('Колонка - {}. Корреляция до масштабирования: {}. Корреляция после масштабирования: {}'.format(
        columns, round(cor_before, 2), round(cor_after, 2)
    ))
    return data

# Логарифмирование
def logNumberValue(data, columns, default='default'):
    for column in columns:
        cor_before = data.query('sample == 1')[column].corr(data[default])
        data[column] = data[column].apply(lambda w: np.log(w + 1))
        cor_after = data.query('sample == 1')[column].corr(data[default])
        print('Колонка - {}. Корреляция до лограмирования: {}. Корреляция после лограмирования: {}'.format(
            column, round(cor_before, 2), round(cor_after, 2)
        ))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].set_title(' Распределение параметра ' + column)
        sns.distplot(data[column], ax=axes[0])
        axes[1].set_title(' Параметр ' + column + ' от целевой переменной')
        sns.boxplot(x='default', y=data[column], data=data, orient='v', ax=axes[1])
    return data

# Полиминальные признаки - для большего понимани расшифруем данные
def PolimicFeatre(data, num_cols):
    poly = PolynomialFeatures(3)
    ppl_data = poly.fit_transform(data[num_cols])
    poly_name = poly.get_feature_names()

    dictionary = {}
    i = 0
    for col in num_cols:
        dictionary['x' + str(i).strip()] = col.strip()
        i += 1

    dictionary_seed = {}
    for col in poly_name:
        col_seed = col
        for k, v in dictionary.items():
            col = col.replace(str(k), str(v))
        dictionary_seed[col_seed] = col

    ppl_data = pd.DataFrame(data=ppl_data, columns=poly_name)
    ppl_data.rename(columns=dictionary_seed, inplace=True)

    num_cols_del = ['age', 'decline_app_cnt', 'income', 'bki_request_cnt', 'score_bki', 'region_rating', 'first_time',
                    '1']
    ppl_data.drop(num_cols_del, axis=1, inplace=True)

    data = data.join(ppl_data, how='left')

    ppl_data_col = ppl_data.columns.to_list()
    num_cols = num_cols + ppl_data_col

    fig, ax = plt.subplots(figsize=(12, 12))
    imp_num = pd.Series(f_classif(data.query('sample == 1')[num_cols], data.query('sample == 1')['default'])[0],
                        index=num_cols)
    imp_num.sort_values(inplace=True, ascending=False)
    imp_num.plot(kind='barh')

    # возьмем 75% самых важных признаков
    num_select = imp_num[imp_num.values >= (imp_num.max() * 0.25)].keys()
    num_drob = imp_num[imp_num.values < (imp_num.max() * 0.25)].keys()
    data.drop(num_drob, axis=1, inplace=True)
    num_cols = num_select.to_list()
    data.columns = data.columns.str.strip()
    return  data, num_cols


df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

df_train['sample'] = 1
df_test['sample'] = 0
df_test['default'] = 0

data = df_test.append(df_train, sort=False).reset_index(drop=True)

# data.info()
#
# data.isna().sum()

# data.query('sample == 1')['default'].value_counts().plot.barh()
# data.query('sample == 1')['default'].value_counts()


num_cols = ['age', 'decline_app_cnt', 'income', 'bki_request_cnt', 'score_bki', 'region_rating', 'first_time']

bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

cat_cols=['education','home_address','work_address','app_date','sna']

target = ['default']

# обработаем данные для столбца
data = Max_min_scalar(data, 'score_bki')

data = logNumberValue(data, num_cols)

# for col in num_cols:
#     data = outliersData(data, col)



# imp_num = pd.Series(f_classif(data.query('sample == 1')[num_cols], data.query('sample == 1')['default'])[0], index=num_cols)
# imp_num.sort_values(inplace=True)
# imp_num.plot(kind='barh')

# num_cor =data.query('sample == 1')[num_cols + target].corr()['default'].sort_values()

data, num_cols = PolimicFeatre(data, num_cols)


# БИНАРНЫЕ ПРИЗНАКИ
label_encoder = LabelEncoder()
for column in bin_cols:
    data[column] = label_encoder.fit_transform(data[column])

# МЕТОД ГЛАВНЫХ КМПОНЕНТ
date_pca = data[['car_type', 'car']]
data.drop(['car_type', 'car'], axis=1, inplace=True)
pca = PCA(n_components=1)
date_pca_ = pca.fit_transform(date_pca)
data['car_type_car'] = date_pca_
data['car_type_car'] = data['car_type_car'].round(6)

bin_cols = ['sex', 'car_type_car', 'good_work', 'foreign_passport']


# bin_cor = data.query('sample == 1')[bin_cols + target].corr()['default'].sort_values()
#
# bin_cat = Series(mutual_info_classif(data.query('sample == 1')[bin_cols +target ], data.query('sample == 1')['default'],
#                                      discrete_features=True), index=bin_cols)
# bin_cat.sort_values(inplace=True)
# bin_cat.plot(kind='barh')



# Стандартизация числовых переменных
X_num = StandardScaler().fit_transform(data[num_cols].values)

# Категориальные признаки
# пропуски в признаке образование / школу должны уж закончить
data = replaceValue(data, ['education'])

# подачи заявки  - примем как категориальный признак (название месяца)
import calendar
data['app_date'] = pd.to_datetime(data['app_date'])
data['app_date']  = data['app_date'] .apply(lambda x: calendar.month_name[x.month])

# get_dummies переменные - удалим старые столбцы
data = pd.get_dummies(data, columns=cat_cols)

#  влияние категориальных переменных на целевой признак
cat_cols = ['education_ACD', 'education_GRD', 'education_PGR', 'education_SCH',
       'education_UGR', 'home_address_1', 'home_address_2', 'home_address_3',
       'work_address_1', 'work_address_2', 'work_address_3', 'app_date_April',
       'app_date_February', 'app_date_January', 'app_date_March', 'sna_1',
       'sna_2', 'sna_3', 'sna_4']

# imp_cat = Series(mutual_info_classif(data[cat_cols], data['default'],
#                                      discrete_features =True), index =cat_cols)
# imp_cat.sort_values(inplace = True)
# imp_cat.plot(kind = 'barh')


# Стандартизация числовых переменных
X_num = StandardScaler().fit_transform(data[num_cols].values)
data[num_cols] = X_num

#  ОБУчение
X = np.hstack([data.query('sample == 1')[num_cols].values, data.query('sample == 1')[bin_cols].values,
               data.query('sample == 1')[cat_cols].values])
Y = data.query('sample == 1')['default'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Гипер параметры
# penalty = ['l1', 'l2']
# C = np.logspace(0, 4, 10)
# hyperparameters = dict(C=C, penalty=penalty)
#
# model = LogisticRegression()
# clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)
# best_model = clf.fit(X_train, y_train)
# print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Лучшее C:', best_model.best_estimator_.get_params()['C'])
# exit()


model = LogisticRegression(solver='liblinear', class_weight='balanced', C=1, penalty='l2')
model.fit(X_train, y_train)
probs_ = model.predict(X_test)
pred_probs = model.predict_proba(X_test)

fpr, tpr, threshold = roc_curve(y_test, pred_probs[:,1])
roc_auc = roc_auc_score(y_test, pred_probs[:,1])

print('accuracy_score:', accuracy_score(y_test, probs_))
print('precision_score:', precision_score(y_test, probs_))
print('recall_score:', recall_score(y_test, probs_))
print('f1_score:', f1_score(y_test, probs_))
сf_mtx = confusion_matrix(y_test, probs_)
print()
print('confusion matrix:', '\n', сf_mtx)
tn, fp, fn, tp = сf_mtx.ravel()
print()
print('Предсказано невозращение кредита клиентом, по факту вернувшим кредит: {} \n\
 или {}% от всех вернувших \n'.format(fp, round((fp/(fp+tn))*100, 2)))
print('Предсказан возврат кредита клиентом, по факту не вернувшим кредит: {} \n\
или {}% от всех не вернувших\n'.format(fn,
                                        round((1-recall_score(y_test,probs_))*100, 2)))
print()
print('roc_auc_score:', roc_auc_score(y_test, pred_probs[:, 1]))






