import numpy as np
import pandas as pd
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import xgboost as xgb
# import lazypredict

# from lazypredict.Supervised import LazyRegressor
# from pandas_profiling import ProfileReport
from scipy.stats import ttest_ind
from itertools import combinations
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor

warnings.simplefilter('ignore')
sns.set()

# Неизвестные значения
UNKNOWN_VAL = -1
# Неизвестные строка
UNKNOWN_STR = 'UNKNOWN'


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def getSellIdModel_fromURL(data):
    # Функция получения МОДЕЛИ И ИД ПРОДАВЦА ИЗ URL МАШИНЫ
    # ИЛИ если url датасете нет - ставим значение по умолчанию
    if ('car_url' in data.columns) and ('model' not in data.columns):
        data['model'] = data['car_url'].str.split('/').str.get(7).str.strip()
    else:
        data['model'] = UNKNOWN_STR

    if ('car_url' in data.columns) and ('sell_id' not in data.columns):
        data['sell_id_arr'] = data['car_url'].str.split('/').str.get(-2).str.strip()
        data['sell_id'] = data['sell_id_arr'].str.split('-').str.get(0).str.strip()

        data.drop('sell_id_arr', axis=1, inplace=True)
    else:
        data['sell_id'] = UNKNOWN_VAL
    return data

RANDOM_SEED = 42
VERSION = 16
DIR_TRAIN = 'input/'
DIR_TEST = 'input/'
VAL_SIZE = 0.20  # 20%

train_site = pd.read_csv(DIR_TRAIN + 'Base.csv', lineterminator='\n')
train = pd.read_csv(DIR_TRAIN + 'all_auto_ru_09_09_2020.csv', lineterminator='\n')
test = pd.read_csv(DIR_TEST + 'test.csv')
sample_submission = pd.read_csv(DIR_TEST + 'sample_submission.csv')
train.columns = [col.strip().replace('\r', '') for col in train.columns]

#  MODEL и SELL_ID
train_site = getSellIdModel_fromURL(train_site)
train = getSellIdModel_fromURL(train)
test = getSellIdModel_fromURL(test)

# ПЕРВОНЧАЛЬНОЕ ЗНАЧЕНИЕ
train.dropna(subset=['productionDate', 'mileage', 'car_class'], inplace=True)
train.dropna(subset=['price'], inplace=True)

columns = ['bodyType', 'brand', 'productionDate', 'engineDisplacement', 'mileage']
df_train = train[columns]
df_test = test[columns]
y = train['price']

df_train['sample'] = 1
df_test['sample'] = 0

data = df_test.append(df_train, sort=False).reset_index(drop=True)
for colum in ['bodyType', 'brand', 'engineDisplacement']:
    data[colum] = data[colum].astype('category').cat.codes

X = data.query('sample == 1').drop(['sample'], axis=1)
X_sub = data.query('sample == 0').drop(['sample'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
model = CatBoostRegressor(iterations=5000,
                          random_seed=RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE'],
                          silent=True,
                          )
model.fit(X_train, y_train,
          # cat_features=cat_features_ids,
          eval_set=(X_test, y_test),
          verbose_eval=0,
          use_best_model=True,
          # plot=True
          )
predict = model.predict(X_test)
print(f"Точность модели по метрике MAPE: {(mape(y_test, predict)) * 100:0.2f}%")

# print("Список колонок, которых нет в train, но есть в test:", dif_list)
# dif_list = list(set(test.columns).difference(train.columns))
# for item in dif_list:
#     print(test[item].nunique())
#     exit()
#     print(item, test[item].nunique())
# exit()
