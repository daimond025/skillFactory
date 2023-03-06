import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# df = pd.read_csv('./data/train.csv')
# df = df[df.store_nbr == 25].copy()
# df = df.groupby(["date"])['unit_sales'].sum().reset_index()

# df.to_csv('./data/trainTMP.csv', index=False)
df = pd.read_csv('./data/trainTMP.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

timeDay = pd.date_range(df['date'].min(), end=df['date'].max())
union = pd.DataFrame()
union["date"] = timeDay
union = pd.merge(left=union, right=df[["date", 'unit_sales']], how="left", on=["date"])
union["unit_sales"].fillna(0, inplace=True)

union = union.set_index(pd.DatetimeIndex(union['date']))
union.drop(['date'], axis=1, inplace=True)


def Upper(value):
    return value.mean() + 3 * value.std()


def Lower(value):
    return value.mean() - 3 * value.std()


def MeanMaxMin(value):
    return (value.max() + value.min()) / 2


# union['sales_mean'] = union["unit_sales"].rolling(window=5).mean()
# union['sales_std'] = union["unit_sales"].rolling(window=5).std()
#
# union['upper']  = union["unit_sales"].rolling(window=5).apply(Upper, raw=False)
# union['lower']  = union["unit_sales"].rolling(window=5).apply(Lower, raw=False)
# union['sales_func'] = union["unit_sales"].rolling(window=5).apply(MeanMaxMin, raw=False)

# union['sales_mean_50'] = union["unit_sales"].rolling(window=50).mean()
# union['sales_ewn_10'] = union["unit_sales"].ewm(min_periods=10, span=10).mean()
#
# union['diff'] = union['sales_mean_50'] - union['sales_ewn_10']
# union['diff'] = np.sign(union['diff']).diff()
# print(union[union['diff'] != 0])


# разделение данных
tscv = TimeSeriesSplit()
train_test_groups = tscv.split(union["unit_sales"])
for train_index, test_index in train_test_groups:
    print("TRAIN size:", len(train_index), "TEST size:", len(test_index))


def decompose(union):
    union = union.set_index(pd.DatetimeIndex(union['date']))
    union.drop(['date'], axis=1, inplace=True)

    decomposition = seasonal_decompose(union, model='additive')
    decomposition.plot()
    pyplot.show()

    trend_part = decomposition.trend
    seasonal_part = decomposition.seasonal
    residual_part = decomposition.resid

    residual_part = residual_part.dropna()
    print(residual_part.shape)

    seasonal_part = seasonal_part.dropna()
    print(seasonal_part.shape)

    trend_part = trend_part.dropna()
    print(trend_part.shape)


def adfulleftDef(df):
    test = adfuller(df["unit_sales"])
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:  # проверка, больше ли критического полученное значение для нашего ряда
        print('ряд не стационарен')
    else:
        print('ряд стационарен')

# adfulleftDef(union)
