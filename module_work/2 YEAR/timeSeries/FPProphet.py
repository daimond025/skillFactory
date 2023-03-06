import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
import holidays

pjme = pd.read_csv('./data/PJME_hourly.csv', index_col=[0], parse_dates=[0])

split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()

# Создадим признаки
def create_features(df, label=None):
    us_holidays = holidays.country_holidays('US')
    """
    создаем признаки из datetime индекса
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    df['holiday'] = df.apply(lambda row: 1 if row["date"] in us_holidays else 0, axis=1)

    predictDA = 24
    predict2DA = 48
    df["ShiftOne"] = df[label].shift(predictDA)
    df["ShiftTwo"] = df[label].shift(predict2DA)
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', "holiday", "ShiftOne", "ShiftTwo"]]

    if label:
        y = df[label]
        return X, y
    return X

X_train, y_train = create_features(pjme_train, label='PJME_MW')
X_test, y_test = create_features(pjme_test, label='PJME_MW')



reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)


pjme_test['MW_Prediction'] = reg.predict(X_test)
err = mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],y_pred=pjme_test['MW_Prediction'])




pjme_test['error'] = pjme_test['PJME_MW'] - pjme_test['MW_Prediction']
pjme_test['abs_error'] = pjme_test['error'].apply(np.abs)
error_by_day = pjme_test.groupby(['year','month','dayofmonth']) \
    .mean()[['PJME_MW','MW_Prediction','error','abs_error']]