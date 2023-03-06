import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

train = pd.read_csv("./data/train.csv")
top1 = train[train.item_nbr == 103501]
# top1.to_csv('./data/top1.csv', index=False)
# exit()

top1 = pd.read_csv("./data/top1.csv")
top1['date'] = pd.to_datetime(top1['date'])
top1['year'] = top1['date'].dt.year

unit_sales_by_date = top1.groupby('date').sum()['unit_sales']

# X = unit_sales_by_date.values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t%s: %.3f' % (key, value))
#
# if result[0] < result[4]["5%"]:
#     print ("Reject Ho - Time Series is Stationary")
# else:
#     print ("Failed to Reject Ho - Time Series is Non-Stationary")


def moving_average_forecast(series, window_size):
    forecast = []

    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_average_days = 6
shown_train_size = moving_average_days * 3

moving_avg = moving_average_forecast(unit_sales_by_date,moving_average_days)
moving_avg = pd.Series(moving_avg, index = unit_sales_by_date[moving_average_days:].index)

print(moving_avg[-moving_average_days:].shape,unit_sales_by_date[-moving_average_days:].shape)

print( "mean_squared_error", mean_squared_error(
    unit_sales_by_date.values[-moving_average_days:],
    moving_avg[-moving_average_days:]
))
print("mean_absolute_error", mean_absolute_error(
    unit_sales_by_date.values[-moving_average_days:],
    moving_avg[-moving_average_days:]))

print("mean_absolute_percentage_error", mean_absolute_percentage_error(
    unit_sales_by_date.values[-moving_average_days:],
    moving_avg[-moving_average_days:]))



# print(unit_sales_by_date)
#
predict_size = 7
df = pd.DataFrame()
df["Original Values"]  = unit_sales_by_date
df["shift7"] = df["Original Values"].shift(predict_size)
df["shift8"] = df["shift7"].shift()
df["shift9"] = df["shift8"].shift()
df["shift10"] = df["shift9"].shift()
df["shift11"] = df["shift10"].shift()
df["shift12"] = df["shift11"].shift()
df["shift13"] = df["shift12"].shift()
df.dropna(inplace=True)
#
x_train, y_train = df[:-predict_size].drop(["Original Values"], axis =1), df[:-predict_size]["Original Values"]
x_test, y_test  = df[-predict_size:].drop(["Original Values"], axis =1), df[-predict_size:]["Original Values"]
#
#
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)
ar_predictions = pd.Series(reg.predict(x_test), index=x_test.index)

print("mean_squared_error",mean_squared_error(y_test, ar_predictions))
print("mean_absolute_error",mean_absolute_error(y_test, ar_predictions))


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(unit_sales_by_date.values.squeeze(), lags=6, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(unit_sales_by_date, lags=6, ax=ax2)



from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y_train.values.reshape(-1), order=(6,0,6))
print()
exit()


from statsmodels.graphics.tsaplots import plot_predict

train_size = len(y_train)
test_size = predict_size
arima_predictions = model.fit().predict(start=train_size,end=train_size+test_size -1,  dynamic=False)

plt.plot(pd.Series(arima_predictions, index=y_test.index) ,label = "Predictions")
plt.plot(pd.concat([y_train, y_test], axis = 0)[-shown_train_size:], label = "Original" )
plt.legend(loc="upper right")

