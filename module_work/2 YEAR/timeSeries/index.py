import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

df = pd.read_csv('./data/AirPassengers.csv')
df.columns = (['Month','Pass'])

df = df.set_index(pd.DatetimeIndex(df['Month']))
df.drop(['Month'], axis = 1, inplace = True)



decomposition = seasonal_decompose(df, model='additive')
decomposition.plot()
pyplot.show()

trend_part = decomposition.trend
seasonal_part = decomposition.seasonal
residual_part = decomposition.resid


residual_part = residual_part.dropna()
residual_part.head()


print(residual_part.head())
print(residual_part.tail())


