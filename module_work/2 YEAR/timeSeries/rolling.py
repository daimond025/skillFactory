import numpy as np, pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/gold_clean.csv')
df['ts'] = pd.to_datetime(df.ts)
df = df.iloc[1000:1100]

df = df.reset_index(drop=True)

def myMax(value):
    return max(value)

df['rolling_mean'] = df.close.rolling(window=5).mean()
df['rolling_std']  = df.close.rolling(window=5).std()
df['rolling_max']  = df.close.rolling(window=5).apply(myMax, raw=False)
print(df.shape)
exit()