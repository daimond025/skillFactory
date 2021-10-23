import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

from bs4 import UnicodeDammit

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))



train = pd.read_csv('./data/all_auto_ru_09_09_2020.csv', engine='python')


print(train.columns)
exit()
