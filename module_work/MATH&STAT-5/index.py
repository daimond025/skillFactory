import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.stats import t



def confidence_interval_norm(alpha, sigma, n, mean):

    value =-norm.ppf(alpha / 2) * sigma / math.sqrt(n)
    return mean - value, mean + value

def confidence_interval_t(alpha, s, n, mean):
    value = -t.ppf(alpha / 2, n - 1) * s / math.sqrt(n)
    return mean - value, mean + value

p = 0.698
n = 189
alpha = 0.1
print(p +  (-norm.ppf(alpha/2)) * math.sqrt((p* (1-p))/(n)))
print(p -  -norm.ppf(alpha/2) * math.sqrt((p* (1-p))/(n)))
print(-norm.ppf(0.1/2))
# print(confidence_interval_norm(0.01,1150, 250,3540))

# print(confidence_interval_norm(0.1,400,15,2000))
# print(confidence_interval_t(0.05,400,15,2000))
# print(confidence_interval_t(0.01,400,15,2000))


# alpha = 0.05
# value = -norm.ppf(alpha/2)
# value = t.ppf((1 + 0.95)/2, 100-1)


# df = pd.read_csv('framingham.xls')
# df.dropna(axis=0,inplace=True)
#
#
# dff = df[df['TenYearCHD'] == 1].copy()

# print(df['diabetes'].value_counts(normalize = True))
# print(df['diabetes'].value_counts())

# diabetes_groups=df.groupby('diabetes')
# srisk_diabetes_=diabetes_groups['TenYearCHD'].value_counts(normalize = True)
# print(srisk_diabetes_)


# diabetes_groups=df.groupby('TenYearCHD')
# srisk_diabetes_=diabetes_groups['diabetes'].value_counts(normalize = True)

# diabetes_groups=df.groupby('diabetes')
# print(diabetes_groups['sysBP'].mean())






# math_ = np.array([20,23,29,22,50,43,35, 98,28])
# math_2 = np.array([20,23,29,22,50,43,35])
#
# ru_ = np.array([70,65,58,90,45,57,50])
# ru_2 = np.array([70,65,58,90,45,57,50,90,38])
#
# #
# # print(np.mean(math_))
# # print(np.mean(math_2))
# #
# # print(np.mean(ru_))
# # print(np.mean(ru_2))
#
# print(np.corrcoef(math_, ru_2))
# # print(np.std(ru_, ddof=1))
# # print(ru_.mean())
# # print(math_.size)
