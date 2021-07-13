import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab

df = pd.read_csv('framingham.xls')
df.dropna(axis=0,inplace=True)


dff = df[df['TenYearCHD'] == 1].copy()



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
