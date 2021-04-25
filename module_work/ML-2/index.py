import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# data = pd.read_csv('data_flats.csv',sep=";")
# apartment = data.dropna()
# print(data.shape)
# print(apartment.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit_transform(test_data)
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform(test_data)

# data = pd.read_csv('train.csv',sep=",")
# balance = data['balance_due'].values
# balance_np = balance.reshape(-1,1)
#
# scaler = StandardScaler()
# balance_np = scaler.fit_transform(balance_np)
# print(balance_np.min())


# data = pd.read_csv('train.csv',sep=",")
#
# balance = data[data['balance_due'] > 0]['balance_due'].values
#
# balance_np = np.sqrt(balance)
#
# print(balance_np)
# print(np.median(balance_np) -np.mean(balance_np) )
# # print(np.mean(balance_np))

# pd.set_option('display.max_rows', None)
data = pd.read_csv('data_flats.csv',sep=";")

data_ = data[(data['sub_area'] == 'Perovo') | (data['sub_area'] == 'Lefortovo') | (data['sub_area'] == 'Basmannoe')
        | (data['sub_area'] == 'Bogorodskoe')][['price_doc', 'sub_area' ]]
ax = sns.boxplot(x="sub_area", y="price_doc", data=data_)
print(data)




