import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
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
# data = pd.read_csv('data_flats.csv',sep=";")
#
# data_ = data[(data['sub_area'] == 'Perovo') | (data['sub_area'] == 'Lefortovo') | (data['sub_area'] == 'Basmannoe')
#         | (data['sub_area'] == 'Bogorodskoe')][['price_doc', 'sub_area' ]]
# ax = sns.boxplot(x="sub_area", y="price_doc", data=data_)
# print(data)

# vis_data = pd.read_csv("train.csv",
#                        encoding = 'ISO-8859-1',
#                        low_memory = False)
# vis_data['payment_date'] = pd.to_datetime(vis_data['payment_date'])
# vis_data['payment_date_day'] = vis_data['payment_date'].dt.weekday
# print(vis_data[(vis_data['payment_date_day'] == 5 )| (vis_data['payment_date_day'] == 6 ) ].shape )


# ecology_dict ={
#     "no data" : 0,
#     "poor" : 1,
#     "satisfactory" : 2,
#     "good" : 3,
#     "excellent" : 4
# }
# vis_data = pd.read_csv("data_flats.csv", sep=';')
# vis_data.ecology = vis_data.ecology.replace(to_replace=ecology_dict)
# dumm = pd.get_dummies(vis_data.ecology)
# print( len(vis_data))
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(lower_bound)
    print(upper_bound)
    return ys[(ys < upper_bound) & (ys > lower_bound)]


vis_data = pd.read_csv("train.csv",
                       encoding = 'ISO-8859-1',
                       low_memory = False)
values  = vis_data['balance_due'].dropna().values
tmp = outliers_iqr(values)
max_ = np.max(tmp)
min_ = np.min(tmp)
print(max_ - min_)
print(max_)
print(min_)





