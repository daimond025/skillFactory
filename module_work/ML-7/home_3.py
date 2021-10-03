import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./data/snsdata.csv')
data.drop(['gradyear', 'gender',  'age' ], axis=1, inplace=True)

ss = StandardScaler()
data_scaled = pd.DataFrame(ss.fit_transform(data), columns = data.columns)
X = data_scaled.values

# TODO KMeans
# from sklearn.cluster import KMeans
# k_means = KMeans(n_clusters=9,random_state=42)
# k_means.fit(X)
#
# labels = np.round(k_means.labels_).astype(np.int)
# data_scaled['cluster_label'] = labels
#
# for k, group in data_scaled.groupby('cluster_label'):
#     print(k)
#     top_words = group.iloc[:,:-1].mean()\
#                  .sort_values(ascending=False)\
#                  .head(10)
#     print(top_words)

# TODO EM-алгоритм
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=9,random_state=123 )
y_pred = gm.fit_predict(X)
data_scaled['cluster_label'] = y_pred

print(gm.n_iter_)
print(y_pred[7])
print(data_scaled.iloc[7:10]['cluster_label'])


