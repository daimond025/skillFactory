from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

n_samples = 1500
dataset = datasets.make_blobs(n_samples=n_samples, centers=2, center_box=(-7.0, 7.5),
                              cluster_std=[1.4, 1.7],
                              random_state=42)
X_2, _ = datasets.make_blobs(n_samples=n_samples, random_state=170, centers=[[-4, -3]], cluster_std=[1.9])
transformation = [[1.2, -0.8], [-0.4, 1.7]]
X_2 = np.dot(X_2, transformation)
X, y = np.concatenate((dataset[0], X_2)), np.concatenate((dataset[1], np.array([2] * len(X_2))))

# TODO KMeans
# from sklearn.cluster import KMeans
# k_means = KMeans(n_clusters=3,random_state=42)
# k_means.fit(X)
# # print(np.round(k_means.cluster_centers_).astype(np.int))
# _, counts = np.unique(k_means.labels_, return_counts=True)
# print(counts)

# TODO EM-алгоритм
# from sklearn.mixture import GaussianMixture
#
# gm = GaussianMixture(n_components=3,random_state=42 )
# gm.fit(X)
# y_pred = gm.predict(X)
# print(gm.means_)
# print(np.round(gm.means_).astype(np.int))
#
# unique, counts = np.unique(y_pred, return_counts=True)
# print(dict(zip(unique, counts)) )

# TODO AgglomerativeClustering
# import warnings
# from sklearn.neighbors import kneighbors_graph
# from sklearn.cluster import AgglomerativeClustering
#
# connectivity = kneighbors_graph(X, n_neighbors=6,
#                                 include_self=False)
# connectivity = 0.5 * (connectivity + connectivity.T)
#
# ac = AgglomerativeClustering(n_clusters=3, connectivity=connectivity)
# ac.fit(X)
# y_pred = ac.labels_.astype(np.int)
# print(ac.n_leaves_)
#
# unique, counts = np.unique(y_pred, return_counts=True)
# print(dict(zip(unique, counts)))
#
#
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import dendrogram, linkage
# import pandas as pd
# df = pd.read_csv('playbook/food.txt', sep=' ')
#
# X = df.drop('Name',axis=1)
# X = StandardScaler().fit_transform(X)
#
# Z = linkage(X, method='average', metric='euclidean')
#
# names = df.Name.values
# dend = dendrogram(Z, color_threshold=0, labels=names, orientation='left')


# TODO DBSCAN
# from sklearn.cluster import DBSCAN
#
# dbscan = DBSCAN(eps=0.8, min_samples=35)
# dbscan.fit(X)
# y_pred = dbscan.labels_.astype(np.int)
#
# unique, counts = np.unique(y_pred, return_counts=True)
# print(dict(zip(unique, counts)) )
# print()

# TODO Меьрики
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict

# X = StandardScaler().fit_transform(X)
# X = MinMaxScaler().fit_transform(X)

# TODO 7.14.10
ac = AgglomerativeClustering(n_clusters=3)
y_pred = ac.fit_predict(X)
print(v_measure_score(labels_true=y, labels_pred=y_pred))

X = MinMaxScaler().fit_transform(X)
ac = AgglomerativeClustering(n_clusters=3)
y_pred = ac.fit_predict(X)
print(v_measure_score(labels_true=y, labels_pred=y_pred))
exit()


# TODO 7.14.9
# ac = AgglomerativeClustering(n_clusters=3)
# y_pred = ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
#
# X = StandardScaler().fit_transform(X)
# ac = AgglomerativeClustering(n_clusters=3)
# y_pred = ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
# exit()

# TODO 7.14.8
# dbscan = DBSCAN(eps=0.9, min_samples=35)
# y_pred = dbscan.fit_predict(X)
#
# y_Seed = y.tolist()
# y_pred_ = []
# y_ = []
# i = 0
# for y in y_pred:
#     if y>=0:
#         y_pred_.append(y)
#         y_.append(y_Seed[i])
#     i +=1
# print(v_measure_score(labels_true=y_, labels_pred=y_pred_))
# exit()

# TODO  7.14.7
dbscan = DBSCAN(eps=0.9, min_samples=35)
y_pred = dbscan.fit_predict(X)
print(v_measure_score(labels_true=y, labels_pred=y_pred))

dbscan = DBSCAN(eps=0.8, min_samples=35)
y_pred = dbscan.fit_predict(X)
print(v_measure_score(labels_true=y, labels_pred=y_pred))
exit()

# TODO  7.14.5
# connectivity = kneighbors_graph(X, n_neighbors=6, include_self=False)
# connectivity = 0.5 * (connectivity + connectivity.T)
# ac = AgglomerativeClustering(n_clusters=3)
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
#
# ac = AgglomerativeClustering(n_clusters=3,connectivity=connectivity)
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
# exit()
#

# TODO  7.14.5
# ac = AgglomerativeClustering(n_clusters=3,linkage='ward')
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
#
# ac = AgglomerativeClustering(n_clusters=3,linkage='complete')
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
#
# ac = AgglomerativeClustering(n_clusters=3,linkage='average')
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
#
# ac = AgglomerativeClustering(n_clusters=3,linkage='single')
# y_pred =ac.fit_predict(X)
# print(v_measure_score(labels_true=y, labels_pred=y_pred))
# exit()

# TODO  7.14.4
# kmeans_mini_batch = MiniBatchKMeans(n_clusters=3, n_init=1, random_state=42)
# kmeans_1 = KMeans(n_clusters=3, n_init=1, random_state=42)
# kmeans_mini_batch_pred = kmeans_mini_batch.fit_predict(X)
# kmeans_pred_1 = kmeans_1.fit_predict(X)
# print( v_measure_score(labels_true=y, labels_pred=kmeans_pred_1))
# print( v_measure_score(labels_true=y, labels_pred=kmeans_mini_batch_pred))
# exit()



# kmeans_1 = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=42)
# kmeans_2 = KMeans(n_clusters=3, init='random', n_init=1, random_state=42)
# kmeans_pred_1 = kmeans_1.fit_predict(X)
# kmeans_pred_2 = kmeans_2.fit_predict(X)
# a = v_measure_score(labels_true=y, labels_pred=kmeans_pred_1)
# c = v_measure_score(labels_true=y, labels_pred=kmeans_pred_2)


X = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_pred = kmeans.labels_
# a = silhouette_score(X=X, labels=kmeans_pred)
# a = homogeneity_score(labels_true=y, labels_pred=kmeans_pred)
# a = completeness_score(labels_true=y, labels_pred=kmeans_pred)
a = v_measure_score(labels_true=y, labels_pred=kmeans_pred)
print(a)


em_gm = GaussianMixture(n_components=3, random_state=42)
em_gm.fit(X)
y_pred = em_gm.predict(X)
# b = silhouette_score(X=X, labels=y_pred)
# b = homogeneity_score(labels_true=y, labels_pred=y_pred)
# b = completeness_score(labels_true=y, labels_pred=y_pred)
b = v_measure_score(labels_true=y, labels_pred=y_pred)
print(b)

ac = AgglomerativeClustering(n_clusters=3)
ac.fit(X)
y_pred = ac.labels_
# c = silhouette_score(X=X, labels=y_pred)
# c = homogeneity_score(labels_true=y, labels_pred=y_pred)
# c = completeness_score(labels_true=y, labels_pred=y_pred)
c = v_measure_score(labels_true=y, labels_pred=y_pred)
print(c)

dbscan = DBSCAN(eps=0.9,min_samples=35 )
dbscan.fit(X)
y_pred = dbscan.labels_
# Z = silhouette_score(X=X, labels=y_pred)
# Z = homogeneity_score(labels_true=y, labels_pred=y_pred)
# Z = completeness_score(labels_true=y, labels_pred=y_pred)
Z = v_measure_score(labels_true=y, labels_pred=y_pred)
print(Z)


# silhouette_dict = defaultdict(list)
# for n in range(2, 11):
#     # инициализируем алгоритмы:
#     KM = KMeans(n_clusters=n, random_state=42)
#     gm = GaussianMixture(n_components=n, random_state=42)
#     ac = AgglomerativeClustering(n_clusters=n)
#
#     # создаём словарь, где ключи - названия алгоритмов, значения - сами алгоритмы:
#     alg_dict = {'K-means': KM, 'EM-алгоритм': gm, 'Агломеративная кластеризация': ac}
#
#     # цикл по словарю:
#     for alg_name, algo in alg_dict.items():
#         labels = algo.fit_predict(X)  # получаем предсказания
#         sil_score = silhouette_score(X, labels)  # считаем коэффициент силуэта
#
#         # добавляем в словарь в list, соответствующему рассматриваемому алгоритму,
#         # пару вида : (число кластеров, коэффициент силуэта)
#         silhouette_dict[alg_name].append((n, sil_score))
#
# for alg_name in silhouette_dict.keys():
#     # сохраняем число кластеров и коэф. силуэта для пары,
#     # в которой коэф. максимальный для данного алгоритма:
#     n_clusters, sil_score = max(silhouette_dict[alg_name], key=lambda x: x[1])
#
#     # выводим  название алгоритма и искомое число кластеров (и коэф. силуэта):
#     print(f"{alg_name} : {n_clusters}, {sil_score}")


