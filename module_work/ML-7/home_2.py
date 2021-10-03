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

X, y = np.concatenate((dataset[0], X_2)), np.concatenate(
    (dataset[1], np.array([2] * len(X_2)))
)

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
import warnings
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

connectivity = kneighbors_graph(X, n_neighbors=2,
                                include_self=False)
# делаем матрицу смежности симметричной
connectivity = 0.5 * (connectivity + connectivity.T)

ac = AgglomerativeClustering(n_clusters=3 ,  linkage='average', connectivity = connectivity)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Error",
        category=UserWarning)
    ac.fit(X)
y_pred = ac.labels_.astype(np.int)
print(ac.n_leaves_)

unique, counts = np.unique(y_pred, return_counts=True)
print(dict(zip(unique, counts)) )
