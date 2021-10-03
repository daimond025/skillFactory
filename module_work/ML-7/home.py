from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Количество объектов в каждом датасете
n_samples = 1500

# Вписанные круги
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
# Оставляем только признаки датасета, так как для кластеризации нам не нужны истинные классы объектов
X, y = noisy_circles
noisy_circles = X

# Полукруги
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
X, y = noisy_moons
noisy_moons = X

# Кластеры в форме круга
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
X, y = blobs
blobs = X

# Отсутствие кластерной структуры
no_structure = np.random.rand(n_samples, 2)

# Кластеры лентовидной формы
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = X_aniso

# Кластеры в форме кругов с различной дисперсией
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
X, y = varied
varied = X


from sklearn.cluster import KMeans

 #TODO KMeans
# k_means = KMeans(n_clusters=8,
#                  init='k-means++', # 'k-means++', 'random', numpy.array
#                  max_iter=300
#                 )
#
# datasets_params_list = [
#     (blobs, {'n_clusters': 3}),
#     (varied, {'n_clusters': 3}),
#     (aniso, {'n_clusters': 3}),
#     (noisy_circles, {'n_clusters': 2}),
#     (noisy_moons, {'n_clusters': 2}),
#     (no_structure, {'n_clusters': 3})]

# for i, (X, k_means_params) in enumerate(datasets_params_list, start=1):
#     X = StandardScaler().fit_transform(X)
#     k_means = KMeans(n_clusters=k_means_params['n_clusters'])
#
#     k_means.fit(X)
#     y_pred = k_means.labels_.astype(np.int)
#
#     plt.subplot(f'23{i}')
#     plt.xticks([]);
#     plt.yticks([])
#     colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                          '#f781bf', '#a65628', '#984ea3',
#                                          '#999999', '#e41a1c', '#dede00']),
#                                   int(max(y_pred) + 1))))
#     plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred])

 #TODO GaussianMixture
# from sklearn.mixture import GaussianMixture
# em_gm = GaussianMixture(n_components=1,
#                         max_iter=100,
#                         init_params='kmeans' # 'kmeans’, ‘random’
#                        )
# datasets_params_list = [
#     (blobs, {'n_clusters': 3}),
#     (varied, {'n_clusters': 3}),
#     (aniso, {'n_clusters': 3}),
#     (noisy_circles, {'n_clusters': 2}),
#     (noisy_moons, {'n_clusters': 2}),
#     (no_structure, {'n_clusters': 3})]
#
# for i, (X, em_gm_params) in enumerate(datasets_params_list, start=1):
#     X = StandardScaler().fit_transform(X)
#     em_gm = GaussianMixture(n_components=em_gm_params['n_clusters'])
#
#     em_gm.fit(X)
#     y_pred = em_gm.predict(X)
#     print(np.unique(y_pred, return_counts=True))
#     # exit()


# TODO GaussianMixture


import warnings
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

datasets_params_list = [
    (blobs, {'n_clusters': 3, 'n_neighbors': 10}),
    (varied, {'n_clusters': 3, 'n_neighbors': 2}),
    (aniso, {'n_clusters': 3, 'n_neighbors': 2}),
    (noisy_circles, {'n_clusters': 2, 'n_neighbors': 10}),
    (noisy_moons, {'n_clusters': 2, 'n_neighbors': 10}),
    (no_structure, {'n_clusters': 3, 'n_neighbors': 10})]

for i, (X, ac_params) in enumerate(datasets_params_list, start=1):
    X = StandardScaler().fit_transform(X)

    # строим матрицу смежности
    connectivity = kneighbors_graph(X,
                                    n_neighbors=ac_params['n_neighbors'],
                                    include_self=False)
    # делаем матрицу смежности симметричной
    connectivity = 0.5 * (connectivity + connectivity.T)

    ac = AgglomerativeClustering(n_clusters=ac_params['n_clusters'],
                                 linkage='average',
                                 connectivity=connectivity)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Error",
            category=UserWarning)
        ac.fit(X)
    y_pred = ac.labels_.astype(np.int)

