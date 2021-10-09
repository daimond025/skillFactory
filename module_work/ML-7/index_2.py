import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

n_samples = 1500
dataset = datasets.make_blobs(n_samples=n_samples, centers=2, center_box=(-7.0, 7.5),
                              cluster_std=[1.4, 1.7],
                              random_state=42)
X_2, _ = datasets.make_blobs(n_samples=n_samples, random_state=170, centers=[[-4, -3]], cluster_std=[1.9])
transformation = [[1.2, -0.8], [-0.4, 1.7]]
X_2 = np.dot(X_2, transformation)
X, y = np.concatenate((dataset[0], X_2)), np.concatenate((dataset[1], np.array([2] * len(X_2))))


# plt.rcParams['figure.figsize'] = 10, 10
# plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
# plt.show()


unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3,  random_state=42)
k_means.fit(X)

a = k_means.cluster_centers_
a_ = k_means.labels_
print(np.round(a).astype(np.int))

unique_, counts_= np.unique(a_, return_counts=True)
print(dict(zip(unique_, counts_)))
