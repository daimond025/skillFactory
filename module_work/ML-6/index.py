from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np


def get_labels(data):
    labels = []
    for idx, item in enumerate(data):
        if item[0]**2 + item[1]**2 < 1:
            labels.append(0)
        elif item[0] > 2 and item[1] > 2:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


N = 500
train_data = 7 * np.random.random_sample((N,2)) - np.array([3,3])

RANDOM_SEED = 139
train_labels = get_labels(train_data)

def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xx, yy = get_grid(train_data)

shallow_rf = RandomForestClassifier(n_estimators=5, max_depth=3, n_jobs=-1,
                                    random_state=RANDOM_SEED)

deep_rf = RandomForestClassifier(n_estimators=5, max_depth=100, n_jobs=-1,
                                 random_state=RANDOM_SEED)
# training the tree
shallow_rf.fit(train_data, train_labels)
deep_rf.fit(train_data, train_labels)


predicted_shallow = shallow_rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
predicted_deep = deep_rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


fig, ax = plt.subplots(2, 1, figsize=(10,15))

ax[0].pcolormesh(xx, yy, predicted_shallow, cmap='coolwarm')
ax[0].scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100,
            cmap='coolwarm', edgecolors='black', linewidth=1.5);
ax[0].set_title('Shallow Random Forest')




