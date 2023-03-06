import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = load_digits()
X, y = data['input'], data['target']

features = data['feature_names']

def estimate_accuracy(clf, X, y, cv=5):
    return cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()

# tree = DecisionTreeClassifier()
# print("Decision tree:", estimate_accuracy(tree, X, y))
#
# bagging_trees = BaggingClassifier(tree, n_estimators=100)
# print("Decision tree bagging:", estimate_accuracy(bagging_trees, X, y))

# bagging_trees = BaggingClassifier(tree, n_estimators=100, max_features=int(np.sqrt(len(features))))
# print("Decision tree bagging:", estimate_accuracy(bagging_trees, X, y))
max_features_ = 40
n_estimators_ = 5
tree_rnd = DecisionTreeClassifier(max_features=max_features_)
bagging_trees = BaggingClassifier(tree_rnd, n_estimators=n_estimators_)
print("Decision tree bagging:", estimate_accuracy(bagging_trees, X, y))

random_forest = RandomForestClassifier(
    n_estimators=n_estimators_,
    n_jobs=-1,
    max_features=max_features_
)
print("Random Forest:", estimate_accuracy(random_forest, X, y))

