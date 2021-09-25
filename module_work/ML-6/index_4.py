import pandas as pd
import numpy as np

from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits

from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats.distributions import randint

np.random.seed(42)

dataset = load_digits()
X, y = dataset['data'], dataset['target']


dataset = load_digits()
X, y = dataset['data'], dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# def compute_meta_feature(clf, X_train, X_test, y_train, cv):
#     n_classes = len(np.unique(y_train))
#     X_meta_train = np.zeros((len(y_train), n_classes), dtype=np.float32)
#
#     splits = cv.split(X_train)
#     for train_fold_index, predict_fold_index in splits:
#         X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
#         y_fold_train = y_train[train_fold_index]
#
#         folded_clf = clone(clf)
#         folded_clf.fit(X_fold_train, y_fold_train)
#
#         X_meta_train[predict_fold_index] = folded_clf.predict_proba(X_fold_predict)
#
#     meta_clf = clone(clf)
#     meta_clf.fit(X_train, y_train)
#
#     X_meta_test = meta_clf.predict_proba(X_test)
#
#     return X_meta_train, X_meta_test


def generate_meta_features(classifiers, X_train, X_test, y_train, cv):
    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(classifiers)
    ]
    stacked_features_train = np.hstack([
        features_train for features_train, features_test in features
    ])

    stacked_features_test = np.hstack([
        features_test for features_train, features_test in features
    ])

    return stacked_features_train, stacked_features_test



cv = KFold(n_splits=10, shuffle=True, random_state=42)

def compute_metric(clf, X_train=X_train, y_train=y_train, X_test=X_test):
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    return np.round(f1_score(y_test, y_test_pred, average='macro'), 6)

# TODO 1
# stacked_features_train, stacked_features_test = generate_meta_features([
#     LogisticRegression(penalty='l1', C=0.001, solver='saga', multi_class='ovr', max_iter=2000, random_state=42),
#     LogisticRegression(penalty='l2', C=0.001, solver='saga', multi_class='multinomial', max_iter=2000, random_state=42),
#     RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
#     GradientBoostingClassifier(n_estimators=200, random_state=42)
# ], X_train, X_test, y_train, cv)
#
# clf = LogisticRegression(penalty='none', solver='lbfgs', multi_class='auto', random_state=42)
# a =  compute_metric(clf, X_train=stacked_features_train, y_train=y_train, X_test=stacked_features_test)

# TODO 2
# stacked_features_train, stacked_features_test = generate_meta_features([
#     RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
#     ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42)],
#     X_train, X_test, y_train, cv)
#
# clf = LogisticRegression(penalty='none', solver='lbfgs', multi_class='auto', random_state=42)
# b = compute_metric(clf, X_train=stacked_features_train, y_train=y_train, X_test=stacked_features_test)
# print(b)

# TODO 3
# stacked_features_train, stacked_features_test = generate_meta_features([
#     KNeighborsClassifier(),
#     ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)],
#     X_train, X_test, y_train, cv)
#
# clf = LogisticRegression(penalty='none', solver='lbfgs', multi_class='auto', random_state=42)
# c = compute_metric(clf, X_train=stacked_features_train, y_train=y_train, X_test=stacked_features_test)

# TODO 4
# stacked_features_train, stacked_features_test = generate_meta_features([
#     LogisticRegression(penalty='l1', C=0.001, solver='saga', multi_class='ovr', max_iter=2000, random_state=42),
#     KNeighborsClassifier(),
#     ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42),
#     AdaBoostClassifier()
# ], X_train, X_test, y_train, cv)
#
# clf = LogisticRegression(penalty='none', solver='lbfgs', multi_class='auto', random_state=42)
# z = compute_metric(clf, X_train=stacked_features_train, y_train=y_train, X_test=stacked_features_test)


def compute_meta_feature(clf, X_train, X_test, y_train, cv):
    n_classes = len(np.unique(y_train))
    X_meta_train = np.zeros((len(y_train), n_classes), dtype=np.float32)

    splits = cv.split(X_train, y_train)
    for train_fold_index, predict_fold_index in splits:
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]

        folded_clf = clone(clf)
        folded_clf.fit(X_fold_train, y_fold_train)

        X_meta_train[predict_fold_index] = folded_clf.predict_proba(X_fold_predict)

    meta_clf = clone(clf)
    meta_clf.fit(X_train, y_train)

    X_meta_test = meta_clf.predict_proba(X_test)

    return X_meta_train, X_meta_test


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

stacked_features_train, stacked_features_test = generate_meta_features([
    RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=24, n_jobs=-1, random_state=42),
    ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    LogisticRegression()
    ],
    X_train, X_test, y_train, cv)

# clf = LogisticRegression(penalty='none', solver='lbfgs', multi_class='auto', random_state=42)
# clf = RandomForestClassifier(random_state=42)
# clf = KNeighborsClassifier()
clf = AdaBoostClassifier(lea)
clf =  ExtraTreesClassifier(n_estimators=100,  n_jobs=-1, random_state=42)
z = compute_metric(clf, X_train=stacked_features_train, y_train=y_train, X_test=stacked_features_test)
print(z)

