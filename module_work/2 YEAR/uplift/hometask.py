
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from catboost import CatBoostClassifier


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

#  TODO tmp qini

discount = pd.read_csv("./data/datased-test.csv", low_memory=False)
def qini(df, treatment = 'treatment', target='visit'):
    tmp = df[["uplift_score", treatment, target]]

    # 1. Отранжируем выборку по значению uplift в убывающем порядке
    ranked = tmp.sort_values("uplift_score", ascending=False)

    N_c = sum(ranked[target] == 0)
    N_t = sum(ranked[target] == 1)

    # Посчитаем в отсортированном датафрейме основные показатели, которые используются при расчете qini
    ranked['n_c1'] = 0
    ranked['n_t1'] = 0
    ranked.loc[ranked[target] == 0, 'n_c1'] = 1
    ranked.loc[ranked[target] == 1, 'n_t1'] = 1
    ranked['n_c1/nc'] = ranked.n_c1.cumsum() / N_c
    ranked['n_t1/nt'] = ranked.n_t1.cumsum() / N_t

    # Посчитаем qini curve и рандомную прямую под ней
    ranked['uplift'] = round(ranked['n_t1/nt'] - ranked['n_c1/nc'], 5)

    # Добавим случайную кривую
    uplift_mean = ranked['uplift'].median()
    ranked['random_uplift'] = round(ranked["uplift_score"].rank(pct=True, ascending=False) * uplift_mean,5)


    ranked["n"] = ranked["uplift_score"].rank(pct=True, ascending=False)
    # Немного кода для визуализации
    plt.plot(ranked['n'], ranked['uplift'], color='r')
    plt.plot(ranked['n'], ranked['random_uplift'], color='b')
    plt.show()
    print(ranked)

    exit()


    return (ranked['uplift'] - ranked['random_uplift']).sum()
qini(discount)


discount = pd.read_csv("./data/bogouplift.csv", sep="\t")
def qini_df(df):
    # 1. Отранжируем выборку по значению uplift в убывающем порядке
    ranked = df.sort_values("uplift_score", ascending=False)

    N_c = sum(ranked['target_class'] <= 1)
    N_t = sum(ranked['target_class'] >= 2)

    # Посчитаем в отсортированном датафрейме основные показатели, которые используются при расчете qini
    ranked['n_c1'] = 0
    ranked['n_t1'] = 0
    ranked.loc[ranked.target_class == 1, 'n_c1'] = 1
    ranked.loc[ranked.target_class == 3, 'n_t1'] = 1
    ranked['n_c1/nc'] = ranked.n_c1.cumsum() / N_c
    ranked['n_t1/nt'] = ranked.n_t1.cumsum() / N_t

    # Посчитаем qini curve и рандомную прямую под ней
    ranked['uplift'] = round(ranked['n_t1/nt'] - ranked['n_c1/nc'], 5)


    # Добавим случайную кривую
    ranked['random_uplift'] = round(ranked["uplift_score"].rank(pct=True, ascending=False) * ranked['uplift'].iloc[-1],
                                    5)

    ranked["n"] = ranked["uplift_score"].rank(pct=True, ascending=False)
    # Немного кода для визуализации
    plt.plot(ranked['n'], ranked['uplift'], color='r')
    plt.plot(ranked['n'], ranked['random_uplift'], color='b')
    plt.show()

    return (ranked['uplift'] - ranked['random_uplift']).sum()
qini_df(discount)


df =  pd.read_csv("./data/criteo-uplift-v2.1.csv")


def qini_df(df , target = ''):
    # 1. Отранжируем выборку по значению uplift в убывающем порядке
    ranked = df.sort_values("uplift_score", ascending=False)

    N_c = sum(ranked[target] == 0)
    N_t = sum(ranked[target] == 1)

    # Посчитаем в отсортированном датафрейме основные показатели, которые используются при расчете qini
    ranked['n_c1'] = 0
    ranked['n_t1'] = 0
    ranked.loc[ranked[target] == 0, 'n_c1'] = 1
    ranked.loc[ranked[target] == 1, 'n_t1'] = 1
    ranked['n_c1/nc'] = ranked.n_c1.cumsum() / N_c
    ranked['n_t1/nt'] = ranked.n_t1.cumsum() / N_t


    # Посчитаем qini curve и рандомную прямую под ней
    ranked['uplift'] = round(ranked['n_t1/nt'] - ranked['n_c1/nc'], 5)
    # Добавим случайную кривую
    ranked['random_uplift'] = round(ranked["uplift_score"].rank(pct=True, ascending=False) * ranked['uplift'].iloc[-1],5)

    ranked["n"] = ranked["uplift_score"].rank(pct=True, ascending=False)

    # Немного кода для визуализации
    plt.plot(ranked['n'], ranked['uplift'], color='r')
    plt.plot(ranked['n'], ranked['random_uplift'], color='b')
    plt.show()

    return (ranked['uplift'] - ranked['random_uplift']).sum()



feature_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10','f11', ]
target_col = 'conversion'
treatment_col = 'treatmen'

df_train, df_test = train_test_split(df, stratify=df[[treatment_col, target_col]], random_state=13, test_size=0.3)


s_learner = BaseSClassifier(learner=CatBoostClassifier(random_seed=13, verbose=0))

# Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
s_learner.fit(X=df_train[feature_cols],
              treatment=df_train[treatment_col],
              y=df_train[target_col])

uplift_vals = s_learner.predict(np.array(df_test[feature_cols].values.copy()))


df_test['uplift_score'] = uplift_vals
qini_df(df_test)