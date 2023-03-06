
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Для создания S и T learnerов используем BaseSClassifier и BaseTClassifier (поскольку мы решаем задачу классификации)
from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from catboost import CatBoostClassifier

df = pd.read_csv("./data/discountuplift.csv", sep="\t")



# Возьмем функцию для оценки Qini curve с прошлого занятия
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


df['old_target'] = (df['target_class'] % 2 == 0).apply(int)


feature_cols = ['recency', 'history', 'used_discount', 'used_bogo', 'is_referral',
                'zip_code_Rural', 'zip_code_Surburban', 'zip_code_Urban',
                'channel_Multichannel', 'channel_Phone', 'channel_Web']
target_col = 'old_target'
treatment_col = 'treatment'

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, stratify=df[[treatment_col, target_col]], random_state=13, test_size=0.3)


# # TODO Создадим базовый S-learner
# s_learner = BaseSClassifier(learner=CatBoostClassifier(random_seed=13, verbose=0))
#
# # Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
# s_learner.fit(X=df_train[feature_cols],
#               treatment=df_train[treatment_col],
#               y=df_train[target_col])
#
# uplift_vals = s_learner.predict(np.array(df_test[feature_cols].values.copy()))
#
# #TODO Построим S-learner с логистической регрессией
# from sklearn.linear_model import LogisticRegression
#
# s_learner = BaseSClassifier(learner=LogisticRegression(verbose=0))
# s_learner.fit(X=df_train[feature_cols],
#               treatment=df_train[treatment_col],
#               y=df_train[target_col])
#
# uplift_vals = s_learner.predict(np.array(df_test[feature_cols].values.copy()))
# df_test['uplift_score'] = uplift_vals
# qini_df(df_test)
#
# # Создадим классификатор
# ctb_clf = CatBoostClassifier(random_seed=13, verbose=0)
#
# # Обучим его на исходных данных
# ctb_clf.fit(df_train[feature_cols + [treatment_col]], df_train[target_col])

#  TODO T-learner прямой метод
# ctb_clf = CatBoostClassifier(random_seed=13, verbose=0)
# # Обучим его на исходных данных
# ctb_clf.fit(df_train[feature_cols + [treatment_col]], df_train[target_col])
#
# # Опишем метод, рассчитывающих аплифт
# def compute_s_uplift(ctb_clf, X):
#     X[treatment_col] = 1
#     predict_treatment = ctb_clf.predict_proba(X[feature_cols + [treatment_col]])[:, 1]
#
#
#     X[treatment_col] = 0
#     predict_control = ctb_clf.predict_proba(X[feature_cols + [treatment_col]])[:, 1]
#
#     uplift = predict_treatment - predict_control
#     return uplift
#
#
# # Оценим аплифт эффекты
# df_test['uplift_score'] = compute_s_uplift(ctb_clf, df_test)
# qini_df(df_test)BaseTClassifier


#  TODO  Создадим базовый T-learner
# t_learner = BaseTClassifier(learner=CatBoostClassifier(random_seed=13, verbose=0))
#
# # Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
# t_learner.fit(X=df_train[feature_cols],
#               treatment=df_train[treatment_col],
#               y=df_train[target_col])
#
# uplift_vals = t_learner.predict(np.array(df_test[feature_cols].values.copy()))
#
# # Мы получили какие-то значения рамках решения задачи классификации, давайте посмотрим на qini score
# df_test['uplift_score'] = uplift_vals
# qini_df(df_test)


#  TODO  Видим, что T-learner уже работает чуть лучше, чем S-learner. Давайте, обучим t-learner руками
ctb_clf_tr = CatBoostClassifier(random_seed=13, verbose=0)
ctb_clf_ctrl = CatBoostClassifier(random_seed=13, verbose=0)

# Обучим их на исходных данных
ctb_clf_tr.fit(df_train[df_train[treatment_col] == 1][feature_cols], df_train[df_train[treatment_col] == 1][target_col])
ctb_clf_ctrl.fit(df_train[df_train[treatment_col] == 0][feature_cols],
                 df_train[df_train[treatment_col] == 0][target_col])


# Опишем метод, рассчитывающих аплифт
def compute_t_uplift(ctb_clf_tr, ctb_clf_ctrl, X):
    X[treatment_col] = 1
    predict_treatment = ctb_clf_tr.predict_proba(X[feature_cols + [treatment_col]])[:, 1]

    X[treatment_col] = 0
    predict_control = ctb_clf_ctrl.predict_proba(X[feature_cols + [treatment_col]])[:, 1]

    uplift = predict_treatment - predict_control
    return uplift


# Оценим аплифт эффекты
df_test['uplift_score'] = compute_t_uplift(ctb_clf_tr, ctb_clf_ctrl, df_test)
qini_df(df_test)
