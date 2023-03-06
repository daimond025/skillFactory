import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from catboost import CatBoostClassifier

df = pd.read_csv("./data/bogouplift.csv", sep="\t")


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


#  TODO S LEARN
s_learner = BaseSClassifier(learner=CatBoostClassifier(random_seed=13, verbose=0))

# Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
s_learner.fit(X=df_train[feature_cols],
              treatment=df_train[treatment_col],
              y=df_train[target_col])

uplift_vals = s_learner.predict(
    np.array(df_test[feature_cols].values.copy())
)

# Мы получили какие-то значения рамках решения задачи классификации, давайте посмотрим на qini score
df_test['uplift_score'] = uplift_vals
print(qini_df(df_test))

#  TODO T LEARN
t_learner = BaseTClassifier(learner=CatBoostClassifier(random_seed=13, verbose=0))

# Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
t_learner.fit(X=df_train[feature_cols],
              treatment=df_train[treatment_col],
              y=df_train[target_col])

uplift_vals = t_learner.predict(
    np.array(df_test[feature_cols].values.copy())
)

# Мы получили какие-то значения рамках решения задачи классификации, давайте посмотрим на qini score
df_test['uplift_score'] = uplift_vals
qini_df(df_test)

