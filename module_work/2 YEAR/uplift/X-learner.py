# Импортируем необходимые библиотеки и загрузим семпл данных
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Для создания X и R learnerов используем BaseXClassifier и BaseRClassifier (поскольку мы решаем задачу классификации)
from causalml.inference.meta import BaseXClassifier, BaseRClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/discountuplift.csv", sep="\t")
df.head()

df['old_target'] = (df['target_class'] % 2 == 0).apply(int)

feature_cols = ['recency', 'history', 'used_discount', 'used_bogo', 'is_referral',
                'zip_code_Rural', 'zip_code_Surburban', 'zip_code_Urban',
                'channel_Multichannel', 'channel_Phone', 'channel_Web']
target_col = 'old_target'
treatment_col = 'treatment'

df_train, df_test = train_test_split(df, stratify=df[[treatment_col, target_col]], random_state=13, test_size=0.3)

# TODO x_learner
x_learner = BaseXClassifier(outcome_learner=LogisticRegression(random_state=13),
                           effect_learner=LinearRegression())
x_learner.fit(X=df_train[feature_cols],
              treatment=df_train[treatment_col],
              y=df_train[target_col])
uplift_vals = x_learner.predict(np.array(df_test[feature_cols].values.copy()))

# TODO r_learner
r_learner = BaseRClassifier(outcome_learner=LogisticRegression(),
                            effect_learner=LinearRegression())
# Для обучения нам нужны датафрем с факторами, колонка с фактом воздействия
r_learner.fit(X=df_train[feature_cols],
              treatment=df_train[treatment_col],
              y=df_train[target_col])

uplift_vals = r_learner.predict(np.array(df_test[feature_cols].values.copy()))

# Мы получили какие-то значения рамках решения задачи классификации, давайте посмотрим на qini score
df_test['uplift_score'] = uplift_vals
qini_df(df_test)

