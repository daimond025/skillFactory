import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

df_train = pd.read_csv("./data/train.csv", index_col="PassengerId")
df_test = pd.read_csv("./data/test.csv", index_col="PassengerId")

df_cleared = df_train.dropna()

df_cleared['age_group'] = df_cleared['Age'].apply(lambda x: x // 10)
p_vals = []
coeffs = []
age_groups = df_cleared['age_group'].unique()
for i in range(len(age_groups)):
    age_group_1_sum = df_cleared[df_cleared['age_group'] == age_groups[i]]['Survived'].sum()
    age_group_1_count = len(df_cleared[df_cleared['age_group'] == age_groups[i]])
    for j in range(i + 1, len(age_groups)):
        age_group_2_sum = df_cleared[df_cleared['age_group'] == age_groups[j]]['Survived'].sum()
        age_group_2_count = len(df_cleared[df_cleared['age_group'] == age_groups[j]])
        p_value = \
        proportions_ztest(count=[age_group_1_sum, age_group_2_sum], nobs=[age_group_1_count, age_group_2_count])[1]
        p_vals.append(p_value)
        coeffs.append(age_groups[j])

p_vals_corrected = multipletests(p_vals, method="bonferroni")[1]
for i, pval in enumerate(p_vals_corrected):
    if pval < 0.05:
        print(coeffs[i])
print(multipletests(p_vals, method="bonferroni"))
exit()

df_clearedTrain = df_train.dropna()
df_clearedTest = df_test.dropna()

ageTrain = df_clearedTrain['Age'].values
ageTest = df_clearedTest['Age'].values

stt = sp.stats.ttest_ind(ageTrain, ageTest)
print(ageTrain)
print(ageTest)

# survived_ages = df_cleared[df_cleared['Survived']==1]['Age'].values
# died_ages = df_cleared[df_cleared['Survived']!=1]['Age'].values
# stt = sp.stats.ttest_ind(died_ages, survived_ages)
