import numpy as np
from scipy import stats

np.random.seed(2)
random_normal = np.random.normal(2, 4, 10)

ее = stats.kstest((random_normal - np.mean(random_normal)) / np.std(random_normal), 'norm')
print(ее)
exit()
# np.random.seed(2)
# random_normal = np.random.normal(2, 4, 10)
# ее = stats.shapiro(random_normal)


np.random.seed(13)
random_normal = np.random.normal(5, 2, 100)
random_uniform = np.random.uniform(0, 1, 100)


# ее = stats.shapiro(random_normal)
# ее1 = stats.shapiro(random_uniform)


ее = stats.kstest((random_normal - np.mean(random_normal)) / np.std(random_normal), 'norm')
ее1 = stats.kstest(random_uniform, 'norm')
print(ее)
print(ее1)