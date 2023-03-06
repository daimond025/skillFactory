import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest



# np.random.seed(13)
# random_normal = np.random.normal(5, 2, 200)
# random_bin = np.random.choice([0, 1], size=(100,), p=[0.8, 0.2])

# tt = stats.ttest_1samp(random_normal, 0.0)
# tt = stats.ttest_1samp(random_normal, 5.0)
# tt = stats.binom_test(x=[sum(random_bin), len(random_bin) - sum(random_bin)], p=0.5)
# tt = stats.binom_test(x=[sum(random_bin), len(random_bin) - sum(random_bin)], p=0.2)
# tt = stats.binom_test(x=[sum(random_bin), len(random_bin) - sum(random_bin)], p=0.2)


# random_normal_5 = np.random.normal(5, 2, 100)
# random_normal_false = np.random.normal(5.1, 2, 100)
# tt = stats.ttest_ind(random_normal_5, random_normal)
# tt1 = stats.ttest_ind(random_normal_false, random_normal)

# random_bin_2 = np.random.choice([0, 1], size=(100,), p=[0.8, 0.2])
# random_bin_false = np.random.choice([0, 1], size=(100,), p=[0.6, 0.4])
#
# tt = proportions_ztest(count=[sum(random_bin), sum(random_bin_2)], nobs=[len(random_bin), len(random_bin_2)])
# tt1 = proportions_ztest(count=[sum(random_bin), sum(random_bin_false)], nobs=[len(random_bin), len(random_bin_false)])

# np.random.seed(21)
# random_normal = np.random.normal(18, 20, 10)
# tt = stats.ttest_1samp(random_normal, 7.0)
# print(tt )
