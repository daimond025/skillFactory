import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# def make_data(N, f=0.3, rseed=1):
#     rand = np.random.RandomState(rseed)
#     x = rand.randn(N)
#     x[int(f * N):] += 5
#     return x
#
# x = make_data(2000)
# x_d = np.linspace(-4, 8, 2000)
#
# kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
# kde.fit(x[:, None])
# logprob = kde.score_samples(x_d[:, None])
# logprob = np.exp(logprob)
# plt.fill_between(x_d, (logprob))
# # widths = bins[1:] - bins[:-1]
# # (density * widths).sum()


# from scipy import stats
# x = make_data(2000)
# x_d = np.linspace(-4, 8, 2000)
# kernel = stats.gaussian_kde(x)
# z = kernel(x_d)

# https://stackoverflow.com/questions/13320262/calculating-the-area-under-a-curve-given-a-set-of-coordinates-without-knowing-t