import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from IPython.core.display import display

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'year': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'temp': [-4.7, -6.1, -5.5, -3.3, -7.1, -3.1, -5.2, -7.3, -12.1, -6.6, -5.9, -6.3],
    'temp_2': [6.1, 9.2, 11.5, 8.6, 12.1, 3.9, 8.4, 10.1, 9.4, 8.9, ]
})

temp = df['temp_2'].copy()
print(temp.mean())
print(temp.std())

temp_q1 = temp.quantile(0.25, interpolation='midpoint')
temp_q3 = temp.quantile(0.75, interpolation='midpoint')
print(temp_q3 - temp_q1)

temp_min = temp_q1 - 1.5 * (temp_q3 - temp_q1)
temp_max = temp_q3 + 1.5 * (temp_q3 - temp_q1)
print(temp[(temp < temp_min) | (temp > temp_max)])

from scipy.stats import norm

p_ = 1 - norm.cdf(820, loc=700, scale=120)

p_2 = norm.cdf(820, loc=700, scale=120)
p_2_ = norm.cdf(730, loc=700, scale=120)
print(p_2 - p_2_)

from scipy.stats import norm
from scipy.stats import t
import math
import numpy as np


def confidence_interval_norm(alpha, sigma, n, mean):
    value = -norm.ppf(alpha / 2) * sigma / math.sqrt(n)
    return mean - value, mean + value
def confidence_interval_t(alpha, s, n, mean):
    value = -t.ppf(alpha / 2, n - 1) * s / math.sqrt(n)
    return mean - value, mean + value


p_ = 132/189
print(p_)
def confidence_interval_norm_prop(alpha, n, p_prop):
    value = -norm.ppf(alpha / 2) * math.sqrt((p_prop * (1-p_prop))/n)
    return p_prop - value, p_prop + value


alpha = 1 - 0.93
value = -norm.ppf(alpha/2)
print(value)
print(-norm.ppf((1-0.99)/2))
print(-t.ppf((1-0.95)/2, 99))
print(-norm.ppf((1-0.95)/2))



print(12.35 + 1.65 * (2.4 / 8))
print(12.35 - 1.65 * (2.4 / 8))
