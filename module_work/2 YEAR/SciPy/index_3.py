from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import tensorflow_probability as tfp
trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)

points = [(0.0, -0.0), (1.0, -1.5574077246549023), (2.0, -1.1578212823495777),
(3.0, 0.45231565944180985), (4.0, -0.3006322420239034), (5.0, 0.13352640702153587),
(6.0, -7.750470905699148), (7.0, 3.172908552159191), (8.0, -2.3478603091954366),
(9.0, 0.8109944158318942), (10.0, 0.5872139151569291)]
points = [list(item) for item in points ]
x = [item[0] for item in points ]
y = [item[1] for item in points ]

f = interp1d(x, y,kind='quadratic')

xx = f(4.5)
print(xx)
exit()

x = np.linspace(0, 30, num=11, endpoint=True)
y = np.cos(-x**2)

xnew = np.linspace(0, 30, num=41, endpoint=True)

f = interp1d(x, y)
f2 = interp1d(x, y, kind='quadratic')
f3 = interp1d(x, y, kind='cubic')
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.show()

# plt.plot(x, y, 'o')
# plt.show()