import numpy as np
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


np.random.seed(2)
pvals = np.random.uniform(0, 0.3, 100)

# plt.hist(pvals, 40, facecolor='green')
# plt.xlabel('data')
# plt.ylabel('Count')
# plt.title("Uniform Distribution Histogram (Bin size 20)")
# plt.axis([0, 0.3, 0, 100]) # x_start, x_end, y_start, y_end
# plt.grid(True)
# plt.show(block = False)


pvals_corrected = multipletests(pvals, method="holm")
plt.hist(pvals_corrected, 40, facecolor='green')
plt.grid(True)
plt.show(block = False)
