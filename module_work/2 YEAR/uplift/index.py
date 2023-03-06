import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, pandas as pd
import seaborn as sns, math, os, warnings
warnings.filterwarnings('ignore')


discount = pd.read_csv("./data/bogouplift.csv", sep="\t")


discount = discount[['uplift_score', 'target_class']]
ranked = discount.sort_values("uplift_score", ascending=False)

N_c = sum(ranked['target_class'] <= 1)
N_t = sum(ranked['target_class'] >= 2)



ranked['n_c1'] = 0
ranked['n_t1'] = 0
ranked.loc[ranked.target_class == 1, 'n_c1'] = 1
ranked.loc[ranked.target_class == 3, 'n_t1'] = 1


ranked['n_c1/nc'] = ranked['n_c1'].cumsum() / N_c
ranked['n_t1/nt'] = ranked['n_t1'].cumsum() / N_t


ranked['uplift'] = round(ranked['n_t1/nt'] - ranked['n_c1/nc'], 5)

ranked['random_uplift'] = round(ranked["uplift_score"].rank(pct=True, ascending=False) * ranked['uplift'].iloc[-1],5)

print(ranked)
exit()