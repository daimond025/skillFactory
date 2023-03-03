import numpy as np
import scipy
import pandas as pd

from implicit.als import AlternatingLeastSquares, np
from implicit.evaluation import mean_average_precision_at_k


ratings = pd.read_csv("./data/ml-100k/u.data", sep="\t", header=None)
ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings.sort_values('timestamp', inplace=True)
ratings['score'] = (ratings['rating'] > 2).apply(int)


from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size=0.2, shuffle=False)
print(ratings)
exit()