from surprise import KNNWithMeans, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import pandas as pd

movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')

movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)


dataset = movies_with_ratings[["title", "userId", "rating"]].rename({
    "userId": "uid",
    "title": "iid"
})


reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset, reader)

trainset, testset = train_test_split(data, test_size=0.15)
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)
test_pred = algo.test(testset)
accuracy.rmse(test_pred, verbose=True)

algo.predict(uid=2, iid='Fight Club (1999)').est