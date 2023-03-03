import pickle
import pandas as pd
import implicit
import lightfm
import scipy

import string
# Библиотека построения индекса приближенного поиска ближайших соседей
import annoy
import numpy as np
import re

from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
from gensim.models import FastText
from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

# Для фильтрации пунктуации
exclude = set(string.punctuation)
# Для приведения слов в начальной форме
morpher = MorphAnalyzer()

# Для фильтрации стоп-слов
sw = get_stop_words("ru")


mostPopular = pd.read_csv('./data/mostPopular.csv')
mostPopularItem = list(mostPopular['itemId'].values)


dataTransform = pd.read_csv('./data/dataTransform.csv')
users = list(dataTransform['user_id'].values)

import random as rnd

usersMostPopular = pd.DataFrame()
for user in users:
    userPopular = dataTransform[(dataTransform['user_id'] == user) &
        .AnnoyIndex(20(dataTransform['itemId'].isin(mostPopularItem))]


    if not userPopular.empty:
        userPopular['Buy'] = 1
        usersMostPopular = pd.concat([usersMostPopular, userPopular], ignore_index=True, axis=0).copy()
    else:
        number = rnd.randrange(0, len(mostPopularItem) - 1)
        itemId = mostPopularItem[number]
        itemName = dataTransform[dataTransform['itemId'] == itemId].iloc[0]['itemName']
        userNotBy = {"user_id": [user],
                     'itemId': [mostPopularItem[number]],
                     'itemName': [itemName],
                     'Buy': [0]}
        userPopular = pd.DataFrame.from_dict(userNotBy)
        usersMostPopular = pd.concat([usersMostPopular, userPopular], ignore_index=True, axis=0).copy()








# def preprocess_txt(line):
#     line = re.sub(r'\.|\"|\,', ' ', line)
#     line = re.sub('\sх\s', 'x', line)
#     line = re.sub('\"', ' ', line)
#     line = re.sub('\-', ' ', line)
#
#     spls = "".join(i for i in str(line).strip() if i not in exclude).split()
#     spls = [i for i in spls if not i.isdigit()]
#
#     spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
#     spls = [i for i in spls if i not in sw and i != ""]
#
#     return spls
#
# dataTest = pd.read_csv('./data/goods.csv')
# dataTest['itemName'] = dataTest["itemName"].apply(preprocess_txt)
