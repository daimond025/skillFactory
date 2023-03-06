import os
import string
import annoy
import codecs

from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
from gensim.models import Word2Vec

import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import re
from string import punctuation
exclude = set(punctuation)
sw = set(get_stop_words("ru"))
morpher = MorphAnalyzer()


goodSimilar = pd.read_csv('./data/ProductsDatasetPrepare.csv',  encoding='utf8')

model = Word2Vec.load('./model/similar_model')
index = annoy.AnnoyIndex(100 ,'angular')

def preprocess_txt(line):
    line = str(line).lower()
    line = re.sub(",", " ", line)
    line = re.sub("\.", " ", line)
    line = re.sub("-", " ", line)

    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [i for i in spls if i not in sw and i != ""]
    # удалим число как мало инфомративное свойство
    spls = [i for i in spls if not str(i).isnumeric()]

    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = " ".join(i for i in spls)
    return spls

index_map = {}

counter = 0

for index, row in goodSimilar.iterrows():

    n_w2v = 0
    prodictId = row['product_id']
    index_map[counter] = prodictId

    productName = preprocess_txt(row['title'])
    vector = np.zeros(100)
    for word in productName:
        if word in model.wv:
            vector += model.wv[word]
            n_w2v += 1
    if n_w2v > 0:
        vector = vector / n_w2v
    index.add_item(counter, vector)
    counter += 1
# goodSimilar = pd.read_csv('./data/ProductsDatasetPrepare.csv',  encoding='utf8')
# goodSimilar['text'] = goodSimilar['feature'] + ' ' + goodSimilar['target']
#
# def preprocessTxtWec2(line):
#     line = str(line).lower()
#     line = re.sub(",", " ", line)
#     line = re.sub("\.", " ", line)
#     line = re.sub("-", " ", line)
#
#     spls = "".join(i for i in line.strip() if i not in exclude).split()
#     spls = [i for i in spls if i not in sw and i != ""]
#     # удалим число как мало инфомративное свойство
#     spls = [i for i in spls if not str(i).isnumeric()]
#
#     spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
#
#     return spls
# goodSimilar['feature'] = goodSimilar['text'].apply(preprocessTxtWec2)
# print(goodSimilar)
# exit()
#


# data = pd.read_csv('./data/ProductsDataset.csv',  encoding='utf8')
# def preprocess_txt(line):
#     line = str(line).lower()
#     line = re.sub(",", " ", line)
#     line = re.sub("\.", " ", line)
#     line = re.sub("-", " ", line)
#
#     spls = "".join(i for i in line.strip() if i not in exclude).split()
#     spls = [i for i in spls if i not in sw and i != ""]
#     # удалим число как мало инфомративное свойство
#     spls = [i for i in spls if not str(i).isnumeric()]
#
#     spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
#     spls = " ".join(i for i in spls)
#     return spls
# data['feature'] = data['descrirption'].apply(preprocess_txt)
# data['target'] = data['title'].apply(preprocess_txt)
# data.to_csv('./data/ProductsDatasetPrepare.csv', index=False)
