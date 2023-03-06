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

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)


index = annoy.AnnoyIndex(100, 'angular')

index_map = {}
counter = 0

def preprocess_txt(line):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls

sentences = []
morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)
c = 0
with codecs.open("./data/Otvety.txt", "r", "utf-8") as fin:
    for line in tqdm(fin):
        spls = preprocess_txt(line)

        if len(spls):
            sentences.append(spls)
            c += 1
            print(spls)
            exit()
        if c > 500000:
            break

# new_model = Word2Vec.load('w2v_model')
# index = annoy.AnnoyIndex(100, 'angular')
# with codecs.open("./data/prepared_answers.txt", "r", "utf-8") as f:
#     for line in tqdm(f):
#         n_w2v = 0
#         spls = line.split("\t")
#         index_map[counter] = spls[1]
#
#         question = preprocess_txt(spls[0])
#
#         vector = np.zeros(100)
#         for word in question:
#             if word in new_model.wv:
#                 vector += new_model.wv[word]
#                 n_w2v += 1
#         if n_w2v > 0:
#             vector = vector / n_w2v
#         index.add_item(counter, vector)
#
#         counter += 1
# #
# index.build(10)
# index.save('speaker.ann')


new_model = Word2Vec.load('w2v_model')
index = index.load('speaker.ann')