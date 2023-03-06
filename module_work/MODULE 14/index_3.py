from string import punctuation

import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from nltk.stem.snowball import PorterStemmer
from nltk.stem.snowball import RussianStemmer

import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from nltk.corpus import stopwords
sw = stopwords.words("russian")[1:50]
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

s = pd.Series(["Мама мыла раму мылом", "У попа была собака он её любил"], dtype="string")

s = s.str.lower()
s = s.str.strip()
s = s.str.split(" ", expand=True)


def preprocess_txt(line):
    exclude = set(punctuation)
    spls = "".join(i for i in line.strip() if i not in exclude).split()


    spls = [morph.parse(i.lower())[0].normal_form for i in spls]

    spls = [i for i in spls if i not in sw and i != ""]

    return spls

s = pd.Series(["Мама мыла раму мылом", "У попа была собака он её любил"], dtype="string")
# s = s.apply(lambda x: preprocess_txt(x))


documents = ["I like this movie, it's funny.", 'I hate this movie.', 'This was awesome! I like it.', 'Nice one. I love it.']
# count_vectorizer = CountVectorizer()
# bag_of_words = count_vectorizer.fit_transform(documents)
# feature_names = count_vectorizer.get_feature_names()
# data = pd.DataFrame(bag_of_words.toarray(), columns = feature_names)

from nltk.util import ngrams
text = "I like this movie, it's funny. I hate this movie. This was awesome! I like it. Nice one. I love it."
tokenized = text.split()
bigrams = ngrams(tokenized, 2)
print(bigrams)




