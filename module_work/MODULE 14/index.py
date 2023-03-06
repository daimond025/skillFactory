import pandas as pd
import datasets
import wandb
from sklearn.feature_extraction.text import CountVectorizer

# documents = ["I like this movie, it's funny.", 'I hate this movie.', 'This was awesome! I like it.', 'Nice one. I love it.']
#
# count_vectorizer = CountVectorizer()
# bag_of_words = count_vectorizer.fit_transform(documents)
# feature_names = count_vectorizer.get_feature_names()
# data  = pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


# from nltk.util import ngrams
# text = "I like this movie, it's funny. I hate this movie. This was awesome! I like it. Nice one. I love it."
# tokenized = text.split()
# bigrams = ngrams(tokenized, 2)
# print(list(bigrams))

# from sklearn.feature_extraction.text import TfidfVectorizer
# document = ["I like this movie, it's funny.", 'I hate this movie.', 'This was awesome! I like it.', 'Nice one. I love it.']
# tfidf_vectorizer = TfidfVectorizer()
# values = tfidf_vectorizer.fit_transform(document)
# # Show the Model as a pandas DataFrame
# feature_names = tfidf_vectorizer.get_feature_names()
# data = pd.DataFrame(values.toarray(), columns = feature_names)
# print(data)


from sklearn.feature_extraction.text import HashingVectorizer
document = ["I like this movie, it's funny.", 'I hate this movie.', 'This was awesome! I like it.', 'Nice one. I love it.']
vectorizer = HashingVectorizer(n_features=2**4)
values = vectorizer.fit_transform(document)
# feature_names = vectorizer.get_feature_names()
data = pd.DataFrame(values.toarray(),)
print(data)