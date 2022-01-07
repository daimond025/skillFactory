import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import fetch_20newsgroups
categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)


X_train = newsgroups_train.data
y_train = newsgroups_train.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)


lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
print(lr.coef_.shape)
exit()

X_test_vec = X_train_vec = vectorizer.transform(X_test)
y_pred = lr.predict(X_test_vec)

print(X_train_vec.shape)
