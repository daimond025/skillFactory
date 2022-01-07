import os
import numpy as np
from collections import Counter

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.datasets import imdb

from tensorflow.keras import optimizers
from keras import preprocessing
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# ПЕРЕМЕННЫЕ
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000


imdb_dir = "./aclImdb"
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)


labels = np.asarray(labels)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# загрузка представления Glove
glove_dir = './glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

print(np.shape(embedding_matrix))
print(embedding_matrix)
exit()


# max_feature = 10000
# maxlen = 20
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
#
# x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
#
# model = Sequential()
# model.add(Embedding(10000, 8, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.Adam(lr=2e-5),
#               metrics=['acc'])
#
# model.summary()
#
# history = model.fit(
#     x_train, y_train,
#     epochs=30,
#     batch_size=32,
#     validation_split=0.2)
#
# print(history.history['acc'])
