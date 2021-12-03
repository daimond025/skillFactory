import keras
from keras.datasets import imdb
from keras.datasets import reuters
from keras import optimizers
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np


#TODO 3.5
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

#TODO 3.4
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#
# # word_index = imdb.get_word_index()
# # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# # decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.  # set specific indices of results[i] to 1s
#     return results
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
#
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')
#
#
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
#
# model.fit(x_train, y_train, epochs=4, batch_size=512)
#
# results = model.evaluate(x_test, y_test)
# print(results)