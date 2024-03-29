from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np

from keras import regularizers


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# TODO
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))
l2_model_val_loss = dpt_model_hist.history['val_loss']

print(l2_model_val_loss)
# TODO РЕГУЛЯРИЗАЦИЯ
# l2_model = models.Sequential()
# l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
#                           activation='relu', input_shape=(10000,)))
# l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
#                           activation='relu'))
# l2_model.add(layers.Dense(1, activation='sigmoid'))
#
#
# l2_model.compile(optimizer='rmsprop',
#                  loss='binary_crossentropy',
#                  metrics=['acc'])
#
#
# l2_model_hist = l2_model.fit(x_train, y_train,
#                              epochs=20,
#                              batch_size=512,
#                              validation_data=(x_test, y_test))
#
# l2_model_val_loss = l2_model_hist.history['val_loss']

print(l2_model_val_loss)
# original_model = models.Sequential()
# original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# original_model.add(layers.Dense(16, activation='relu'))
# original_model.add(layers.Dense(1, activation='sigmoid'))
#
# original_model.compile(optimizer='rmsprop',
#                        loss='binary_crossentropy',
#                        metrics=['acc'])
#
# smaller_model = models.Sequential()
# smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
# smaller_model.add(layers.Dense(4, activation='relu'))
# smaller_model.add(layers.Dense(1, activation='sigmoid'))
#
# smaller_model.compile(optimizer='rmsprop',
#                       loss='binary_crossentropy',
#                       metrics=['acc'])
#
#
# original_hist = original_model.fit(x_train, y_train,
#                                    epochs=20,
#                                    batch_size=512,
#                                    validation_data=(x_test, y_test))
#
#
# smaller_model_hist = smaller_model.fit(x_train, y_train,
#                                        epochs=20,
#                                        batch_size=512,
#                                        validation_data=(x_test, y_test))
#
# epochs = range(1, 21)
# original_val_loss = original_hist.history['loss']
# smaller_model_val_loss = smaller_model_hist.history['loss']
#
# print(original_val_loss)
# print(smaller_model_val_loss)
