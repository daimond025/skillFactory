from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# The local path to our target image
img_path = '../5/data/cats_and_dogs_small/maxresdefault.jpg'


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v3 import mobilenet_v3

model = VGG16(weights='imagenet')
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=10)[0])

# model = load_model('../../deep-learning/5/cats_and_dogs_small_2.h5')
# img_path = '../../deep-learning/5/data/cats_and_dogs_small/test/cats/cat.1700.jpg'
#
# img = image.load_img(img_path, target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.
#
# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#
# activations = activation_model.predict(img_tensor)
#
# import keras
#
# layer_names = []
# for layer in model.layers[:8]:
#     layer_names.append(layer.name)
# images_per_row = 16
#
# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = layer_activation.shape[-1]
#
#     size = layer_activation.shape[1]
#
#     # We will tile the activation channels in this matrix
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#
#     # We'll tile each filter into this big horizontal grid
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = layer_activation[0,
#                             :, :,
#                             col * images_per_row + row]
#             # Post-process the feature to make it visually palatable
#             channel_image -= channel_image.mean()
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size: (col + 1) * size,
#             row * size: (row + 1) * size] = channel_image
#
#     # Display the grid
#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
# plt.show()
# print(layer_names)
# exit()
#
# images_per_row = 16

