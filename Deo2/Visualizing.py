from keras.models import load_model

import matplotlib.pyplot as plt

from keras import models

import matplotlib.pyplot as plt

import keras

from keras.preprocessing import image
import numpy as np


# model = load_model('cats_and_dogs_small_1.h5')
model = load_model('cats_and_dogs_small_2.h5')

model.summary() 

img_path = 'C:/Users/Korisnik/Pop/Pmf/Pmf/Trece godina/Semestar II/Neuronske Mreze/Projekat/NN/dogs cats/dogs-vs-cats/train/train/cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
print('a',img_tensor.shape,'a')
# Iz dve u tri dimenzije
img_tensor = np.expand_dims(img_tensor, axis=0) 
print('b',img_tensor.shape,'b')
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    print('asd',layer_activation.shape[-1],'asd');
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
    
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # print('asdasd',display_grid.shape,'asdasd');
    scale = 1. / size
    # Display grid je (y,x) a treba kontra (x,y) pa je zato ovo dole
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()