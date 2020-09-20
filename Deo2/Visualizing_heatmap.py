from keras.models import load_model
import matplotlib.pyplot as plt
from keras import models
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
import cv2
def dole():    
    layer_activation=activations[7]   
    layer_name="max pool"    
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()      
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')  
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
model = load_model('cats_and_dogs_small_2.h5')
model.summary() 
img_path = 'C:/Users/Korisnik/Pop/Pmf/Pmf/Trece godina/Semestar II/Neuronske Mreze/Projekat/NN/dogs cats/dogs-vs-cats/train/example/test/cats/cat.1504.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[7]
heatmap = np.mean(first_layer_activation[0], axis=-1)
# Da se izbace negativne
heatmap = np.maximum(heatmap, 0)
# Normalizacija i dobijanje svetlijih nijansi
heatmap /= np.max(heatmap)

img = cv2.imread(img_path)
# Da heatmap bude dimenzija kao originalna, posto hocemo da je prelepimo
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
plt.imshow(cv2.cvtColor(np.clip(heatmap, 0, 255).astype('uint8'), cv2.COLOR_BGR2RGB))
plt.show()
# Da bude prozirna
heatmap=heatmap * 0.4
superimposed_img = heatmap + img
plt.imshow(cv2.cvtColor(np.clip(superimposed_img, 0, 255).astype('uint8'), cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('C:/Users/Korisnik/Pop/Pmf/Pmf/Trece godina/Semestar II/Neuronske Mreze/SLike/SLika 18.jpg', superimposed_img)
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
dole()


