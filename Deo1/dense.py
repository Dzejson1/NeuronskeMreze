from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.preprocessing import image
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

img1=test_images[0]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=1, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

img=test_images[0]

predict_result =model.predict(img.reshape((1, 28 * 28)))

print('predict_acc:', predict_result)

print('The predicted digit is: ', predict_result.argmax())

plt.imshow(img1)
plt.show()






