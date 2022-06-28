# Convolutional Neural Network on cars
# using Stanford's car image dataset with 16,185 images labeled into 196 classes
# Alex Trujillo 05/09/2020

from keras import layers
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import random

# np.random.seed(314)
# tf.random.set_seed(314)
# random.seed(314)

datadir = r"C:\Users\Alex\PycharmProjects\Convolutional_Neural_Network\data\train"
categories = ["car", "truck"]
for cateogory in categories:
    path = os.path.join(datadir, cateogory)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
img_size = 60
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()

data_trainn = []
def create_data_train():
    for cateogory in categories:
        path = os.path.join(datadir, cateogory)
        class_num = categories.index(cateogory)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                data_trainn.append([new_array, class_num])
            except Exception as e:
                pass
create_data_train()
random.shuffle(data_trainn)

X = []
y = []

# form training images and labels
for features, label in data_trainn:
    X.append(features)
    y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# plot 4 images as gray scale images to show them
plt.subplot(221) # 221 means 2x2 subplot and 1st entry
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()

# gets all the training images to img_size x img_size grayscale images, use 3 for RGB
X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X.astype('float32')/255

# build the model
model = models.Sequential()
# 3x3 + 1 filter + bias -> 10 X 32 == 320 parameters in layer
model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(img_size, img_size, 1), padding='Same'))
model.add(layers.MaxPooling2D((2, 2))) # decreases size of feature maps by factor of 2 in each dimension

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='Same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='Same'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten()) # vectorizes the (3, 3, 64) outputs into shape (576,)

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X, y, epochs=15, batch_size=32, validation_split=0.25)
model.fit(X, y, epochs=20, batch_size=20, validation_split=0.2)
test_lost, test_acc, = model.evaluate(X, y)
print(f'Test accuracy = {test_acc*100:.2f}%')