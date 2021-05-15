import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

TRAIN_DIR = r'D:\Documents\GitHub\opencv-computer-vision\code\dataset_cd\train'
TEST_DIR = r'D:\Documents\GitHub\opencv-computer-vision\code\dataset_cd\test1'
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 110

train_data = []
test_data = []

for file in os.listdir(TRAIN_DIR):
    category = file.split('.')[0]
    img_path = os.path.join(TRAIN_DIR, file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    train_data.append([img, CATEGORIES.index(category)])

for file in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    test_data.append(img)


random.shuffle(train_data)

train_X = []
train_y = []

for features, labels in train_data:
    train_X.append(features)
    train_y.append(labels)

X = np.array(train_X)
y = np.array(train_y)

X = X/255

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, input_shape=(IMG_SIZE, IMG_SIZE, 3),))

model.add(Dense(2, activation='softmax'))
print('hdfgd')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_y, epochs=5, validation_data=test_data)




