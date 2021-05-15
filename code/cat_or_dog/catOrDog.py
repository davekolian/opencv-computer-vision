import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle
import csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

TRAIN_DIR = r'D:\Documents\GitHub\opencv-computer-vision\code\dataset_cd\train'
TEST_DIR = r'D:\Documents\GitHub\opencv-computer-vision\code\dataset_cd\test1'
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 100

train_data = []
test_data = []

if 'model.h5' not in os.listdir():
    if 'train.csv' in os.listdir():
        print("Loading files")
        reader = csv.reader('train.csv', delimiter=" ")
        train_data = [row for row in reader]
        test_data = open('test.csv', 'r').read()
        print("Done loading")
    else:
        for file in os.listdir(TRAIN_DIR):
            category = file.split('.')[0]
            img_path = os.path.join(TRAIN_DIR, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            print("1. reading: {}".format(img_path))
            train_data.append([img, CATEGORIES.index(category)])

    #     for file in os.listdir(TEST_DIR):
    #         img_path = os.path.join(TEST_DIR, file)
    #         img = cv2.imread(img_path)
    #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    #         print("2. reading: {}".format(img_path))
    #
    #         test_data.append(img)
    #
    #
    #
    #
    # test_data = np.array(test_data)
    # test_data = test_data / 255


    random.shuffle(train_data)

    train_X = []
    train_y = []


    for features, labels in train_data:
        train_X.append(features)
        train_y.append(labels)

    X, test_X, y, test_y = train_test_split(train_X, train_y, test_size=0.1, random_state=1)

    X = np.array(X)
    y = np.array(y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    X = X / 255

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(128, input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=5, batch_size=64)

    model.save('model.h5')

    #model = tf.keras.models.load_model('model.h5')
    res = model.evaluate(test_X, test_y)

    print(res)



