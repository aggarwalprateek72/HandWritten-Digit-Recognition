# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:09:45 2019

@author: Prateek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train= pd.read_csv('train.csv', header=None, skiprows=1)
dataset_test= pd.read_csv('test.csv', header=None, skiprows=1)

X_train= dataset_train.iloc[:, 1:].values
y_train= dataset_train.iloc[:, 0].values

X_test= dataset_test.iloc[:, :].values

from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

X_train= X_train.reshape(X_train.shape[0],*(28,28,1))
X_test= X_test.reshape(X_test.shape[0],*(28,28,1))

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

classifier= Sequential()

classifier.add(Conv2D(32,3,3, input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(Dense(output_dim=10, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=5000, epochs=20, verbose=1, validation_split=0.2)