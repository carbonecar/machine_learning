import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt


model = Sequential( [
    Dense(units=25,activation='sigmoid'),
    Dense(units=15,activation='sigmoid'),
    Dense(units=1,activation='sigmoid')
])

model.compile(optimizer='adam',loss=BinaryCrossentropy(),metrics=['accuracy'])

