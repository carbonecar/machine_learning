import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
import matplotlib.pyplot as plt
from CustomNeuralNetwork import CustomNeuralNetwork

x=np.array([[2000.0,17.0]])
layer_1= Dense(units=3,activation='sigmoid')

a1=layer_1(x)
layer_2=Dense(units=1,activation='sigmoid')
a2=layer_2(a1)

model= Sequential([layer_1,layer_2])



if(a2>0.5):
    print('The model predicts that the house will be sold')
else:
    print('The model predicts that the house will not be sold')
    

cnn=CustomNeuralNetwork()
cnn.addLayer(Dense(units=3,activation='sigmoid')).addLayer(Dense(units=1,activation='sigmoid'))

print(x)
cnn.predict(x)

print(x)

Dense(units=3,activation='sigmoid')


