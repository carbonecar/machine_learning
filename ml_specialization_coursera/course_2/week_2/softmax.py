import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
#matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

model= Sequential([
        Dense(units=25,activation='sigmoid'),
        Dense(units=15,activation='sigmoid'),
        Dense(units=10,activation='softmax')
    ])

model.compile(loss=SparseCategoricalCrossentropy())

x = tf.random.normal([100,5])
y = tf.random.uniform([100],maxval=10,dtype=tf.int32)

model.fit(x,y,epochs=100)
model.summary()


# one better implemetation is: 
model= Sequential([
        Dense(units=25,activation='relu'),
        Dense(units=15,activation='relu'),
        Dense(units=10,activation='linear')
    ])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

model.fit(x,y,epochs=100)
logits=model(x)
f_x= tf.nn.sigmoid(logits)


def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)


plt.close("all")
plt_softmax(my_softmax)

