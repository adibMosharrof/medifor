import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.regularizers import l1
from keras.optimizers import SGD
tf.get_logger().setLevel('WARN')

class NnImg():
    
    def get_model(self, image_size, num_indicators, padding="same", strides=1, config=None):
        

        inputs = keras.layers.Input((image_size, image_size, num_indicators))
        filters = 1
        kernel_size = (3,3)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(inputs)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        outputs = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#         outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(inputs)
        outputs = outputs[:,:,:,-1]
        
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        return model