import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('WARN')

class SingleLayerNN():
    
    def get_model(self, image_size, num_indicators, layers=1,padding="same", strides=1):

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