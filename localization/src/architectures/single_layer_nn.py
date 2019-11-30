import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('WARN')

class SingleLayerNN():
    
    def get_model(self, image_size, num_indicators, layers=1):

        inputs = keras.layers.Input((image_size, image_size, num_indicators))
        
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(inputs)
        outputs = outputs[:,:,:,-1]
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        return model