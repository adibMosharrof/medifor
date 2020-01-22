import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('WARN')

class UNet():
    
    def __init__(self):
        a = 1
    
    def down_block(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    def up_block(self,x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
    
    def bottleneck(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
    
    def get_model(self, image_size, num_indicators, layers=3):
        f = np.around(np.geomspace(image_size/2**(layers-1), image_size, num=layers)).astype('uint8')
#         f = [8, 16, 32, 64, 128]
        inputs = keras.layers.Input((image_size, image_size, num_indicators))
        
        p=[None]* layers
        p[0] = inputs
        c=[None]* layers
        for i in range(0,layers-1):
            c[i+1], p[i+1] = self.down_block(p[i], f[i])

        bn = self.bottleneck(p[layers-1], f[layers-1])
        u=[None]* layers
        u[0] = bn
        for i in range(0,layers-1):
            u[i+1] = self.up_block(u[i], c[layers-i-1], f[layers-i-2])
            
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u[layers-1])
        outputs = outputs[:,:,:,-1]
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        return model
        
        
        c1, p1 = self.down_block(p0, f[0]) #128 -> 64
        c2, p2 = self.down_block(p1, f[1]) #64 -> 32
        c3, p3 = self.down_block(p2, f[2]) #32 -> 16
        c4, p4 = self.down_block(p3, f[3]) #16->8
        
        bn = self.bottleneck(p4, f[4])
        
        u1 = self.up_block(bn, c4, f[3]) #8 -> 16
        u2 = self.up_block(u1, c3, f[2]) #16 -> 32
        u3 = self.up_block(u2, c2, f[1]) #32 -> 64
        u4 = self.up_block(u3, c1, f[0]) #64 -> 128
        
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        outputs = outputs[:,:,:,-1]
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        return model