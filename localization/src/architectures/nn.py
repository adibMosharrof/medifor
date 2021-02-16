import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Model
from keras.models import Sequential 
from keras.layers import UpSampling2D, Conv2D, Activation, LeakyReLU, BatchNormalization,Input,Conv2DTranspose,Dropout
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate,add
from keras.regularizers import l1
from keras.optimizers import SGD
tf.get_logger().setLevel('WARN')

class Nn():
    
#     def get_model(self, image_size, num_indicators, padding="same", strides=1, config=None):
        

        
        
#         layers = config['nn_layers']
# 
#         inputs = keras.layers.Input((image_size, image_size, num_indicators))
#         filters = 1
#         kernel_size = (3,3)
#         c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(inputs)
#         c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#         outputs = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
# #         outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(inputs)
#         outputs = outputs[:,:,:,-1]
#         
#         model = keras.models.Model(inputs, outputs)
#         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
#         return model
#     def get_model(self, image_size, num_indicators, padding="same", strides=1, config=None):
    def get_model(self, image_size, num_indicators, n_filters=8, dropout=0.2, batchnorm=False, config=None, strides=1, padding="same"):
        """Function to define the UNET Model"""
        # Contracting Path
        
        input_img = Input((image_size,image_size,num_indicators), name='img')
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = self.conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path
        u6 = UpSampling2D()(c5)
        u6 = Conv2D(filters = n_filters *8, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u6)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = UpSampling2D()(c6)
        u7 = Conv2D(filters = n_filters *4, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u7)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = UpSampling2D()(c7)
        u8 = Conv2D(filters = n_filters *2, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u8)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = UpSampling2D()(c8)
        u9 = Conv2D(filters = n_filters *1, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u9)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
#         print(model.summary())
        return model

    def conv2d_block(self, input_tensor, n_filters, kernel_size = 3, batchnorm = False):
        '''returns a block of two 3x3 convolutions, each  followed by a rectified linear unit (ReLU)'''
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x