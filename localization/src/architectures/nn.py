import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.regularizers import l1
from keras.optimizers import SGD
tf.get_logger().setLevel('WARN')

class Nn():
    
    def get_model(self, image_size, num_indicators, padding="same", strides=1, config=None):
        
        layers = config['nn_layers']

        model = Sequential()
        model.add(Dense(num_indicators, input_dim=num_indicators, activation='linear', activity_regularizer=l1(config['regularization'])))
        model.add(Activation('sigmoid'))
        
        for i in range(layers):
            model.add(Dense(int(num_indicators/(i+1)*2),
                        activation='linear', activity_regularizer=l1(config['regularization'])))           
            model.add(Activation('sigmoid'))
        
        model.add(Dense(1, activation='linear', activity_regularizer=l1(config['regularization'])))
        model.add(Activation('sigmoid'))
        
#         opt = keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.9, beta_2=0.999, amsgrad=False)
#         opt= SGD(lr=config['learning_rate'], momentum=0.9, decay=1e-2/config['epochs'])
        opt= SGD(lr=config['learning_rate'], momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=opt,  metrics=['accuracy'])        
        
        return model

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