from keras.models import Sequential 
from keras.layers import Dense, Activation 

class Lr():
    
    def get_model(self, image_size, num_indicators, layers=1):
        
        model = Sequential()
        model.add(Dense(1, input_dim=num_indicators, activation='sigmoid'))
        model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
        return model