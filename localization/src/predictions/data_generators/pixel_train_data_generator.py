import numpy as np
from data_generators.train_data_generator import TrainDataGenerator
import math

class PixelTrainDataGenerator(TrainDataGenerator):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                 shuffle=False, patches_path="", patch_shape=128, num_patches=8):
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        super().__init__(batch_size, indicator_directories, 
                 shuffle, patches_path, patch_shape, num_patches)
        
    def __getitem__(self, index):
        starting_index = index*self.batch_size
        ending_index = (index+1)* self.batch_size

        indicator_imgs, target_imgs = super().__getitem__(index)
        x = []
        y = []
        
        indicator_imgs = np.array(indicator_imgs).reshape(-1, len(indicator_imgs), self.patch_shape, self.patch_shape)
        
        for indicators, target in zip(indicator_imgs, target_imgs):
            for (i,j) , t in np.ndenumerate(target):
                y.append(t)
                _x = []
                for ind in indicators:
                    _x.append(ind[i,j])
                x.append(_x)
        return np.array(x),np.array(y)