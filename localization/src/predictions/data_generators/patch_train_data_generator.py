import numpy as np
import random
import os
import cv2
import itertools
from time import time
import psutil
import sys

from data_generators.train_data_generator import TrainDataGenerator

class PatchTrainDataGenerator(TrainDataGenerator):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                 shuffle=False, patches_path="", patch_shape=128, num_patches=8,patch_tuning=False):
        super().__init__(batch_size, indicator_directories, 
                 shuffle, patches_path, patch_shape, num_patches, patch_tuning = patch_tuning)
        
    def __getitem__(self, index):
        indicator_imgs, target_imgs = super().__getitem__(index) 
        x = np.array(indicator_imgs).reshape(-1, self.patch_shape, self.patch_shape, len(self.indicator_directories) )
        y = np.array(target_imgs).reshape(-1, self.patch_shape, self.patch_shape, 1)
        return x,y