import numpy as np
import keras
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import os
import cv2
import itertools
from time import time
import psutil
import sys


from shared.image_utils import ImageUtils
from shared.patch_utils import PatchUtils

class PatchTrainDataGenerator(Sequence):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                 shuffle=False, patches_path="", patch_shape=128, num_patches=8):
        self.batch_size = batch_size
        self.indicator_directories = indicator_directories
        self.shuffle = shuffle
        self.patches_path = patches_path
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.on_epoch_end()
        
    def __getitem__(self, index):
        starting_index = index*self.batch_size
        ending_index = (index+1)* self.batch_size
        
        indicator_imgs = []
        for indicator_name in self.indicator_directories:
            indicator_path = self.patches_path + indicator_name
            indicator_patches = self._read_images_from_directory(indicator_path, starting_index, ending_index)
            indicator_imgs.append(indicator_patches)
            
        target_imgs = []
        target_imgs_path = self.patches_path+ 'target_image'
        target_imgs = self._read_images_from_directory(target_imgs_path, starting_index, ending_index)
        
        x = np.array(indicator_imgs).reshape(-1, self.patch_shape, self.patch_shape, len(self.indicator_directories) )
        y = np.array(target_imgs).reshape(-1, self.patch_shape, self.patch_shape, 1)
        
        return x,y
        
    def _read_images_from_directory(self, dir_path, starting_index, ending_index):
        img_names = os.listdir(dir_path)[starting_index:ending_index]
        imgs = []
        for name in img_names:
            img_path = os.path.join(dir_path, name)
            img = ImageUtils.read_image(img_path)
            imgs.append(img)
        return imgs
    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return int(np.ceil(self.num_patches/float(self.batch_size)))