import numpy as np
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import os
import cv2
import itertools
from time import time
import psutil
import sys
import csv

from shared.image_utils import ImageUtils
from shared.patch_utils import PatchUtils



from data_generators.csv_pixel_test_data_generator import CsvPixelTestDataGenerator



class CsvNnDataGenerator(CsvPixelTestDataGenerator):
        
    def __init__(self, data_size=10,
                 shuffle=False, test_starting_index=None, csv_path=None, img_refs=None, patch_shape=(None, None)):
        
        super().__init__(data_size, test_starting_index = test_starting_index,csv_path = csv_path,
                   img_refs = img_refs
                   )
        self.patch_shape = patch_shape

    def __getitem__(self, index):
        
        x_list , y_list = super().__getitem__(index)
        num_indicators = len(x_list[0][0])
        x = []
        y = []
        for _x, _y, img_ref in zip(x_list, y_list, self.img_refs):
            _y = np.array(_y).reshape(img_ref.img_width, img_ref.img_height)
            y.append(cv2.resize(_y, (self.patch_shape, self.patch_shape)))
            x.append(self._reshape_resize_x(np.array(_x), img_ref))
            
        x = np.array(x).reshape(-1, self.patch_shape, self.patch_shape, num_indicators )
        y = np.array(y).reshape(-1, self.patch_shape, self.patch_shape, 1)
        return x, y
    
    def _reshape_resize_x(self, x, img_ref):
        x_list = []
        x = x.reshape(x.shape[1], img_ref.img_width, img_ref.img_height)
        for _x in x:
           x_list.append(cv2.resize(_x, (self.patch_shape, self.patch_shape))) 
        return np.array(x_list)
        