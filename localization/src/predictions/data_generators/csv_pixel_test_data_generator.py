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


class CsvPixelTestDataGenerator(Sequence):
    
    def __init__(self, data_size=10,
                 shuffle=False, test_starting_index=None, csv_path=None):
        self.data_size = data_size
        self.test_starting_index = test_starting_index
        self.shuffle = shuffle
        self.csv_path = csv_path
        self.on_epoch_end()
        
    def __getitem__(self, index):
        ending_index = self.test_starting_index + self.data_size
        starting_index = self.test_starting_index
        
        x = []
        y = []
        header = None
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            counter = 0
            image_id = None
            index_image_id = header.index('image_id')
            _x = []
            _y = []
            for row in reader:
                if counter >= starting_index:
                    _x.append(row[:-3])
                    _y.append(row[-3])
                if row[index_image_id] != image_id:
                    image_id = row[index_image_id]
                    counter +=1
                    if len(_y) > 0:
                        x.append(np.array(_x).astype(np.float))
                        y.append(np.array(_y).astype(np.float))
                        _x = []
                        _y = []
                if counter is ending_index:
                    break
        
#         exclude = ['image_id', 'pixel_id', 'label']
#         x_cols = [i for i,x in enumerate(header) if x not in exclude]
        
        return np.array(x), np.array(y)
    
    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return self.data_size
