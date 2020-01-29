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


class CsvPixelTrainDataGenerator(Sequence):
    
    def __init__(self, data_size=10,
                 shuffle=False, csv_path=None):
        self.data_size = data_size
        self.shuffle = shuffle
        self.csv_path = csv_path
        self.on_epoch_end()
        
    def __getitem__(self, index):
        starting_index = index * self.data_size
        ending_index = (index + 1) * self.data_size
        
        data_size = ending_index - starting_index
        
        x = []
        y = []
        header = None
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            counter = 0
            image_id = None
            index_image_id = header.index('image_id')
            for row in reader:
                x.append(row[:-3])
                y.append(row[-3])
                if row[index_image_id] != image_id:
                    image_id = row[index_image_id]
                    counter +=1
                if counter is ending_index:
                    break
        
#         exclude = ['image_id', 'pixel_id', 'label']
#         x_cols = [i for i,x in enumerate(header) if x not in exclude]
        return np.array(x).astype(np.float), np.array(y).astype(np.float)
    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return self.data_size
