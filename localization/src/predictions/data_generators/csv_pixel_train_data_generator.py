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
                 shuffle=False, csv_path=None, img_refs = None):
        self.data_size = data_size
        self.shuffle = shuffle
        self.csv_path = csv_path
        self.img_refs = img_refs
        self.on_epoch_end()
        
    def __getitem__(self, index):
        starting_index = index * self.data_size
        ending_index = (index + 1) * self.data_size
        
#         data_size = ending_index - starting_index
#         
#         x = []
#         y = []
#         header = None
#         with open(self.csv_path, 'r') as f:
#             reader = csv.reader(f, delimiter=',')
#             header = next(reader)
#             counter = 0
#             image_id = None
#             index_image_id = header.index('image_id')
#             for row in reader:
#                 x.append(row[:-3])
#                 y.append(row[-3])
#                 if row[index_image_id] != image_id:
#                     image_id = row[index_image_id]
#                     counter +=1
#                 if counter is ending_index:
#                     break
#         
# #         exclude = ['image_id', 'pixel_id', 'label']
# #         x_cols = [i for i,x in enumerate(header) if x not in exclude]
#         return np.array(x).astype(np.float), np.array(y).astype(np.float)

        x = []
        y = []
        for i in range(self.data_size):
            x.append([])
            y.append([])
        header = None
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            index_image_id = header.index('image_id')
            for row in reader:
                index = next((i for i, img_ref in enumerate(self.img_refs) if img_ref.probe_file_id == row[index_image_id] ), None ) 
                if index == None:
                    continue    
                try:
                    x[index].append(np.array(row[:-3]).astype(np.float))
                    y[index].append(float(row[-3]))
                    if y[-1] != None and len(y[-1]) == (self.img_refs[-1].img_width * self.img_refs[-1].img_height): 
                        break
                except:
                    print(index)
#         exclude = ['image_id', 'pixel_id', 'label']
#         x_cols = [i for i,x in enumerate(header) if x not in exclude]
        index = [i for i, _y in enumerate(y) if len(_y) == 0 ]
        for i in sorted(index, reverse=True):
            try:
                del y[i]
                del x[i]
                del self.img_refs[i]
            except:
                a=1
        a=  np.concatenate(x, axis=0)
        b= np.concatenate(y, axis=0)
        return a,b
#         return np.array(x).reshape(-1,27), np.array(y).reshape(-1)
#         return np.array(x).reshape(-1,27), np.array(y).reshape(-1)

    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return self.data_size
