import numpy as np
import pandas as pd
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
                 shuffle=False, data=None, img_refs=None, batch_size=None):
        self.data_size = data_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data= data
        self.img_refs = img_refs
        self.on_epoch_end()
        
    def __getitem__(self, index):

        #create a numpy array of the shape [data_size, 1]
        #from img ref get the image id and use it as the keys
        #add data to the above array. do an image id check 
        #on the data that u are reading from the csv file
#         x = []
#         y = []
#         ids = [None]*self.data_size
#         for i in range(self.data_size):
#             x.append([])
#             y.append([])
#         header = None
#         with open(self.csv_path, 'r') as f:
#             reader = csv.reader(f, delimiter=',')
#             header = next(reader)
#             index_image_id = header.index('image_id')
#             for row in reader:
#                 index = next((i for i, img_ref in enumerate(self.img_refs) if img_ref.probe_file_id == row[index_image_id] ), None ) 
#                 if index == None:
#                     continue    
#                 x[index].append(np.array(row[:-3]).astype(np.float))
#                 y[index].append(float(row[-3]))
#                 ids[index] = row[index_image_id]
#                 if y[-1] != None and len(y[-1]) == (self.img_refs[-1].img_width * self.img_refs[-1].img_height): 
#                     break
#          
# #         exclude = ['image_id', 'pixel_id', 'label']
# #         x_cols = [i for i,x in enumerate(header) if x not in exclude]
#         index = [i for i, _y in enumerate(y) if len(_y) == 0 ]
#  
#         for i in sorted(index, reverse=True):
#             try:
#                 del y[i]
#                 del x[i]
#                 del ids[i]
#                 del self.img_refs[i]
#             except IndexError as e:
#                 a = 1
#         a= np.concatenate(x, axis=0)
#         b= np.concatenate(y, axis=0)
#         return x, y, ids
#         self.csv_path = 'C:/MyFiles/Study/research/localization/data/original/MFC18_EvalPart1/csv_data/test.csv'
        #df = pd.read_csv(self.csv_path)
#         return df
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        img_refs = self.img_refs[starting_index:ending_index]
        df = self.data
        exclude = ['image_id', 'pixel_id', 'label']
        filtered_df = df[df['image_id'].isin( [i.probe_file_id for i in img_refs] )]
        exclude = ['image_id', 'pixel_id', 'label']
         
        x_cols = [x for x in filtered_df.columns if x not in exclude]
        grouped = filtered_df.groupby('image_id')
        x = []
        y = []
        ids = []
        for image_id, group in grouped:
            x.append(np.array(group[x_cols].values))
            y.append(np.array(group['label'].values))
            ids.append(image_id)
 
#         img_refs_to_remove = [i for i, item in enumerate(self.img_refs) if item.probe_file_id not in grouped.groups]
#         for i in sorted(img_refs_to_remove, reverse=True):
#             try:
#                 print(f'deleted img with id {self.img_refs[i].probe_file_id} at index {i}')
#                 del self.img_refs[i]
# #                 del image_ids[i]
#             except IndexError as e:
#                 a = 1
         
        if len(x) != len(img_refs):
            for image_id, group in grouped:
                probe_file_ids_to_remove = [item.probe_file_id for i, item in enumerate(img_refs) if item.probe_file_id not in grouped.groups]
                for i,item in sorted(self.img_refs, reverse=True):
                    if item.probe_file_id not in probe_file_ids_to_remove:
                        continue
                    try:
                        print(f'deleted img with id {self.img_refs[i].probe_file_id} at index {i}')
                        del self.img_refs[i]
#                     del image_ids[i]
                    except IndexError as e:
                        a = 1
        
        return np.array(x),np.array(y) , ids

    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return self.data_size
