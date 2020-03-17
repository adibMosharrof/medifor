import os
import socket
import sys
sys.path.append('..')
import json
from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import re
import math
import logging
import pandas as pd
from typing import overload, List
from pythonlangutil.overload import Overload, signature
from shared.path_utils import PathUtils


class ImgRefBuilder:
    image_ref_csv_path = None
    
    def __init__(self, image_ref_csv_path):
        self.image_ref_csv_path =  image_ref_csv_path

    @staticmethod
    def get_img_ref_from_patch_ref(patch_refs):
        img_refs = []
        for patch_ref in patch_refs:
            img_refs.append(ImgRefs(
                patch_ref.probe_file_id, patch_ref.probe_mask_file_name))
        return img_refs    
        
    @Overload
    @signature("int")
    def get_img_ref(self, num_rows) :
        return self.get_img_ref(0, num_rows)            
    
    @get_img_ref.overload
    @signature("list")
    def get_img_ref(self, indices):
        #need to change this if indices are not all in order
        #that is if they have random values in them
        return self.get_img_ref(indices[0], indices[-1])
            
    @get_img_ref.overload
    @signature("int","int")
    def get_img_ref(self, starting_index, ending_index):
        if ending_index is -1:
            ending_index = math.inf
        rows = []
#         with open(self.image_ref_csv_path, 'r') as f:
#             reader = csv.reader(f, delimiter='|')
#             headers = next(reader)
#             if not len(headers) > 1:
#                 reader = csv.reader(f, delimiter=',')
#                 headers = next(reader)
#             counter = 0
#             for row in reader:
#                 #only selected images that have a reference(we are only scoring the manipulated images)
#                 if(row[4] == ''):
#                     continue
#                 if counter>= starting_index and counter <ending_index:
#                     rows.append([row[1], row[4]])
#                 counter +=1
#                 if counter is ending_index:
#                     break
        data = pd.read_csv(self.image_ref_csv_path, sep="|")
        print(f'length of columns {len(data.columns)}')
        print(f'columns {data.columns}')
        if not len(data.columns) > 1:
            data = pd.read_csv(self.image_ref_csv_path, sep=",")
        rows  = data[data['ProbeMaskFileName'].notnull()]
        rows = rows[['image_id','ProbeMaskFileName']]
        rows.sort_values(by=['image_id'])
        rows = rows.to_numpy()[starting_index:ending_index]
#         rows = np.array(rows)        
        sys_masks = rows[:,0]
        ref_masks = list(map(lambda x:self.extract_ref_mask_file_name(x) ,rows[:,1]))
        img_refs = []
        for i in range(len(sys_masks)):
            img_refs.append(ImgRefs(sys_masks[i], ref_masks[i]))
        img_refs.sort(key=lambda x:x.probe_file_id)
        return img_refs
        
    def extract_ref_mask_file_name(self, text):
        return re.search("(?<=reference\/manipulation-image\/mask\/).*(?=.ccm.png)", text).group()

    @staticmethod
    def add_image_width_height(img_refs, config):
        index_csv_path = PathUtils.get_index_csv_path(config)
        with open(index_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            img_id_col_index = headers.index('image_id')
            
            img_orig_height_col_index = headers.index('image_height_original')
            img_orig_width_col_index = headers.index('image_width_original')
            img_height_col_index = headers.index('image_height')
            img_width_col_index = headers.index('image_width')
            counter = 0
            for row in reader:
                img_ref = next((i for i in img_refs if i.probe_file_id == row[img_id_col_index]),None)
                if img_ref == None :
                    continue
                img_ref.img_orig_height = int(float(row[img_orig_height_col_index]))
                img_ref.img_orig_width = int(float(row[img_orig_width_col_index]))
                img_ref.img_height = int(float(row[img_height_col_index]))
                img_ref.img_width = int(float(row[img_width_col_index]))
                counter +=1
                if counter == len(img_refs):
                    break

class ImgRefs:
    probe_file_id = None
    probe_mask_file_name = None
    img_orig_height=None
    img_orig_width = None
    img_height=None
    img_width = None
        
    def __init__(self,sys, ref):
        self.probe_file_id = sys
        self.probe_mask_file_name = ref    

        
    