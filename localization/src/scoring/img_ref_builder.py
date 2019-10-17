import os
import socket
import sys
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
from typing import overload, List
from pythonlangutil.overload import Overload, signature

class ImgRefBuilder:
    image_ref_csv_path = None
    
    def __init__(self, image_ref_csv_path):
        self.image_ref_csv_path =  image_ref_csv_path

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
        with open(self.image_ref_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            headers = next(reader)
            counter = 0
            for row in reader:
                #only selected images that have a reference(we are only scoring the manipulated images)
                if(row[4] == ''):
                    continue
                if counter>= starting_index and counter <ending_index:
                    rows.append([row[1], row[4]])
                counter +=1
                if counter is ending_index:
                    break
                
        rows = np.array(rows)        
        sys_masks = rows[:,0]
        ref_masks = list(map(lambda x:self.extract_ref_mask_file_name(x) ,rows[:,1]))
        img_refs = []
        for i in range(len(sys_masks)):
            img_refs.append(ImgRefs(sys_masks[i], ref_masks[i]))
        return img_refs
        
    def extract_ref_mask_file_name(self, text):
        return re.search("(?<=reference\/manipulation-image\/mask\/).*(?=.ccm.png)", text).group()

class ImgRefs:
    sys_mask_file_name = None
    ref_mask_file_name = None
        
    def __init__(self,sys, ref):
        self.sys_mask_file_name = sys
        self.ref_mask_file_name = ref    

        
    