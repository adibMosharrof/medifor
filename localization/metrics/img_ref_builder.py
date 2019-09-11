import os
import sys
from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import re
from medifordata import MediforData

class ImgRefBuilder:
    data_path = "../data/"
    ref_data_path = data_path + "MFC18_EvalPart1/targets/manipulation/mask/"
    sys_data_path = data_path +"MFC18_EvalPart1/c8-lgb_local_40_nb_a/masks/"
    
    def get_img_ref_data(self):
        img_refs = self.get_img_ref()
        data = []
        for img_ref in img_refs[0:1]:
            result = self.read_img_ref(img_ref)
            data.append(MediforData(result['ref'], result['sys'], ''))
        return data
        
    def read_img_ref(self, img_ref):
        sys_img_path = self.sys_data_path+img_ref.sys_mask_file_name + ".png"
        try:
            sys_image = Image.open(sys_img_path)
        except:
            error_msg = 'failed to open: %s' % sys_img_path
            print(error_msg)
            sys.exit(error_msg)
        ref_img_path = self.ref_data_path + img_ref.ref_mask_file_name +".png"
        try:
            ref_image = Image.open(ref_img_path)
        except:
            error_msg = 'failed to open: %s' % ref_img_path
            print(error_msg)
            sys.exit(error_msg)
        return {'ref':ref_image, 'sys':sys_image} 
                
    def get_img_ref(self):
        path = self.data_path+"image_ref.csv"
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            # get header from first row
            headers = next(reader)
            # get all the rows as a list
            all_rows = np.array(list(reader))
        valid_rows =all_rows[all_rows[:,4] != '']
        required_data = valid_rows[:,[1,4]]
        sys_masks = required_data[:,0]
        ref_masks = list(map(lambda x:self.extract_ref_mask_file_name(x) ,required_data[:,1]))
        img_refs = []
        for i in range(len(sys_masks)):
            img_refs.append(ImgRefs(sys_masks[i], ref_masks[i]))
        return img_refs
        
    def extract_ref_mask_file_name(self, text):
        return re.search("(?<=reference\/manipulation-image\/mask\/).*(?=.ccm.png)", text).group()

class ImgRefs:
    sys_mask_file_name = None
    ref_mask_file_name = None
    
    def __init__(self, data):
        self.sys_mask_file_names = data[:,0]
        self.ref_mask_file_names = data[:,1]
        
    def __init__(self,sys, ref):
        self.sys_mask_file_name = sys
        self.ref_mask_file_name = ref    
        
# irb = ImgRefBuilder()
# irb.get_img_ref_data()
        
    