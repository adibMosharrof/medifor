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
from medifordata import MediforData
import logging

class ImgRefBuilder:
    base_data_path = "../data/"
    ref_data_path = base_data_path + "MFC18_EvalPart1/targets/manipulation/mask/"
    sys_data_path = base_data_path +"MFC18_EvalPart1/c8-lgb_local_40_nb_a/mask/"
    image_ref_csv_path = None
    
    def __init__(self, config_json, env_json):
        
        env_path = env_json['path']
        self.base_data_path = env_path['data']
        self.current_data_path = self.base_data_path+ config_json["default"]["data"]
        
        self.image_ref_csv_path =  self.current_data_path + env_path['image_ref_csv']
        self.ref_data_path = '{}{}'.format(self.current_data_path, env_path["target_mask"])
        model_type = config_json["default"]["model_type"]
        self.sys_data_path = '{}{}'.format(env_path["model_sys_predictions"], env_path["model_type"][model_type])
    
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
            error_msg = 'FAILED to open: %s' % sys_img_path
            logging.debug(error_msg)
            sys.exit(error_msg)
        ref_img_path = self.ref_data_path + img_ref.ref_mask_file_name +".png"
        try:
            ref_image = Image.open(ref_img_path)
        except:
            error_msg = 'FAILED to open: %s' % ref_img_path
            logging.debug(error_msg)
            sys.exit(error_msg)
        return {'ref':ref_image, 'sys':sys_image} 
                
    def get_img_ref(self):
        with open(self.image_ref_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            headers = next(reader)
            all_rows = np.array(list(reader))
        #only selected images that have a reference(we are only scoring the manipulated images)
        valid_rows =all_rows[all_rows[:,4] != '']
        #only need the sys and ref image names
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
        
    def __init__(self,sys, ref):
        self.sys_mask_file_name = sys
        self.ref_mask_file_name = ref    

        
    