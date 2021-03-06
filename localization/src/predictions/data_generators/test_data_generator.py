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
import re

from shared.image_utils import ImageUtils
from shared.patch_utils import PatchUtils


class TestDataGenerator(Sequence):
    
    def __init__(self,
                batch_size=3, indicator_directories=[],
                shuffle=False, patches_path="", patch_shape=128,
                data_size=8, patch_img_refs=[], patch_tuning=None):
        
        self.batch_size = batch_size
        self.indicator_directories = indicator_directories
        self.shuffle = shuffle
        self.patches_path = patches_path
        self.patch_shape = patch_shape
        self.data_size = data_size
        self.patch_img_refs = patch_img_refs
        self.patch_tuning = patch_tuning
        self.on_epoch_end()
        
    def __getitem__(self, patch_img_ref):
        indicator_imgs = []
        target_imgs = []
        for indicator_name in self.indicator_directories:
            indicator_path = self.patches_path + indicator_name
            indicator_patches = self._get_img_patches_by_id(
                        indicator_path, patch_img_ref.probe_file_id)
            indicator_imgs.append(indicator_patches)
        
        target_imgs_path = self.patches_path + 'target_image'
        target_imgs= self._get_img_patches_by_id(
            target_imgs_path, patch_img_ref.probe_file_id)
            
        x = np.array(indicator_imgs)
        y = np.array(target_imgs)
        return x, y
    
    def _get_img_patches_by_id(self, dir_path, probe_file_id):
        imgs = []
        img_names = self._get_img_file_names(dir_path, probe_file_id)
        for name in img_names:
            img_path = os.path.join(dir_path, name)
            img = ImageUtils.read_image(img_path)
            imgs.append(img)
        return imgs
    
    def _get_img_file_names(self, dir_path, probe_file_id):
        files = os.listdir(dir_path)
        img_names = [f for f in files if str(f).startswith(probe_file_id) ]
        return sorted(img_names, key=lambda x: int(re.search("(?<=_).*(?=.png)", x).group()))
 
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))
