import numpy as np
import os
import cv2
import itertools
import sys

from data_generators.patch_test_data_generator import PatchTestDataGenerator

class PatchTrainDataGenerator(PatchTestDataGenerator):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                shuffle=False, patches_path="", patch_shape=128,
                num_patches=8,patch_tuning=None, img_refs=None, data_size = 100,
                targets_path = "", indicators_path="", missing_probe_file_ids=[]
                ):
        super().__init__(data_size=data_size,
                        img_refs = img_refs,
                        patch_shape = patch_shape,
                        batch_size = batch_size,
                        indicator_directories = indicator_directories,
                        indicators_path = indicators_path,
                        targets_path = targets_path,
                        missing_probe_file_ids = missing_probe_file_ids)

    def __getitem__(self, index):
        indicator_imgs, target_imgs, ids = super().__getitem__(index) 
        x = []
        y = []
        for i,j in zip(indicator_imgs,target_imgs):
            try:
                x = np.concatenate((x,i))
            except ValueError as err:
                x = i
            try:
                y = np.concatenate((y,j))
            except ValueError as err:
                y = j
        
        if len(x) == 0:
            x = np.empty([0,self.patch_shape, self.patch_shape,len(self.indicator_directories)])        
        if len(y) == 0:
            y = np.empty([0,self.patch_shape, self.patch_shape, 1])       
        return np.array(x),np.array(y), None