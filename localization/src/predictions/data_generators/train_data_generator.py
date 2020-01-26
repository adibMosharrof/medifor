import numpy as np
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import os
import cv2
import itertools
from time import time
import psutil
import sys

from shared.image_utils import ImageUtils
from shared.patch_utils import PatchUtils


class TrainDataGenerator(Sequence):
    
    def __init__(self, batch_size=10, indicator_directories=[],
                 shuffle=False, patches_path="", patch_shape=128, num_patches=8, patch_tuning=None):
        self.batch_size = batch_size
        self.indicator_directories = indicator_directories
        self.shuffle = shuffle
        self.patches_path = patches_path
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.patch_tuning = patch_tuning
        self.on_epoch_end()
        
    def __getitem__(self, index):
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        
        indicator_imgs = []
        for indicator_name in self.indicator_directories:
            indicator_path = self.patches_path + indicator_name
            indicator_patches = self._read_images_from_directory(indicator_path, starting_index, ending_index)
            indicator_imgs.append(indicator_patches)
            
        target_imgs = []
        target_imgs_path = self.patches_path + 'target_image'
        target_imgs = self._read_images_from_directory(target_imgs_path, starting_index, ending_index)
        
        if self.patch_tuning['dilate_y']:
            for img in target_imgs:
                target_imgs.append(ImageUtils.dilate(img))
        elif self.patch_tuning['patch_black']:
            target_imgs = self.patch_black(target_imgs)
        return indicator_imgs, target_imgs    
        
    def _read_images_from_directory(self, dir_path, starting_index, ending_index):
        img_names = os.listdir(dir_path)[starting_index:ending_index]
        imgs = []
        for name in img_names:
            img_path = os.path.join(dir_path, name)
            img = ImageUtils.read_image(img_path)
            imgs.append(img)
        return imgs

    def patch_black(self, imgs):
        patch_black = []
        for img in imgs:
            if len(np.nonzero( img < 255)[0]):
                new_img= np.zeros(img.shape)
                patch_black.append(new_img) 
            else:
                patch_black.append(img)
        return patch_black
    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
                    
    def __len__(self):
        return int(np.ceil(self.num_patches / float(self.batch_size)))
