
import numpy as np
import keras
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import os
import cv2
import itertools
from time import time
import psutil


from shared.image_utils import ImageUtils
from shared.patch_utils import PatchUtils

class DataGenerator(Sequence):

    def __init__(self, img_refs, targets_path, 
                 indicator_directories, indicators_path,
                 batch_size=2, patch_shape=(256,256), img_size=256,
                 shuffle=False):
        
        self.img_refs = img_refs
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.img_size = img_size
        self.targets_path = targets_path
        self.indicator_directories= indicator_directories
        self.indicators_path = indicators_path
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __load__(self, img_ref):
        target_image_path = os.path.join(self.targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
        try:
            original_image, border_vertical, border_horizontal = ImageUtils.read_image_add_border(target_image_path, self.patch_shape)
            start = time()
            
            original_image_patches, patch_window_shape = PatchUtils.get_patches(original_image, self.patch_shape)
            meta = MetaData(original_image.shape, patch_window_shape,  img_ref.sys_mask_file_name)
            
        except ValueError as err:
            print(err)
            raise
            
        indicators = None
        for dir in self.indicator_directories:
            img_path = os.path.join(self.indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
            try:
                img, _, _ = ImageUtils.read_image_add_border(
                    img_path, self.patch_shape,
                    normalize=True,
                    vertical= border_vertical,
                    horizontal=border_horizontal)
            except FileNotFoundError as err:
                img = self.handle_missing_indicator_image(original_image.shape)
            finally:
                img_patches, _ = PatchUtils.get_patches(img, self.patch_shape)  
                try:
                    indicators = list(itertools.chain(indicators, img_patches))
                except TypeError:
                    indicators = img_patches
        return indicators, original_image_patches, meta

    def __getitem__(self, index, include_meta=False):
        self.print_memory_usage()
        if(index+1)*self.batch_size > len(self.img_refs):
            self.batch_size = len(self.img_refs) - index*self.batch_size

        img_refs = self.img_refs[index*self.batch_size:(index+1)*self.batch_size]   
        
        x = []
        y = []
        metas = []
        for img_ref in img_refs:
            indicators , target_image , meta = self.__load__(img_ref)
            self.print_memory_usage()
            x.append(indicators)
            self.print_memory_usage()
            y.append(target_image)
            self.print_memory_usage()
            metas.append(meta)
            self.print_memory_usage()
        x = np.array(x).reshape(-1, self.patch_shape[0], self.patch_shape[1], len(x[0]))
        self.print_memory_usage()
        y = np.array(y).reshape(-1, self.patch_shape[0], self.patch_shape[1], 1)
#         x = x/255
        if include_meta is True:
            metas = np.array(metas)
            return x, y, metas
        return x, y
    
    
    def handle_missing_indicator_image(self, shape):
        return np.zeros(shape)
    
    def __len__(self):
        return int(np.ceil(len(self.img_refs)/float(self.batch_size)))
    
    
    def on_epoch_end(self):
        if self.shuffle is True:    
            random.shuffle(self.img_refs)
    
    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/10**6
        print(memory)
    
class MetaData:
    def __init__(self, shape, patch_window_shape, id):
        self.original_image_shape = shape
        self.patch_window_shape = patch_window_shape
        self.probe_file_id = id
    
    