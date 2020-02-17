import numpy as np
from data_generators.train_data_generator import TrainDataGenerator
import math
import os

from shared.image_utils import ImageUtils

class ImgPixelTrainDataGenerator():
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                shuffle=False, patches_path="", patch_shape=128,
                num_patches=8,patch_tuning=None, img_refs=None, data_size = 100,
                targets_path = "", indicators_path=""
                ):
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.img_refs = img_refs
        self.data_size = data_size
        self.indicators_path = indicators_path
        self.indicator_directories = indicator_directories
        self.targets_path = os.path.join(targets_path, "manipulation","mask")
        
    def __getitem__(self, index):
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        
        img_refs = self.img_refs[starting_index:ending_index]

        indicator_imgs = []
        for indicator_name in self.indicator_directories:
            indicator_path = self.indicators_path + indicator_name + "/mask/"
            indicator_patches = self._read_images_from_directory(indicator_path, img_refs)
            indicator_imgs.append(indicator_patches)
            
        target_imgs = []
#         target_imgs_path = self.patches_path + 'target_image'
        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        x = np.array(indicator_imgs).reshape(-1, len(self.indicator_directories))
        y = np.array(target_imgs).reshape(-1, 1)
        return x, y
        
    def _read_images_from_directory(self, dir_path, img_refs):
        imgs = []
        
        for img_ref in img_refs:
            img_path = os.path.join(dir_path, img_ref.probe_file_id + ".png")
            try:
                img = ImageUtils.read_image(img_path)
            except FileNotFoundError as err:
                img = np.zeros(img_ref.img_width, img_ref.img_height)
            imgs.append(img)
        return imgs
    
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))