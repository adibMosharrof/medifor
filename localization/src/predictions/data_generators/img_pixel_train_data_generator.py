import numpy as np
from data_generators.img_pixel_test_data_generator import ImgPixelTestDataGenerator
import math
import os

from shared.image_utils import ImageUtils

class ImgPixelTrainDataGenerator(ImgPixelTestDataGenerator):
    
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
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        self.index = index
        img_refs = self.img_refs[starting_index:ending_index]

        target_imgs = []
#         target_imgs_path = self.patches_path + 'target_image'
        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        y = []
        for target_img in target_imgs:
            y = np.concatenate((y, target_img))
        
        indicator_imgs =[] 
        for img_ref in img_refs:
            if img_ref.probe_file_id in self.missing_probe_file_ids:
                continue
            indicator_imgs.append(self._read_indicators(img_ref))
        
        x = []    
        for indicators in indicator_imgs:
            reshaped =  np.array(indicators).reshape(-1, len(self.indicator_directories))
            if len(x) == 0:
                x = reshaped
                continue
            x = np.concatenate((x,reshaped))
            
        if len(x) == 0:
            x = np.empty([0,len(self.indicator_directories)])     
        if len(x) != len(y):   
            print(f'length of x {len(x)} and y {len(y)} is not the same for index {index}')
            return np.empty([0,len(self.indicator_directories)]), np.array([]), None
        return np.array(x), np.array(y), None
        
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))