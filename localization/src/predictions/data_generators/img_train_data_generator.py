import numpy as np
from data_generators.img_test_data_generator import ImgTestDataGenerator
import math
import os

from shared.image_utils import ImageUtils

class ImgTrainDataGenerator(ImgTestDataGenerator):
    
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
        
        img_refs = self.img_refs[starting_index:ending_index]

        target_imgs = []
        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        y = []
        for target_img in target_imgs:
           y.append(target_img) 
        
        indicator_imgs =[] 
        for img_ref in img_refs:
            if img_ref.probe_file_id in self.missing_probe_file_ids:
                continue
            indicator_imgs.append(self._read_indicators(img_ref))
        
        x = np.array(indicator_imgs)
        y = np.array(y)
        return x, y, None
        
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))