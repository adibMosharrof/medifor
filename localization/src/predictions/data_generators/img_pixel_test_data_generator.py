import numpy as np
import math
import os

from shared.image_utils import ImageUtils
from tensorflow.python.keras.utils.data_utils import Sequence


class ImgPixelTestDataGenerator(Sequence):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                shuffle=False, patches_path="", patch_shape=128,
                num_patches=8,patch_tuning=None, img_refs=None, data_size = 100,
                targets_path = "", indicators_path="", 
                missing_probe_file_ids = None,image_downscale_factor=1
                ):
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.img_refs = img_refs
        self.data_size = data_size
        self.indicators_path = indicators_path
        self.indicator_directories = indicator_directories
        self.targets_path = os.path.join(targets_path, "manipulation","mask")
        self.missing_probe_file_ids = missing_probe_file_ids
        self.image_downscale_factor = image_downscale_factor
        self.index= None
        
    def __getitem__(self, index):
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        self.index = index
        img_refs = self.img_refs[starting_index:ending_index]

        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        
        indicator_imgs = [None]*len(img_refs)
        for i,img_ref in enumerate(img_refs):
            indicator_imgs[i] = self._read_indicators(img_ref)
        if len(indicator_imgs) ==0:
            indicator_imgs = np.empty([0,len(self.indicator_directories)])
        return np.array(indicator_imgs), np.array(target_imgs), [i.probe_file_id for i in img_refs]
        
    def _read_indicators(self, img_ref):
#        indicators = []
        shrunken_height, shrunken_width =ImageUtils.get_shrunken_dimensions(
            img_ref.img_height, img_ref.img_width, self.image_downscale_factor) 
        indicators = np.empty(
            [shrunken_height * shrunken_width, len(self.indicator_directories)])
        for i,indicator_name in enumerate(self.indicator_directories):
            indicator_path = self.indicators_path + indicator_name + "/mask/" + img_ref.probe_file_id + ".png"
            try:
                img = ImageUtils.read_image(indicator_path)
                img = ImageUtils.shrink_image(img,self.image_downscale_factor)
                indicator_img = 255 - img 
            except FileNotFoundError as err:
                indicator_img = np.zeros([
                    shrunken_height,
                    shrunken_width
                    ])
 #           indicators.append(indicator_img.ravel())
            try:
                raveled = indicator_img.ravel()
                indicators[:,i] = raveled
            except ValueError as err:
                print(f'img ref dims {(img_ref.img_height//self.image_downscale_factor)* (img_ref.img_width//self.image_downscale_factor)}')
                print(f'img ref original dims {img_ref.img_orig_height* img_ref.img_orig_width}')
                print(f'index value {self.index}')
                print(f'indicators shape {indicators.shape} current indicator shape {raveled.shape}')
                
                
#        return np.column_stack(indicators)
        return indicators
    
    def _read_images_from_directory(self, dir_path, img_refs):
        imgs = [None]*len(img_refs)
#         imgs = np.empty((len(img_refs),), dtype=object)
        for (i,img_ref) in enumerate(img_refs):
            img_path = os.path.join(dir_path, img_ref.probe_mask_file_name + ".png")
            try:
                img = ImageUtils.read_image(img_path)
                img = ImageUtils.shrink_image(img,self.image_downscale_factor)
                img_raveled = img.ravel()
                flipped_img = 255-img_raveled
                thresholded_img = np.where(flipped_img > 127, 1,0)
                imgs[i]=thresholded_img
#                 imgs.append(thresholded_img)
            except FileNotFoundError as err:
                self.missing_probe_file_ids.append(img_ref.probe_file_id)
        return imgs
    
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))