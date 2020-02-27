import numpy as np
from data_generators.train_data_generator import TrainDataGenerator
import math
import os

from shared.image_utils import ImageUtils
from tensorflow.python.keras.utils.data_utils import Sequence

class ImgPixelTrainDataGenerator(Sequence):
    
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
        self.missing_probe_file_ids = []
        
    def __getitem__(self, index):
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        
        img_refs = self.img_refs[starting_index:ending_index]



        target_imgs = []
#         target_imgs_path = self.patches_path + 'target_image'
        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        y = []
        for target_img in target_imgs:
            y = np.concatenate((y, target_img))
        
        indicator_imgs = []
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
                
        return x, y, None
        
    def _read_indicators(self, img_ref):
        indicators = []
        for indicator_name in self.indicator_directories:
            indicator_path = self.indicators_path + indicator_name + "/mask/" + img_ref.probe_file_id + ".png"
            try:
                img = ImageUtils.read_image(indicator_path)
                indicator_img = 255 - img 
            except FileNotFoundError as err:
                indicator_img = np.zeros([img_ref.img_height, img_ref.img_width])
            indicators.append(indicator_img.ravel())
        return np.column_stack(indicators)
    
    def _read_images_from_directory(self, dir_path, img_refs):
#         imgs = np.array([])
        imgs = []
        for (i,img_ref) in enumerate(img_refs):
            img_path = os.path.join(dir_path, img_ref.probe_mask_file_name + ".png")
            try:
                img = ImageUtils.read_image(img_path)
                img_raveled = img.ravel()
                flipped_img = 255-img_raveled
                thresholded_img = np.where(flipped_img > 127, 1,0)
                imgs.append(thresholded_img)
            except FileNotFoundError as err:
#                 print(f'deleting img with id {img_ref.probe_mask_file_name}')
                self.missing_probe_file_ids.append(img_ref.probe_file_id)
#                 del img_refs[i]
#                 print(f'deleted img with id {self.img_refs[i].probe_file_id} at index {i}')
#                 img = np.zeros([img_ref.img_height, img_ref.img_width])
#             imgs = np.concatenate((imgs, img.ravel()))
#         for i in missing_indexes:
#             print(f'deleting img with id {img_refs[i].probe_mask_file_name}')
#             del img_refs[i]
#         if len(missing_ids) > 0:
#             indexes  = [i for i,img_ref in enumerate(self.img_refs) if img_ref.probe_file_id in missing_ids]
#             for i in indexes:
#                 del self.img_refs[i]
        return imgs
    
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))