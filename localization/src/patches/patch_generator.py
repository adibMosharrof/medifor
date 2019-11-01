import sys
sys.path.append('..')

import numpy as np
import os
import csv

from shared.image_utils import ImageUtils
from shared.folder_utils import FolderUtils
from shared.patch_utils import PatchUtils
from shared.path_utils import PathUtils
from patch_image_ref import PatchImageRefFactory


class PatchGenerator:
    
    def __init__(self, output_dir, indicators_path, patch_shape, img_downscale_factor):
        self.output_dir = output_dir
        self.patch_shape = patch_shape, patch_shape
        self.indicators_path = indicators_path
        self.indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
        self.img_downscale_factor = img_downscale_factor
        
    def create_img_patches(self, img_refs, targets_path):
        patch_metas = []
        for img_ref in img_refs:
            patch_metas.append(self._create_img_patch(img_ref, targets_path))
        self.create_patch_img_ref(patch_metas)

    def _create_img_patch(self, img_ref, targets_path):
        target_image_path = os.path.join(targets_path, "manipulation", "mask", img_ref.ref_mask_file_name) + ".png"
        try:
            original_image, border_vertical, border_horizontal = ImageUtils.get_image_with_border(target_image_path, self.patch_shape, self.img_downscale_factor)
            target_image_out_dir = self.output_dir + 'target_image/'
            original_image_patches, patch_window_shape = PatchUtils.get_patches(original_image, self.patch_shape)
            
            for i, patch in enumerate(original_image_patches):
                path = f'{target_image_out_dir}{img_ref.sys_mask_file_name}_{i}.png'
                ImageUtils.save_image(patch, path)
            meta = PatchImageRefFactory.get_img_ref(img_ref.sys_mask_file_name, original_image.shape, patch_window_shape)
        except ValueError as err:
            print(err)
            raise
            
        indicators = None
        for dir in self.indicator_directories:
            indicator_out_path = self.output_dir + dir + '/'
            img_path = os.path.join(self.indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
            try:
                img, _, _ = ImageUtils.get_image_with_border(
                    img_path, self.patch_shape,
                    self.img_downscale_factor,
                    vertical=border_vertical,
                    horizontal=border_horizontal)
            except FileNotFoundError as err:
                img = self._handle_missing_indicator_image(original_image.shape)
            finally:
                img_patches, _ = PatchUtils.get_patches(img, self.patch_shape)  
                for i, patch in enumerate(img_patches):
                    path = f'{indicator_out_path}{img_ref.sys_mask_file_name}_{i}.png'
                    ImageUtils.save_image(patch, path)
        
        return meta
    
    def create_patch_img_ref(self, patch_metas):
        file_name = self.output_dir + 'patch_image_ref.csv'
        with open(file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['ProbeFileID', 'OriginalImageShape', 'PatchWindowShape'])
            writer.writerows(patch_metas)
        csv_file.close()
            
    def _handle_missing_indicator_image(self, shape):
        return np.full(shape, 255, dtype='uint8')
    
