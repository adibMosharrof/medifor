import sys
sys.path.append('..')

import numpy as np
import os
import csv

from shared.image_utils import ImageUtils
from shared.folder_utils import FolderUtils
from shared.patch_utils import PatchUtils

from patch_metadata import PatchMetadata

class PatchGenerator:
    
    def __init__(self, img_refs, output_dir, targets_path, indicator_directories, indicators_path, patch_shape):
        self.img_refs = img_refs
        self.output_dir = output_dir
        self.patch_shape = patch_shape, patch_shape
        self.targets_path = targets_path
        self.indicator_directories= indicator_directories
        self.indicators_path = indicators_path
    
    def start(self):
        patch_metas= []
        for img_ref in self.img_refs:
            patch_metas.append(self.create_patches(img_ref))
        self.create_patch_img_ref(patch_metas)

    def create_patches(self, img_ref):
        probe_file_out_dir = FolderUtils.make_dir(self.output_dir + img_ref.sys_mask_file_name)
        target_image_path = os.path.join(self.targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
        try:
            original_image, border_vertical, border_horizontal = ImageUtils.read_image_add_border(target_image_path, self.patch_shape)
            target_image_out_dir = FolderUtils.make_dir(probe_file_out_dir+ 'target_image')
            original_image_patches, patch_window_shape = PatchUtils.get_patches(original_image, self.patch_shape)
            
            for i, patch in enumerate(original_image_patches):
                path = f'{target_image_out_dir}{i}.png'
                ImageUtils.save_image(patch, path)
            meta = PatchMetadata(original_image.shape, patch_window_shape,  img_ref.sys_mask_file_name)
        except ValueError as err:
            print(err)
            raise
            
        indicators = None
#         indicators = []
        for dir in self.indicator_directories:
            indicator_out_path = FolderUtils.make_dir(probe_file_out_dir + dir)
            img_path = os.path.join(self.indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
            try:
                img, _, _ = ImageUtils.read_image_add_border(
                    img_path, self.patch_shape,
                    vertical= border_vertical,
                    horizontal=border_horizontal)
            except FileNotFoundError as err:
                img = self._handle_missing_indicator_image(original_image.shape)
            finally:
                img_patches, _ = PatchUtils.get_patches(img, self.patch_shape)  
                for i, patch in enumerate(img_patches):
                    path = f'{indicator_out_path}{i}.png'
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
    
    
    