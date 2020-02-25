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
    
    def __init__(self, output_dir, indicators_path, patch_shape, img_downscale_factor, tuning):
        self.output_dir = output_dir
        self.patch_shape = patch_shape, patch_shape
        self.indicators_path = indicators_path
        self.indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
        self.img_downscale_factor = img_downscale_factor
        self.tuning = tuning
        
    def create_img_patches(self, img_refs, targets_path):
        patch_img_refs = []
        for img_ref in img_refs:
            try:
                patch_img_ref = self._create_img_patch(img_ref, targets_path)
            except ZeroDivisionError as err:
                #not including images that have size less than the patch size
                print(f"division by zero {img_ref.probe_file_id}")
                continue
            except FileNotFoundError as err:
                print(f"Could not find image {err}")
                continue
            patch_img_refs.append(patch_img_ref)
        self.write_patch_img_refs_to_csv(patch_img_refs)

    def _create_img_patch(self, img_ref, targets_path):
        target_image_path = os.path.join(targets_path, "manipulation", "mask", img_ref.probe_mask_file_name) + ".png"
        
        border_value = [255,255,255]
        if self.tuning['black_border_y'] is True or self.tuning['dilate_y_black_border_y'] is True:
            border_value = [0,0,0]
        
        bordered_img, border_top, border_left, original_img_shape = ImageUtils.get_image_with_border(target_image_path, self.patch_shape, self.img_downscale_factor, border_value= border_value)
        if self.tuning["dilate_y"] is True:
            diluted_target = ImageUtils.dilate(bordered_img)
        target_image_out_dir = self.output_dir + 'target_image/'
        bordered_image_patches, patch_window_shape = PatchUtils.get_patches(bordered_img, self.patch_shape)
        for i, patch in enumerate(bordered_image_patches):
            path = f'{target_image_out_dir}{img_ref.probe_file_id}_{i}.png'
            ImageUtils.save_image(patch, path)
        patch_img_ref = PatchImageRefFactory.create_img_ref(
            img_ref.probe_file_id, bordered_img.shape, 
            patch_window_shape, img_ref.probe_mask_file_name, 
            original_img_shape, border_top, border_left)
            
        indicators = None
        for indicator_dir in self.indicator_directories:
            indicator_out_path = self.output_dir + indicator_dir + '/'
            img_path = os.path.join(self.indicators_path, indicator_dir, "mask", img_ref.probe_file_id) + ".png"
            try:
                img, _, _, _ = ImageUtils.get_image_with_border(
                    img_path, self.patch_shape,
                    self.img_downscale_factor,
                    top=border_top,
                    left=border_left)
            except FileNotFoundError as err:
                img = self._handle_missing_indicator_image(bordered_img.shape)
            finally:
                img_patches, _ = PatchUtils.get_patches(img, self.patch_shape)  
                for i, patch in enumerate(img_patches):
                    path = f'{indicator_out_path}{img_ref.probe_file_id}_{i}.png'
                    ImageUtils.save_image(patch, path)
        
        return patch_img_ref
    
    def write_patch_img_refs_to_csv(self, patch_img_refs):
        file_name = self.output_dir + 'patch_image_ref.csv'
        with open(file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['ProbeFileID', 'BorderedImageShape', 'PatchWindowShape', 'ProbeMaskFileName', 'OriginalImageShape', 'BorderTop', 'BorderLeft'])
            writer.writerows(patch_img_refs)
        csv_file.close()
            
    def _handle_missing_indicator_image(self, shape):
        return np.full(shape, 255, dtype=np.float32)
    
    def test_patch(self, patches, window_shape, original_img):
        return PatchUtils.get_image_from_patches(patches, original_img.shape, window_shape)
