import sys
sys.path.append('..')

import numpy as np
import os
import csv

from shared.image_utils import ImageUtils
from shared.folder_utils import FolderUtils
from shared.patch_utils import PatchUtils
from shared.path_utils import PathUtils
from shared.json_loader import JsonLoader

from scoring.img_ref_builder import ImgRefBuilder
import pandas as pd


class CsvToImageGenerator:
    
    def __init__(self,config=None, output_dir=None, df=None,indicators=None,index_dir=None,img_ref_dir=None):
        self.config = config
        self.output_dir = output_dir
        self.df = df
        self.indicators = indicators
        self.index_dir = index_dir
        self.img_ref_dir = img_ref_dir
        
        starting_index, ending_index = JsonLoader.get_data_size(self.config)
        irb = ImgRefBuilder(self.img_ref_dir+"image_ref.csv")
        self.img_refs = irb.get_img_ref(starting_index, ending_index)
        self.img_ref_dict = {}
        for img_ref in self.img_refs:
            self.img_ref_dict[img_ref.probe_file_id] = img_ref.probe_mask_file_name
        
    def generate_images(self):
        self.index_df = pd.read_csv(self.index_dir+'index.csv')
        grouped = self.df.groupby('image_id')
        targets_path = self.output_dir+"targets/manipulation/mask/"
            
        for image_id, group in grouped:
            target_img = self._get_img_from_image_id(group, image_id, "label")
            if target_img == None:
                continue
            try:
                file_name=self.img_ref_dict[image_id]
            except KeyError:
                continue
            self._save_image(target_img*255, targets_path, file_name)
            
            for indicator in self.indicators:
                indicator_img = self._get_img_from_image_id(group, image_id, indicator)
                indicator_path = f'{self.output_dir}indicators/{indicator}/mask/'
                self._save_image(indicator_img,indicator_path,image_id)

    def _get_img_from_image_id(self,group, image_id, column_name):
        item = group[column_name]
        index = self.index_df[self.index_df['image_id']==image_id]
        height = int(index['image_height'])
        width = int(index['image_width'])
        if int(height*width) != len(item):
            print("image dimensions did not match")
            return None
        return np.array(item).reshape(height,width)

    def _save_image(self,img,base_path, file_name):
        path = f'{base_path}{file_name}.png'
        ImageUtils.save_image(img,path)
