'''

load data
get model class
train and test 
    splits
    execution
_reconstruct images from predictions    

'''
import sys, os
from pathlib import Path
import json
import logging
import cv2
import sys
import numpy as np
import multiprocessing
from datetime import datetime
import math
from tensorflow.keras.models import load_model
import itertools
import gc
import matplotlib.pyplot as plt

sys.path.append('..')
from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.image_utils import ImageUtils
from shared.path_utils import PathUtils
from shared.patch_utils import PatchUtils
from shared.timing import Timing
from shared.json_loader import JsonLoader
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from shared.medifordata import MediforData
from patches.patch_image_ref import PatchImageRefFactory

from data_generators.patch_train_data_generator import PatchTrainDataGenerator
from data_generators.patch_test_data_generator import PatchTestDataGenerator

from predictions import Predictions

class PatchPredictions(Predictions):
    
    def __init__(self, config, model_name=None, output_dir=None, target_id=None):
        super().__init__(config, model_name, output_dir)
        
        self.patches_path, self.patch_img_ref_path, self.indicators_path, img_ref_csv, self.ref_data_path = PathUtils.get_paths_for_patches(self.config)
        self.indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
           
        self.set_target_paths(target_id or config['target_id'])
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        
        self._prepare_img_refs(self.patch_img_ref_path)

    def get_data_generators(self, missing_probe_file_ids):
        train_gen = PatchTrainDataGenerator(
                        data_size=len(self.train_img_refs),
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.train_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )
        test_gen = PatchTestDataGenerator(
                        data_size=len(self.test_img_refs),
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.train_batch_size,
#                         batch_size = self.test_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        missing_probe_file_ids = missing_probe_file_ids
                        )
        valid_gen = PatchTrainDataGenerator(
                        data_size=len(self.test_img_refs),
                        img_refs = self.test_img_refs,
                        batch_size = self.train_batch_size,
                        patch_shape = self.patch_shape,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )  
#         a,b,c = test_gen.__getitem__(0)
        return train_gen, test_gen, valid_gen
            
    def set_target_paths(self, target_id):
        self.target_id = target_id
        self._prepare_img_refs(self.patch_img_ref_path)
        if target_id is -1:
            self.targets_path= self.patches_path +"target_image/"
        else:
            self.targets_path = self.patches_path + self.indicator_directories[target_id] + '/'
            self.ref_data_path = self.ref_data_path.replace('targets/manipulation', 'indicators/'+self.indicator_directories[target_id])
    
    def _reconstruct(self, predictions, ids):
        for (prediction,id) , patch_img_ref in zip(predictions, self.test_img_refs):
            prediction = 255- (prediction*255)
#             prediction = prediction * 255
            img = self._reconstruct_image_from_patches(prediction, patch_img_ref)
            img_original_size = cv2.resize(
                img, (patch_img_ref.img_orig_width, patch_img_ref.img_orig_height))
            
            file_name = f'{patch_img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)

    def _reconstruct_image_from_patches(self, prediction, patch_img_ref):
        img_from_patches = PatchUtils.get_image_from_patches(
                                    prediction,
                                    patch_img_ref.bordered_img_shape,
                                    patch_img_ref.patch_window_shape)
        img_without_border = ImageUtils.remove_border(
                img_from_patches, patch_img_ref.border_top, patch_img_ref.border_left)
        
        return img_without_border

    def _get_test_train_data_size(self, env_json, patch_img_refs, starting_index, ending_index):
        train_batch_size = env_json['train_batch_size']
        test_batch_size = env_json['test_batch_size']
        train_data_size = env_json['train_data_size']
        num_training_patches = self._get_num_patches(patch_img_refs[:train_data_size])
        
        test_data_size = ending_index - starting_index - train_data_size
        
        return train_batch_size, test_batch_size, train_data_size, test_data_size, num_training_patches
    
    def _get_num_patches(self, patch_img_refs):
        num_patches = 0
        for patch_img_ref in patch_img_refs:
            window_shape = patch_img_ref.patch_window_shape
            num_patches += window_shape[0] * window_shape[1]
        return num_patches    

                
    def _get_architecture(self):
        model_name = self.model_name
        if model_name == "unet":
            from architectures.unet import UNet
            arch = UNet()
        elif model_name == "single_layer_nn":
            from architectures.single_layer_nn import SingleLayerNN
            arch = SingleLayerNN()
        elif model_name == 'lr':
            from architectures.lr import Lr
            arch = Lr()
        elif model_name == 'mlp':
            from architectures.mlp import Mlp
            arch = Mlp()
        return arch 
    
    def _prepare_img_refs(self, patch_img_ref_path):
        self.img_refs, self.ending_index = PatchImageRefFactory.get_img_refs_from_csv(
            patch_img_ref_path, self.starting_index, self.ending_index, 
            target_index=self.target_id)
        ImgRefBuilder.add_image_width_height(self.img_refs, self.config)
                
        self.num_training_patches = self._get_num_patches(self.img_refs[:self.train_data_size])
        self.num_testing_patches = self._get_num_patches(self.img_refs[self.train_data_size:])
#         self.test_img_refs = self.img_refs[self.train_data_size:]
        self.train_img_refs = self.img_refs[:self.train_data_size]
        self.test_img_refs = self.train_img_refs
        