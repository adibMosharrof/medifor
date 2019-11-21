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
import graphviz
import json
import logging
import cv2
import sys
import numpy as np
import multiprocessing
from datetime import datetime
import math

import matplotlib.pyplot as plt

sys.path.append('..')
from data_generator import DataGenerator
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
from patch_train_data_generator import PatchTrainDataGenerator
from patch_test_data_generator import PatchTestDataGenerator


class PatchPredictions():
    
    def __init__(self, config):
        self.config = config
        model_name = self.config["model_name"]
        print(model_name)
        self.patch_shape = self.config['patch_shape']
        img_downscale_factor = self.config['image_downscale_factor']
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir = FolderUtils.create_predictions_output_folder(
            model_name, self.patch_shape, img_downscale_factor,
            output_folder)
        
        self.my_logger = LogUtils.init_log(self.output_dir)
        
        self.patches_path, patch_img_ref_path, indicators_path, img_ref_csv, self.ref_data_path = PathUtils.get_paths_for_patches(self.config)
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        
        self.indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        self.patch_img_refs, self.ending_index = PatchImageRefFactory.get_img_refs_from_csv(
            patch_img_ref_path, self.starting_index, self.ending_index)
                
        self.train_batch_size, self.test_batch_size, self.train_data_size, self.test_data_size, self.num_training_patches = self._get_test_train_data_size(
            self.config, self.patch_img_refs, self.starting_index, self.ending_index)
        
        self.test_patch_img_refs = self.patch_img_refs[self.ending_index - self.test_data_size :]
        
    def run(self):
        my_logger = logging.getLogger()
        
        starting_index, ending_index = JsonLoader.get_data_size(self.config)
        indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        
        patch_img_refs, ending_index = PatchImageRefFactory.get_img_refs_from_csv(
            patch_img_ref_path, starting_index, ending_index)
        
        train_batch_size, test_batch_size, train_data_size, test_data_size, num_training_patches = self._get_test_train_data_size(
            self.config, patch_img_refs, starting_index, ending_index)

        test_patch_img_refs = patch_img_refs[ending_index - test_data_size :]
        
        train_gen, test_gen = self.get_data_generators(
            train_batch_size, test_batch_size, test_data_size, indicator_directories,
            patches_path, self.patch_shape, num_training_patches, test_patch_img_refs)
        
        arch = self._get_architecture()
        model = self.train_model(arch, indicator_directories, train_gen)
        predictions = self.make_predictions(model, test_gen, test_data_size, test_batch_size)
        recon = self._reconstruct(predictions, test_patch_img_refs)
        img_refs = ImgRefBuilder.get_img_ref_from_patch_ref(test_patch_img_refs)
        
        score = self.get_score(img_refs, self.output_dir, ref_data_path)
        print(self.output_dir)
        
    def get_data_generators(self):
        
        train_gen = PatchTrainDataGenerator(
                        batch_size=self.train_batch_size,
                        indicator_directories=self.indicator_directories,
                        patches_path=self.patches_path,
                        patch_shape=self.patch_shape,
                        num_patches=self.num_training_patches
                        )
        
        test_gen = PatchTestDataGenerator(
                        batch_size=self.test_batch_size,
                        indicator_directories=self.indicator_directories,
                        patches_path=self.patches_path,
                        patch_shape=self.patch_shape,
                        data_size=self.test_data_size,
                        patch_img_refs=self.test_patch_img_refs
                        )
        return train_gen, test_gen
        
    def train_model(self, train_gen):
        arch = self._get_architecture()
        model = arch.get_model(self.patch_shape, len(self.indicator_directories))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        
        epochs = self.config["epochs"]
        workers = self.config["workers"]
        
        model.fit_generator(generator=train_gen,
                                epochs=epochs,
                                use_multiprocessing=True,
                                workers=workers,
                                )
        return model
    
    def predict(self, model, test_gen):
        predictions = []
        for i in range(int(math.ceil(self.test_data_size / self.test_batch_size))):
            x_list, y_list = test_gen.__getitem__(i)
            for x in x_list:
                predictions.append(model.predict(x))
        self._reconstruct(predictions)
    
    def get_score(self):
        img_refs = ImgRefBuilder.get_img_ref_from_patch_ref(self.test_patch_img_refs)
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)   
            
    
    def _reconstruct(self, predictions):
        for prediction , patch_img_ref in zip(predictions, self.test_patch_img_refs):
#             prediction = 255- (prediction*255)
            prediction = prediction * 255
            img_from_patches = PatchUtils.get_image_from_patches(
                                    prediction,
                                    patch_img_ref.bordered_img_shape,
                                    patch_img_ref.patch_window_shape)
            img_without_border = ImageUtils.remove_border(
                img_from_patches, patch_img_ref.border_top, patch_img_ref.border_left)
            img_original_size = cv2.resize(
                img_without_border, patch_img_ref.original_img_shape)
            file_name = f'{patch_img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)


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
        model_name = self.config['model_name']
        if model_name == "unet":
            from architectures.unet import UNet
            arch = UNet()
        elif model_name == "single_layer_nn":
            from architectures.single_layer_nn import SingleLayerNN
            arch = SingleLayerNN()
        return arch 
  