'''

load data
get model class
train and test 
    splits
    execution
reconstruct images from predictions    



'''
import sys, os
from pathlib import Path
import sklearn

from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef

from sklearn import metrics
import graphviz
import json
import logging
import socket
import cv2
import sys
import numpy as np
import pickle
import psutil
import multiprocessing
from datetime import datetime
import math

import matplotlib.pyplot as plt

sys.path.append('..')
from unet import UNet
from data_generator import DataGenerator
# from medifor_prediction import MediforPrediction
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

class PredictionRunner():
    config_path = "../../configurations/predictions/"
    indicators_path = "../../data/MFC18_EvalPart1/indicators"
    targets_path = "../../data/MFC18_EvalPart1/targets"
    env_json = None
    config_json = None
    image_utils = None
    image_size = 128
    my_timing = None
    
    def __init__(self):
        self.config_json, self.env_json , self.email_json =JsonLoader.load_config_env_email(self.config_path) 
        model_name = self.config_json["default"]["model_name"]
        self.patch_shape = self.env_json['patch_shape']
        img_downscale_factor = self.env_json['image_downscale_factor']
        patches_output_folder = self.env_json["path"]["outputs"] + "patches/"
        self.output_dir = FolderUtils.create_predictions_output_folder(
            model_name, self.patch_shape, img_downscale_factor, 
            patches_output_folder)
        
        self.my_logger = LogUtils.init_log(self.output_dir)
        
    def run(self):
        my_logger = logging.getLogger()
        patches_path, patch_img_ref_path, indicators_path, img_ref_csv, ref_data_path = PathUtils.get_paths_for_patches(self.config_json, self.env_json)
        
        starting_index, ending_index = JsonLoader.get_data_size(self.env_json)
        indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        
        patch_img_refs = PatchImageRefFactory.get_img_refs_from_csv(
            patch_img_ref_path, starting_index, ending_index)
        
        train_batch_size, test_batch_size, train_data_size, test_data_size, num_training_patches = self._get_test_train_data_size(
            self.env_json, patch_img_refs, starting_index, ending_index)

        test_patch_img_refs = patch_img_refs[ending_index - test_data_size -1 :]
        
        train_gen, test_gen = self._get_train_test_generators(
            train_batch_size, test_batch_size, test_data_size, indicator_directories, 
            patches_path, self.patch_shape, num_training_patches, test_patch_img_refs)
        
        unet = UNet()
        model = unet.get_model(self.patch_shape, len(indicator_directories))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        epochs = self.env_json["epochs"]
        workers = self.env_json["workers"]
        a = model.fit_generator(generator=train_gen,
                                epochs=epochs,
                                use_multiprocessing=True,
                                workers= workers,
                                )
        predictions = self._get_test_predictions(model, test_gen, test_data_size, test_batch_size)
        recon = self._reconstruct_images_from_predictions(predictions, test_patch_img_refs)
        img_refs = ImgRefBuilder.get_img_ref_from_patch_ref(test_patch_img_refs)
        
        threshold_step = self.env_json['threshold_step']
        score = self._get_score(img_refs, threshold_step, self.output_dir, ref_data_path)
        
    def _get_train_test_generators(self, train_batch_size, test_batch_size, 
            test_data_size,indicator_directories, patches_path, patch_shape, 
            num_training_patches, test_patch_img_refs):
        
        train_gen = PatchTrainDataGenerator(
                        batch_size= train_batch_size,
                        indicator_directories = indicator_directories,
                        patches_path= patches_path,
                        patch_shape=self.patch_shape,
                        num_patches = num_training_patches
                        )
        test_gen = PatchTestDataGenerator(
                        batch_size= test_batch_size,
                        indicator_directories = indicator_directories,
                        patches_path= patches_path,
                        patch_shape=self.patch_shape,
                        data_size = test_data_size,
                        patch_img_refs = test_patch_img_refs
                        )
        return train_gen, test_gen
        
        
    def _get_num_patches(self, patch_img_refs):
        num_patches = 0
        for patch_img_ref in patch_img_refs:
            window_shape = patch_img_ref.patch_window_shape
            num_patches += window_shape[0] * window_shape[1]
        return num_patches    
        
    def _get_test_predictions(self, model, test_gen, data_size, batch_size):
        predictions = []
        for i in range(int(math.ceil(data_size/batch_size))):
            x_list,y_list = test_gen.__getitem__(i)
            for x in x_list:
                predictions.append(model.predict(x))
        return predictions
        
    def _reconstruct_images_from_predictions(self, predictions, patch_img_refs):
        for prediction , patch_img_ref in zip(predictions, patch_img_refs):
#             prediction = 255- (prediction*255)
            prediction = prediction*255
            img_from_patches = PatchUtils.get_image_from_patches(
                                    prediction, 
                                    patch_img_ref.bordered_img_shape,
                                    patch_img_ref.patch_window_shape)
            img_original_size = cv2.resize(
                img_from_patches, patch_img_ref.original_img_shape)
            file_name = f'{patch_img_ref.probe_file_id}.png'
            file_path = self.output_dir+file_name
            ImageUtils.save_image(img_original_size, file_path)
            a=1

    def _get_score(self, img_refs, threshold_step, output_dir, ref_data_path):
        data = MediforData.get_data(img_refs, output_dir, ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data, threshold_step)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)   
            
    def _get_test_train_data_size(self, env_json, patch_img_refs, starting_index, ending_index):
        train_batch_size =env_json['train_batch_size']
        test_batch_size = env_json['test_batch_size']
        train_data_size = env_json['train_data_size']
        num_training_patches = self._get_num_patches(patch_img_refs[:train_data_size])
        
        test_data_size = ending_index-starting_index - train_data_size
        
        return train_batch_size, test_batch_size, train_data_size, test_data_size, num_training_patches

    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
    
if __name__ == '__main__':
    
    m = PredictionRunner()
    m.run()