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

class Main():
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
        self.output_dir = FolderUtils.create_output_folder(model_name,self.env_json["path"]["outputs"])
        self.my_logger = LogUtils.init_log(self.output_dir)
        
        a = 1

    def run(self):
        my_logger = logging.getLogger()
        patches_path, patch_img_ref_path, indicators_path = PathUtils.get_paths_for_patches(self.config_json, self.env_json)
        starting_index, ending_index = JsonLoader.get_data_size(self.env_json)
        indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        patch_shape = self.env_json['patch_shape']
        
        patch_img_refs = PatchImageRefFactory.get_img_refs_from_csv(patch_img_ref_path, starting_index, ending_index)
        
        train_batch_size = self.env_json['train_batch_size']
        test_batch_size = self.env_json['test_batch_size']
        train_data_size = self.env_json['train_data_size']
        num_training_patches = self._get_num_patches(patch_img_refs[:train_data_size])
        
        test_data_size = ending_index-starting_index - train_data_size

        train_gen = PatchTrainDataGenerator(
                        batch_size= train_batch_size,
                        indicator_directories = indicator_directories,
                        patches_path= patches_path,
                        patch_shape=patch_shape,
                        num_patches = num_training_patches
                        )
        test_patch_img_refs = patch_img_refs[ending_index - test_data_size -1 :]
        test_gen = PatchTestDataGenerator(
                        batch_size= test_batch_size,
                        indicator_directories = indicator_directories,
                        patches_path= patches_path,
                        patch_shape=patch_shape,
                        data_size = test_data_size,
                        patch_img_refs = test_patch_img_refs
                        )
        
        
        unet = UNet()
        model = unet.get_model(patch_shape, len(indicator_directories))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        a = model.fit_generator(generator=train_gen,
#                                 validation_data = validation_gen,
#                                 epochs=1,
#                                 use_multiprocessing=True,
#                                 workers= 4,
                                )
        predictions, test_patch_img_refs = self._get_test_predictions(model, test_gen, test_data_size, test_batch_size)
        recon = self.reconstruct_images_from_predictions(predictions, test_patch_img_refs)
        a=1
    
    def _get_num_patches(self, patch_img_refs):
        num_patches = 0
        for patch_img_ref in patch_img_refs:
            window_shape = patch_img_ref.patch_window_shape
            num_patches += window_shape[0] * window_shape[1]
        return num_patches    
        
    def _get_test_predictions(self, model, test_gen, data_size, batch_size):
        predictions = []
#         x_list = []
        patch_img_ref_list = []
        for i in range(int(math.ceil(data_size/batch_size))):
            x_list,y_list, patch_img_refs = test_gen.__getitem__(i)
            patch_img_ref_list += patch_img_refs
            for x in x_list:
                predictions.append(model.predict(x))
        return predictions,patch_img_ref_list

        
    def reconstruct_images_from_predictions(self, predictions, patch_img_refs):
        for prediction , patch_img_ref in zip(predictions, patch_img_refs):
            prediction = 255- (prediction*255)
            img_from_patches = PatchUtils.get_image_from_patches(
                                    prediction, 
                                    patch_img_ref.original_image_shape,
                                    patch_img_ref.patch_window_shape)
            file_name = f'{patch_img_ref.probe_file_id}.png'
            file_path = self.output_dir+file_name
            ImageUtils.save_image(img_from_patches, file_path)
            a=1
    
    def reconstruct_images_from_predictions1(self, model, validation_gen, validation_data_size, batch_size, output_dir):
        
        for i in range(validation_data_size//batch_size):
            x_list,y_list,meta_list = validation_gen.__getitem__(i, include_meta=True)
            predictions = model.predict(x_list)
            for (prediction, y, meta) in zip(predictions, y_list, meta_list):
                prediction = 255 - (prediction*255).astype(np.uint8)
                resized = cv2.resize(prediction, meta.original_image_shape)
                file_name =  f'{meta.probe_file_id}.png'
                file_path = f'{output_dir}/{file_name}'
                ImageUtils.save_image(resized,file_path)
    
     
    def train_model(self, train_gen, validation_gen, num_indicators, patch_shape):
        unet = UNet()
        model = unet.get_model(patch_shape, num_indicators)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        a = model.fit_generator(generator=train_gen,
#                                 validation_data = validation_gen,
#                                 epochs=1,
#                                 use_multiprocessing=True,
#                                 workers= 4,
                                )
        print(a)
        return model    

    def start(self):
        my_logger = logging.getLogger()
        self.my_timing = Timing(my_logger)
        
        train_data_size = 2
        validation_data_size = 2
        batch_size = 2
        
        indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
        indicator_directories = indicator_directories[:2]
        
        img_refs_validation = img_refs[train_data_size: train_data_size+validation_data_size:]
        train_gen = DataGenerator(img_refs[:train_data_size], self.targets_path, indicator_directories, self.indicators_path, batch_size, self.image_size) 
        validation_gen = DataGenerator(img_refs_validation, self.targets_path, indicator_directories, self.indicators_path, batch_size, self.image_size)
        
        LogUtils.print_memory_usage("Before starting training")
        model = self.train_model1(train_gen, validation_gen, len(indicator_directories))
#         model = self.train_model(validation_gen, train_gen, len(indicator_directories))
        self.reconstruct_images_from_predictions1(model, 
                                                validation_gen, 
                                                validation_data_size, 
                                                batch_size,
                                                self.output_dir)

        score = self.get_score(img_refs_validation)
        a = 1
        
    def train_model1(self, train_gen, test_gen, num_indicators):
        unet = UNet()
        model = unet.get_model(self.image_size, num_indicators)
        LogUtils.print_memory_usage("After getting model")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        LogUtils.print_memory_usage("Before fitting generator")
        a = model.fit_generator(generator=train_gen,
#                                 validation_data = test_gen,
#                                 epochs=1,
#                                 use_multiprocessing=True,
#                                 workers= 4,
                                )
        print(a)
        return model
        
    def get_score(self, img_refs):
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            scorer.start(data, self.env_json["threshold_step"])
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)

    

    

    def get_indicator_directories(self, indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    
    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
    
if __name__ == '__main__':
    
    m = Main()
    m.run()