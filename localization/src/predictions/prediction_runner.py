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
from config.config_loader import ConfigLoader

class PredictionRunner():
    
    def __init__(self):
        self.config , self.email_json = ConfigLoader.get_config()
        model_name = self.config["model_name"]
        self.patch_shape = self.config['patch_shape']
        img_downscale_factor = self.config['image_downscale_factor']
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir = FolderUtils.create_predictions_output_folder(
            model_name, self.patch_shape, img_downscale_factor, 
            output_folder)
        
        self.my_logger = LogUtils.init_log(self.output_dir)
        
    def run(self):
        my_logger = logging.getLogger()
        patches_path, patch_img_ref_path, indicators_path, img_ref_csv, ref_data_path = PathUtils.get_paths_for_patches(self.config)
        
        starting_index, ending_index = JsonLoader.get_data_size(self.config)
        indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        
        patch_img_refs = PatchImageRefFactory.get_img_refs_from_csv(
            patch_img_ref_path, starting_index, ending_index)
        
        train_batch_size, test_batch_size, train_data_size, test_data_size, num_training_patches = self._get_test_train_data_size(
            self.config, patch_img_refs, starting_index, ending_index)

        test_patch_img_refs = patch_img_refs[ending_index - test_data_size :]
        
        train_gen, test_gen = self._get_train_test_generators(
            train_batch_size, test_batch_size, test_data_size, indicator_directories, 
            patches_path, self.patch_shape, num_training_patches, test_patch_img_refs)
        
        arch = self._get_architecture()
        model = self._train_model(arch, indicator_directories, train_gen)
        predictions = self._get_test_predictions(model, test_gen, test_data_size, test_batch_size)
        recon = self._reconstruct_images_from_predictions(predictions, test_patch_img_refs)
        img_refs = ImgRefBuilder.get_img_ref_from_patch_ref(test_patch_img_refs)
        
        
        score = self._get_score(img_refs, self.output_dir, ref_data_path)
        print(self.output_dir)
        
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
        
    def _train_model(self, arch, indicator_directories, train_gen):
        model = arch.get_model(self.patch_shape, len(indicator_directories))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        epochs = self.config["epochs"]
        workers = self.config["workers"]
        if workers < 0:
            workers = os.cpu_count()//2
        model.fit_generator(generator=train_gen,
                                epochs=epochs,
                                use_multiprocessing=True,
                                workers= workers,
                                )
        return model
    
    def _reconstruct_images_from_predictions(self, predictions, patch_img_refs):
        for prediction , patch_img_ref in zip(predictions, patch_img_refs):
#             prediction = 255- (prediction*255)
            prediction = prediction*255
            img_from_patches = PatchUtils.get_image_from_patches(
                                    prediction, 
                                    patch_img_ref.bordered_img_shape,
                                    patch_img_ref.patch_window_shape)
            img_without_border = ImageUtils.remove_border(
                img_from_patches, patch_img_ref.border_top, patch_img_ref.border_left)
            img_original_size = cv2.resize(
                img_without_border, patch_img_ref.original_img_shape)
            file_name = f'{patch_img_ref.probe_file_id}.png'
            file_path = self.output_dir+file_name
            ImageUtils.save_image(img_original_size, file_path)
                 

    def _get_score(self, img_refs, output_dir, ref_data_path):
        data = MediforData.get_data(img_refs, output_dir, ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data)
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

    def _get_architecture(self):
        model_name = self.config['model_name']
        if model_name == "unet":
            from architectures.unet import UNet
            return UNet()
        elif model_name == "single_layer_nn":
            from architectures.single_layer_nn import SingleLayerNN
            return SingleLayerNN()

    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
    
if __name__ == '__main__':
    
    m = PredictionRunner()
    m.run()