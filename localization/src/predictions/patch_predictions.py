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
        
    def get_data_generators(self):
        train, test = self._get_data_generator_names()
        if not self._isNN():
            self.train_batch_size = self.num_training_patches
            self.test_batch_size = self.test_data_size
        train_gen = train(
                        batch_size=self.train_batch_size,
                        indicator_directories=self.indicator_directories,
                        patches_path=self.patches_path,
                        patch_shape=self.patch_shape,
                        num_patches=self.num_training_patches,
                        patch_tuning = self.config["patch_tuning"]
                        )

        test_gen = test(
                        batch_size=self.test_batch_size,
                        indicator_directories=self.indicator_directories,
                        patches_path=self.patches_path,
                        patch_shape=self.patch_shape,
                        data_size=self.test_data_size,
                        patch_img_refs=self.test_patch_img_refs,
                        patch_tuning = self.config["patch_tuning"]
                        )
        
        return train_gen, test_gen
        
    def train_model(self, train_gen):
#         try:
#             model = load_model('my_model.h5')
#             return model
#         except:
#             model = None
            
        arch = self._get_architecture()
        model = arch.get_model(self.patch_shape, len(self.indicator_directories))
#         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        
        epochs = self.config["epochs"]
        workers = self.config["workers"]
        
        if self._isNN(): 
            return self._fit_nn_model(model)
        return self._fit_sklearn_model(model, train_gen)
    
    def predict(self, model, test_gen):
        predictions = []
        for i in range(int(math.ceil(self.test_data_size / self.test_batch_size))):
            x_list, y_list = test_gen.__getitem__(i)
            for x in x_list:
                if self._isNN():
                    pred = model.predict(x)
                else:
                    pred = model.predict_proba(x)
                predictions.append(pred)
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
            
            if self._isNN():
                img = self._reconstruct_image_from_patches(prediction, patch_img_ref)
            else:
                img = prediction
            img_original_size = cv2.resize(
                img, patch_img_ref.original_img_shape)
            
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
    
    def _get_data_generator_names(self):
        model_name = self.config['model_name']
        train = None
        test = None 
        if self._isNN():
            from data_generators.patch_train_data_generator import PatchTrainDataGenerator
            from data_generators.patch_test_data_generator import PatchTestDataGenerator
            train = PatchTrainDataGenerator
            test = PatchTestDataGenerator
        elif model_name in ["lr"]:
            from data_generators.pixel_train_data_generator import PixelTrainDataGenerator
            from data_generators.pixel_test_data_generator import PixelTestDataGenerator
            train = PixelTrainDataGenerator
            test = PixelTestDataGenerator
        return train, test
                
    def _get_architecture(self):
        model_name = self.config['model_name']
        if model_name == "unet":
            from architectures.unet import UNet
            arch = UNet()
        elif model_name == "single_layer_nn":
            from architectures.single_layer_nn import SingleLayerNN
            arch = SingleLayerNN()
        elif model_name == 'lr':
            from architectures.lr import Lr
            arch = Lr()
            
        return arch 
    
    def _fit_NN_model(self, model):
        epochs = self.config["epochs"]
        workers = self.config["workers"]
        
        multiprocessing = self.config["multiprocessing"]
        if multiprocessing:
            model.fit_generator(generator=train_gen,
                                epochs=epochs,
                                use_multiprocessing=True,
                                workers=workers,
                                )
        else:
            model.fit_generator(generator=train_gen,
                                epochs=epochs,
                                use_multiprocessing=False
                                )
#         model.save('my_model.h5')    
        return model
    
    def _fit_sklearn_model(self, model, train_gen):
        x, y = train_gen.__getitem__(0)
        return model.fit(x,y)
    
    def _isNN(self):
        if self.config['model_name'] in ['single_layer_nn', 'unet']:
            return True
        return False
  
  
    def my_append(self, dest, new_item):
        try:
            dest = np.array(list(itertools.chain(dest, new_item)))
        except TypeError:
            dest = new_item
        return dest 