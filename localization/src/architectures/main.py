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
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='2'

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

import matplotlib.pyplot as plt

sys.path.append('..')
from unet import UNet
from data_generator import DataGenerator
# from medifor_prediction import MediforPrediction
from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.image_utils import ImageUtils
from shared.timing import Timing
from shared.json_loader import JsonLoader
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from shared.medifordata import MediforData

class Main():
    config_path = "../../configurations/predictions/"
    indicators_path = "../../data/MFC18_EvalPart1/indicators"
    targets_path = "../../data/MFC18_EvalPart1/targets"
    env_json = None
    config_json = None
    image_utils = None
    image_size = 256
    my_timing = None
    
    def __init__(self):
        model_name = "unet"
        self.config_json, self.env_json , self.email_json =JsonLoader. load_config_env_email(self.config_path) 
        self.output_dir = FolderUtils.create_output_folder(model_name,self.env_json["path"]["outputs"])
        self.my_logger = LogUtils.init_log(self.output_dir)
        
        env_path = self.env_json['path']
        current_data_path = env_path['data']+ self.config_json["default"]["data"]
        image_ref_csv_path =  current_data_path + env_path['image_ref_csv']
        self.ref_data_path = '{}{}'.format(current_data_path, env_path["target_mask"])
        
        self.targets_path = f"{env_path['data']}{self.config_json['default']['data']}targets/"
        self.indicators_path = f"{env_path['data']}{self.config_json['default']['data']}indicators/"
        self.irb = ImgRefBuilder(image_ref_csv_path)
        

    def start(self):
        my_logger = logging.getLogger()
        self.my_timing = Timing(my_logger)
        
        train_data_size = 10
        validation_data_size = 10
        batch_size = 1
        
        indicator_directories = self.get_indicator_directories(self.indicators_path)
        img_refs = self.irb.get_img_ref(train_data_size+validation_data_size)
        
        img_refs_validation = img_refs[train_data_size: train_data_size+validation_data_size:]
        train_gen = DataGenerator(img_refs[:train_data_size], self.targets_path, indicator_directories, self.indicators_path, batch_size) 
        validation_gen = DataGenerator(img_refs_validation, self.targets_path, indicator_directories, self.indicators_path, batch_size)
        
        model = self.train_model(train_gen, validation_gen)
        self.reconstruct_images_from_predictions(
                                                                model, 
                                                                validation_gen, 
                                                                validation_data_size, 
                                                                batch_size,
                                                                self.output_dir)

        score = self.get_score(img_refs_validation)
        a = 1
        
    def train_model(self, train_gen, validation_gen):
#         x = x/255
        unet = UNet()
        item = train_gen.__getitem__(0)
        model = unet.get_model(self.image_size, item[0].shape[-1])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        a = model.fit_generator(generator=train_gen,
                                validation_data = validation_gen,
                                epochs=1,
                                use_multiprocessing=True,
                                workers= 1
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

    def reconstruct_images_from_predictions(self, model, validation_gen, validation_data_size, batch_size, output_dir):
        
        for i in range(validation_data_size//batch_size):
            x_list,y_list,meta_list = validation_gen.__getitem__(i, include_meta=True)
            predictions = model.predict(x_list)
            for (prediction, y, meta) in zip(predictions, y_list, meta_list):
                prediction = 255 - (prediction*255).astype(np.uint8)
                resized = cv2.resize(prediction, meta.original_image_size)
                file_name =  f'{meta.probe_file_id}.png'
                file_path = f'{output_dir}/{file_name}'
                ImageUtils.save_image(resized,file_path)

    def get_indicator_directories(self, indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    
    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
    
if __name__ == '__main__':
    
    m = Main()
    m.start()