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
from medifor_prediction import MediforPrediction
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
    mp = None
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
        
        self.mp = MediforPrediction(model_name)

    def start(self):
        my_logger = logging.getLogger()
        self.my_timing = Timing(my_logger)
        
        train_data_size = 10
        validation_data_size = 10
        batch_size = 2
        
        indicator_directories = self.get_indicator_directories(self.indicators_path)
        img_refs = self.irb.get_img_ref(train_data_size+validation_data_size)
        
        img_refs_validation = img_refs[train_data_size: train_data_size+validation_data_size:]
        train_gen = DataGenerator(img_refs[:train_data_size], self.targets_path, indicator_directories, self.indicators_path, batch_size) 
        validation_gen = DataGenerator(img_refs_validation, self.targets_path, indicator_directories, self.indicators_path, batch_size)
#         data = self.mp.get_data()
#         x,y = self.prep_data_for_training(data)

#         train_x, train_y, test_x, test_y, test_indices = self.get_train_test_data(x,y)
        model = self.train_model(train_gen, validation_gen)
        self.mp.reconstruct_images_from_predictions(
                                                                model, 
                                                                validation_gen, 
                                                                validation_data_size, 
                                                                batch_size,
                                                                self.output_dir)

#         item = validation_gen.getmeta()
        score = self.get_score(img_refs_validation)
        a = 1
        
    def get_score(self, img_refs):
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            scorer.start(data, self.env_json["threshold_step"])
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)

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
                                workers= 4
                                )
        print(a)
        return model
    
    def test_model(self, model, x, y):
        return model.predict(x)

    def get_train_test_data(self, x, y):
        data_split_index = len(x)//2
        train_x = x[:data_split_index]
        train_y =  y[:data_split_index]
        
        test_x = x[data_split_index:]
        test_y =  y[data_split_index:]
        
        test_indices = list(range(data_split_index, len(x)))
        return train_x, train_y, test_x, test_y, test_indices
        

    def prep_data_for_training(self, data):
#         x = [d.indicators for d in data]
#         y = [d.target_image for d in data] 

        x,y = zip(*[(d.indicators, d.target_image) for d in data])

        x = np.array(x).reshape(-1, self.image_size, self.image_size, len(x[0]))
        y = np.array(y).reshape(-1, self.image_size, self.image_size, 1)
        return x,y
    
    
    
    def get_indicator_directories(self, indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    
    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
    
if __name__ == '__main__':
    
    m = Main()
    m.start()