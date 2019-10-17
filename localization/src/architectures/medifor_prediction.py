import logging
import pickle
import socket
import json
import os
import cv2
import numpy as np
import sys
from datetime import datetime

from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.image_utils import ImageUtils
from shared.json_loader import JsonLoader
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from shared.medifordata import MediforData

class MediforPrediction:
    
    config_path = "../../configurations/predictions/"
    image_size = 256
    config_json = None
    env_json = None
    email_json = None
    output_dir = None
    my_logger = None
    irb = None
    ref_data_path = None
    
    def __init__(self, model_name):
#         self.config_json, self.env_json , self.email_json =JsonLoader. load_config_env_email(self.config_path) 
#         self.output_dir = FolderUtils.create_output_folder(model_name,self.env_json["path"]["outputs"])
#         self.my_logger = LogUtils.init_log(self.output_dir)
#         
#         env_path = self.env_json['path']
#         current_data_path = env_path['data']+ self.config_json["default"]["data"]
#         image_ref_csv_path =  current_data_path + env_path['image_ref_csv']
#         self.ref_data_path = '{}{}'.format(current_data_path, env_path["target_mask"])
#         self.irb = ImgRefBuilder(image_ref_csv_path)
        a=1
    
    def get_score(self, img_refs):
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            scorer.start(data, self.env_json["threshold_step"])
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)

        
    def get_data(self ):
        try:
            data = self.load_pickle_data()
        
        except FileNotFoundError as err:
            indicators_path = f"{env_path['data']}{self.config_json['default']['data']}indicators/"
            targets_path = f"{env_path['data']}{self.config_json['default']['data']}targets/"
            img_refs = self.irb.get_img_ref(5)
            data = self.load_data(img_refs, targets_path, indicators_path )
            self.dump_pickle_data(data)
            
        return data
        
    def load_data(self, img_refs, targets_path, indicators_path):
        indicator_directories = self.get_indicator_directories(indicators_path)
        data = []
        for img_ref in img_refs:
            row = Model()
            target_image_path = os.path.join(targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
            try:
                original_image = ImageUtils.read_image(target_image_path)
                row.target_image = self.resize_image(original_image)
                row.original_image_size = original_image.shape
                row.probe_file_id = img_ref.sys_mask_file_name  
                
            except ValueError as err:
                a = 1
                
            for dir in indicator_directories:
                img_path = os.path.join(indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
                try:
                    img = self.resize_image(ImageUtils.read_image(img_path))
                except ValueError as err:
                    img = self.handle_missing_indicator_image(row.target_image)
                finally:
                    row.indicators.append(img)
            data.append(row)
        return np.array(data)
        
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
        
        
#         for i, row in enumerate(y):
#             pred = predictions[i]
#             # scale 0-1 values to be between 0 and 255, then subtract 255 from it
#             pred = 255 - (pred*255).astype(np.uint8)
#             resized = cv2.resize(pred, row.original_image_size)
#             file_name =  f'{row.probe_file_id}.png'
#             file_path = f'{self.output_dir}/{file_name}'
#             ImageUtils.save_image(resized,file_path)
            
    def handle_missing_indicator_image(self, target_image):
        return np.zeros(target_image.shape)

    def resize_image(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))
    
    def get_indicator_directories(self, indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]
            
    def dump_pickle_data(self, data ):
        pickle_out = open("temp/data.pickle","wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def load_pickle_data(self):
        pickle_in = open("temp/data.pickle","rb")
        return pickle.load(pickle_in)


class Model:
    def __init__(self):
        self.indicators = []
        self.target_image = ''
        self.original_image_size = ()
        self.probe_file_id =''
        