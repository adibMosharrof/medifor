import logging
import pickle
import socket
import json
import os
import cv2
import numpy as np

from scoring.img_ref_builder import ImgRefBuilder
from scoring.image_utils import ImageUtils

class MediforPredictionData:
    
    config_path = "../../configurations/predictions/"
    image_size = 256
    
    def __init__(self):
        a=1
        
    def get_data(self ):
        my_logger = logging.getLogger()
        image_utils = ImageUtils(my_logger)

        try:
            data = self.load_pickle_data()
        
        except FileNotFoundError as err:
            config_json, env_json = self.load_json_files()
            indicators_path = f"{env_json['path']['data']}{config_json['default']['data']}indicators/"
            targets_path = f"{env_json['path']['data']}{config_json['default']['data']}targets/"
            irb = ImgRefBuilder(config_json, env_json, my_logger)
            img_refs = irb.get_img_ref()[:5]
            data = self.load_data(img_refs, targets_path, indicators_path, image_utils)
            self.dump_pickle_data(data)
            
        return data
        
    def load_data(self, img_refs, targets_path, indicators_path, image_utils):
        indicator_directories = self.get_indicator_directories(indicators_path)
        data = []
        for img_ref in img_refs:
            row = Model()
#             row = {"x" : [], "y" : None}
            target_image_path = os.path.join(targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
            try:
                original_image = image_utils.read_image(target_image_path, grayscale=True)
                row.target_image = self.resize_image(original_image)
                row.original_image_size = original_image.shape
                row.probe_file_id = img_ref.sys_mask_file_name  
                
            except ValueError as err:
                a = 1
                
            for dir in indicator_directories:
                img_path = os.path.join(indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
                try:
                    img = self.resize_image(image_utils.read_image(img_path, grayscale=True))
                except ValueError as err:
                    img = self.handle_missing_indicator_image(row.target_image)
                finally:
                    row.indicators.append(img)
            data.append(row)
        return np.array(data)
        
    def handle_missing_indicator_image(self, target_image):
        return np.zeros(target_image.shape)

    def resize_image(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))
    
    def get_indicator_directories(self, indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]

    def load_json_files(self):
        hostname = socket.gethostname()
        
        with open(self.config_path+"predictions.config.json") as json_file:
            config_json = json.load(json_file)
        
        env_file_name = config_json['hostnames'][hostname]
        with open(self.config_path+env_file_name) as json_file:
            env_json = json.load(json_file)
        return config_json, env_json
            
    def dump_pickle_data(self, data ):
        pickle_out = open("data.pickle","wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def load_pickle_data(self):
        pickle_in = open("data.pickle","rb")
        return pickle.load(pickle_in)

class Model:
    def __init__(self):
        self.indicators = []
        self.target_image = ''
        self.original_image_size = ()
        self.probe_file_id =''
        