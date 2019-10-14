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
import numpy as np
import pickle
from datetime import datetime 


import matplotlib.pyplot as plt

sys.path.append('..')
from scoring.img_ref_builder import ImgRefBuilder
from scoring.image_utils import ImageUtils
from medifor_prediction_data import MediforPredictionData
from unet import UNet

class Main():
    config_path = "../../configurations/predictions/"
    indicators_path = "../../data/MFC18_EvalPart1/indicators"
    targets_path = "../../data/MFC18_EvalPart1/targets"
    env_json = None
    config_json = None
    image_utils = None
    image_size = 256
    mpd = None
    
    
    def __init__(self):
        self.mpd = MediforPredictionData()
        

    def start(self):
        my_logger = logging.getLogger()
        self.image_utils = ImageUtils(my_logger)
#         try:
#             x, y = self.load_pickle_data()
#              
#         except FileNotFoundError as err:
#             self.load_json_files()
#             self.indicators_path = f"{self.env_json['path']['data']}{self.config_json['default']['data']}indicators/"
#             self.targets_path = f"{self.env_json['path']['data']}{self.config_json['default']['data']}targets/"
#     #         self.indicators_path = '{}{}'.format(self.env_json['path']['data'], self.config_json['default']['data'])
#             irb = ImgRefBuilder(self.config_json, self.env_json, my_logger)
#             img_refs = irb.get_img_ref()[:5]
#             data = self.load_data(img_refs)
# #             data = self.mpd.get_data()
#             x, y = self.prep_data_for_training(data)
#         finally:
#             train_x, train_y, test_x, test_y, test_indices = self.get_train_test_data(x,y)
#             model = self.train_model(train_x, train_y)
#             predictions = self.test_model(model, test_x, test_y)
#             self.reconstruct_images_from_predictions(y[test_indices], predictions)
        data = self.mpd.get_data()
        x,y = self.prep_data_for_training(data)

#         x = [d.indicators for d in data]
#         y = [d.target_image for d in data] 
        train_x, train_y, test_x, test_y, test_indices = self.get_train_test_data(x,y)
        model = self.train_model(train_x, train_y)
        predictions = self.test_model(model, test_x, test_y)
        self.reconstruct_images_from_predictions(data[test_indices], predictions)

    
    def train_model(self, x, y):
        x = x/255
        unet = UNet()
        model = unet.get_model(self.image_size, x.shape[-1])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        a = model.fit(x,y, batch_size=32, epochs=1, validation_split=0.1)
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
        
        test_indices = range(data_split_index, len(x))
        return train_x, train_y, test_x, test_y, test_indices
        
#     def prep_data_for_training(self, data):
#         x = []
#         y = []
#         
#         for d in data:
#             x.append(d['x'])
#             y.append(d['y'])
#         
#         x = np.array(x).reshape(-1, self.image_size, self.image_size, len(x[0]))
#         y = np.array(y).reshape(-1, self.image_size, self.image_size, 1)
#         self.dump_pickle_data(x, y)
#         return x,y

    def prep_data_for_training(self, data):
#         x = [d.indicators for d in data]
#         y = [d.target_image for d in data] 

        x,y = zip(*[(d.indicators, d.target_image) for d in data])

        x = np.array(x).reshape(-1, self.image_size, self.image_size, len(x[0]))
        y = np.array(y).reshape(-1, self.image_size, self.image_size, 1)
#         self.dump_pickle_data(x, y)
        return x,y
        
    def dump_pickle_data(self, x , y):
        pickle_out = open("x.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    def load_pickle_data(self):
        pickle_in = open("x.pickle","rb")
        x = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)
        return x , y

    def create_train_test_splits(self):
        a=1
    
    def train_test(self):
        a = 1   
        
        
    def load_json_files(self):
        hostname = socket.gethostname()
        
        with open(self.config_path+"predictions.config.json") as json_file:
            self.config_json = json.load(json_file)
        
        env_file_name = self.config_json['hostnames'][hostname]
        with open(self.config_path+env_file_name) as json_file:
            self.env_json = json.load(json_file)
        
    def load_data(self, img_refs):
        data = []
        indicator_directories = self.get_indicator_directories()
        for img_ref in img_refs:
            row = {"x" : [], "y" : None}
            target_image_path = os.path.join(self.targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
            try:
                target_image = self.resize_image(self.image_utils.read_image(target_image_path, grayscale=True))
                row["y"] = target_image
                #add image size info here
            except ValueError as err:
                a = 1
                
            for dir in indicator_directories:
                img_path = os.path.join(self.indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
                try:
                    img = self.resize_image(self.image_utils.read_image(img_path, grayscale=True))
                except ValueError as err:
                    img = self.handle_missing_indicator_image(target_image)
                row["x"].append(img)
            data.append(row)
        return data
    
    def get_img_ref(self):
        with open(self.image_ref_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            headers = next(reader)
            all_rows = np.array(list(reader))
        #only selected images that have a reference(we are only scoring the manipulated images)
        valid_rows =all_rows[all_rows[:,4] != '']
        #only need the sys and ref image names
        required_data = valid_rows[:,[1,4]]
        sys_masks = required_data[:,0]
        ref_masks = list(map(lambda x:self.extract_ref_mask_file_name(x) ,required_data[:,1]))
        img_refs = []
        for i in range(len(sys_masks)):
            img_refs.append(ImgRefs(sys_masks[i], ref_masks[i]))
        return np.array(img_refs)
    
    def get_indicator_directories(self):
        return [name for name in os.listdir(self.indicators_path)
            if os.path.isdir(os.path.join(self.indicators_path, name))]
    
    def handle_missing_indicator_image(self, target_image):
        return np.zeros(target_image.shape)
    
    def resize_image(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))
        
    
        
    def reconstruct_images_from_predictions(self, y, predictions):
        
        for i, row in enumerate(y):
            pred = predictions[i]
            # scale 0-1 values to be between 0 and 255, then subtract 255 from it
            pred = 255 - (pred*255).astype(np.uint8)
            resized = cv2.resize(pred, row.original_image_size)
            file_path =  f'{row.probe_file_id}.png'
            self.image_utils.save_image(resized,file_path)
            
            a=1
            
        
    
if __name__ == '__main__':
    
    m = Main()
    m.start()