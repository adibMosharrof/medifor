import sys, os
sys.path.append('..')

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shared.path_utils import PathUtils
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.json_loader import JsonLoader
from shared.image_utils import ImageUtils
from shared.medifordata import MediforData

from training_callback import TrainingCallback

from sklearn.preprocessing import MinMaxScaler


class Predictions():
    
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model_name']
        
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir, self.graphs_path = FolderUtils.create_predictions_pixel_output_folder(
            self.model_name,
            self.config['data_prefix'],
            output_folder)
        print(f'Graphs path {self.graphs_path}')
        self.my_logger = LogUtils.init_log(self.output_dir)
        self.patch_shape= self.config['patch_shape']
        self._set_data_size()
    
    def train_predict(self):
        avg_scores = []
        all_models = []
        for it in range(self.config['iterations']):
            print(f'running iteration {it}')
            score =[]
            current_models = []
            for i in range(2):
                missing_probe_file_ids = []
                if i == 1:
                    print('flipping train and test set')
                    temp = self.train_img_refs
                    self.train_img_refs = self.test_img_refs
                    self.test_img_refs = temp
                    temp = len(self.train_img_refs)
                    self.train_data_size = len(self.test_img_refs)
                    self.test_data_size = temp 
                self.train_gen, self.test_gen, self.valid_gen = self.get_data_generators(missing_probe_file_ids)
                model = self.train_model(self.train_gen, self.valid_gen)
                current_models.append(model)
                self.predict(model, self.test_gen)
                img_refs_to_score=self._delete_missing_probe_file_ids(missing_probe_file_ids)
                try:
                    score.append(self.get_score(img_refs_to_score))
                except Exception:
                    print("failed in scoring")
                    exit
            all_models.append(current_models)
            avg_score = (score[0]*len(self.train_img_refs)+ score[1]*len(self.test_img_refs))/(len(self.test_img_refs)+ len(self.train_img_refs))
            avg_scores.append(avg_score)
        if self.config['graphs'] == True:
            self.create_graphs(all_models)
        
        for i, avg_score in enumerate(avg_scores):
            print(f'average score for iteration {i} : {avg_score}')
        print(f'max score {max(avg_scores)}')
        print(f'avg score over iterations {np.mean(avg_scores)}')
        
        
    def create_graphs(self, all_models):
        for (j, models) in enumerate(all_models):

            counter = 1
            for i,model in enumerate(models):
                plt.figure(figsize=(12.8,9.6))
                plt.subplot(2,1,1)
                plt.plot(model.history.history['loss'], color='b')
                plt.title('model loss training')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train'], loc='upper left')
    
                plt.subplot(2,1,2)
                plt.plot(model.history.history['val_loss'], color='r')
                plt.title('model loss test')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['test'], loc='upper left')
                
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
                plt.savefig(f'{self.graphs_path}/iteration_{j+1}_{i+1}.png')
                
    def _delete_missing_probe_file_ids(self, missing_probe_file_ids):
        start = len(self.test_img_refs)
        img_refs = [img_ref for img_ref in self.test_img_refs if img_ref.probe_file_id not in missing_probe_file_ids]
        print(f'Number of Missing Images {start - len(self.test_img_refs)}')
        return img_refs            
                
    def train_model(self, train_gen, valid_gen):
#         try:
#             model = load_model('my_model.h5')
#             return model
#         except:
#             model = None
            
        epochs = self.config["epochs"]
        workers = self.config["workers"]
        arch = self._get_architecture()
        x, y, ids = train_gen.__getitem__(0)
        if self._is_keras_pixel_model():
            model = arch.get_model(self.patch_shape,x.shape[1], config=self.config)
        elif self._is_keras_img_model():
            model = arch.get_model(self.patch_shape,len(self.indicator_directories), config=self.config)
        else:
            model = arch.get_model(self.patch_shape,x.shape[3])
#         model.fit(x,y)
#         class_weight = np.array([0.5,0.5])
#         model.fit_generator(generator = train_gen, epochs=epochs, validation_data= valid_gen, use_multiprocessing=self.config['multiprocessing'], workers = workers , class_weight=class_weight)
        train_callback = None
        if self.config['graphs']:
            train_callback = [TrainingCallback(self)]
        model.fit_generator(generator = train_gen, epochs=epochs,
            validation_data= valid_gen, 
            use_multiprocessing=self.config['multiprocessing'], 
            workers = workers,
            callbacks=train_callback)
        
        return model           
    
    
    def predict(self, model, test_gen):
        predictions = []
        counter = 0
        for i in range(int(math.ceil(self.test_data_size / self.test_batch_size))):
            x_list, y_list, ids = test_gen.__getitem__(i)
                
            for id, x in zip(ids, x_list):
                try:
                    if self._is_keras_pixel_model():
                        x = np.array(x)
                        pred = (model.predict(x), id)
                    elif self._is_keras_img_model():
                        pred = (model.predict(np.array(x)),id)
                    else:
                        pred= (model.predict_proba(x)[:,1],id) 
                except:
                    counter +=1
                    pred = (np.zeros(self.test_img_refs[i].img_height * self.test_img_refs[i].img_width), id)
                predictions.append(pred)
        print(f"Num of missing images {counter}")
        self._reconstruct(predictions, ids)            
        
    def get_score(self, img_refs):
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)
    
    def _set_data_size(self):
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        self.train_data_size = self.config['train_data_size']
        self.test_data_size = self.ending_index - self.starting_index - self.train_data_size
        self.train_batch_size = self.config['train_batch_size']
        self.test_batch_size = self.config['test_batch_size'] 
        
    def _is_keras_pixel_model(self):
        if self.config['model_name'] in ['nn', 'lr']:
#         if self.config['model_name'] in ['single_layer_nn', 'unet']:
            return True
        return False
    
    def _is_keras_img_model(self):
        if self.config['model_name'] in ['nn_img', 'unet']:
#         if self.config['model_name'] in ['single_layer_nn', 'unet']:
            return True
        return False
             
    def _get_architecture(self):
        model_name = self.config['model_name']
        if model_name == 'lr':
            from architectures.lr import Lr
            arch = Lr()
        elif model_name == 'mlp':
            from architectures.mlp import Mlp
            arch = Mlp()
        elif model_name == 'unet':
            from architectures.unet import UNet
            arch = UNet()    
        elif model_name == 'nn':
            from architectures.nn import Nn
            arch = Nn()  
        elif model_name == 'nn_img':
            from architectures.nn_img import NnImg
            arch = NnImg()    
        return arch    
        
        