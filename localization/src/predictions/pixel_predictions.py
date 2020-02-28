import sys, os
sys.path.append('..')

import math
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
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
from sklearn.linear_model import LogisticRegression



class PixelPredictions():
    
    def __init__(self, config):
        self.config = config
        self.model_name = self.config["model_name"]
        img_downscale_factor = self.config['image_downscale_factor']
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir, self.graphs_path = FolderUtils.create_predictions_pixel_output_folder(
            self.model_name,
            self.config['data_prefix'],
            output_folder)
        print(f'Graphs path {self.graphs_path}')
        self.my_logger = LogUtils.init_log(self.output_dir)
        self.patch_shape= self.config['patch_shape']
        img_ref_csv_path, self.ref_data_path, self.targets_path, self.indicators_path = PathUtils.get_paths(self.config)
        self.indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        self.train_data_size = self.config['train_data_size']
        self.test_data_size = self.ending_index - self.starting_index - self.train_data_size 
        self.train_batch_size = self.config['train_batch_size']
        self.test_batch_size = self.config['test_batch_size']
        
        irb = ImgRefBuilder(img_ref_csv_path)
        img_refs = irb.get_img_ref(self.starting_index, self.ending_index)
        irb.add_image_width_height(img_refs, self.config)
        self.train_img_refs = img_refs[:self.train_data_size]
        self.test_img_refs = img_refs[self.train_data_size:]
        
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
                score.append(self.get_score(img_refs_to_score))
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
                
#                 plt.subplot(3,1,3)
#                 plt.plot(model.history.history['mcc_scores'][0], color='g')
#                 plt.title('Mcc Scores')
#                 plt.ylabel('mcc_score')
#                 plt.xlabel('epoch')
#                 plt.legend(['mcc_scores'], loc='upper left')
                
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
                plt.savefig(f'{self.graphs_path}/iteration_{j+1}_{i+1}.png')
    
    def _delete_missing_probe_file_ids(self, missing_probe_file_ids):
        start = len(self.test_img_refs)
        img_refs = [img_ref for img_ref in self.test_img_refs if img_ref.probe_file_id not in missing_probe_file_ids]
        print(f'Missing Images {start - len(self.test_img_refs)}')
        return img_refs
    
    def get_data_generators(self, missing_probe_file_ids):
        
        if self.model_name == "nn_img":
            from data_generators.img_train_data_generator import ImgTrainDataGenerator
            from data_generators.img_test_data_generator import ImgTestDataGenerator

            train_gen = ImgTrainDataGenerator(
                        data_size=self.train_data_size,
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.train_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )
            test_gen = ImgTestDataGenerator(
                        data_size=self.test_data_size,
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.test_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        missing_probe_file_ids = missing_probe_file_ids
                        )
            valid_gen = ImgTrainDataGenerator(
                        data_size=self.test_data_size,
                        img_refs = self.test_img_refs,
                        batch_size = self.train_batch_size,
                        patch_shape = self.patch_shape,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )  
        
        elif self.config['data_type'] == "image":
            from data_generators.img_pixel_train_data_generator import ImgPixelTrainDataGenerator
            from data_generators.img_pixel_test_data_generator import ImgPixelTestDataGenerator

            train_gen = ImgPixelTrainDataGenerator(
                        data_size=self.train_data_size,
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.train_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )
            test_gen = ImgPixelTestDataGenerator(
                        data_size=self.test_data_size,
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.test_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        missing_probe_file_ids = missing_probe_file_ids
                        )
            valid_gen = ImgPixelTrainDataGenerator(
                        data_size=self.test_data_size,
                        img_refs = self.test_img_refs,
                        batch_size = self.train_batch_size,
                        patch_shape = self.patch_shape,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = self.targets_path,
                        )    
        
        elif self.config['data_type'] == 'csv':
            csv_path = PathUtils.get_csv_data_path(self.config)
            df = pd.read_csv(csv_path)

            from data_generators.csv_pixel_test_data_generator import CsvPixelTestDataGenerator
            from data_generators.csv_pixel_train_data_generator import CsvPixelTrainDataGenerator
            from data_generators.csv_nn_data_generator import CsvNnDataGenerator
            train_gen = CsvPixelTrainDataGenerator(
                        data_size=self.train_data_size,
                        data = df,
                        img_refs = self.train_img_refs,
                        batch_size = self.train_batch_size
                        )
            valid_gen = CsvPixelTrainDataGenerator(
                        data_size=self.test_data_size,
                        data= df,
                        img_refs = self.test_img_refs,
                        batch_size = self.train_batch_size
                        )
 
            test_gen = CsvPixelTestDataGenerator(
                        data_size=self.test_data_size,
                        data = df,
                        img_refs = self.test_img_refs
                        )
#         q,w, id = test_gen.__getitem__(0)
#         a,b, id = train_gen.__getitem__(0)
        return train_gen, test_gen, valid_gen
    
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
        if self.model_name in ['lr', 'nn']:
            model = arch.get_model(self.patch_shape,x.shape[1], config=self.config)
        elif self.model_name in ['nn_img']:
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
                    if self._is_keras_img_model():
                        x = np.array([x])
                        pred = (model.predict(x),id)
                    else:
                        pred= (model.predict_proba(x)[:,1],id) 
                except:
                    counter +=1
                    pred = (np.zeros(self.test_img_refs[i].img_height * self.test_img_refs[i].img_width), id)
                predictions.append(pred)
        print(f"Num of missing images {counter}")
        del model
        gc.collect()
        self._reconstruct(predictions, ids)
        
    def _reconstruct(self, predictions, ids):
        counter = 0
        for (prediction, id) in predictions:
#             prediction = 255- (prediction*255)
#             prediction = prediction * 255
            img_ref = next((x for x in self.test_img_refs if x.probe_file_id == id), None)
            try:
                if self._is_keras_img_model():
                    pred = 255 - np.array(MinMaxScaler((0, 255)).fit_transform(prediction[0].reshape(-1, 1))).flatten()
                    img = pred.reshape(self.patch_shape, self.patch_shape)
                else:
                    pred = 255 - np.array(MinMaxScaler((0, 255)).fit_transform(prediction.reshape(-1, 1))).flatten()
                    img = pred.reshape(img_ref.img_height, img_ref.img_width)
#                 img = pred.reshape(prediction.shape[1], prediction.shape[1])
                img_original_size = cv2.resize(
                    img, (img_ref.img_orig_width, img_ref.img_orig_height))
            except:
                counter +=1
                img_original_size = np.zeros((img_ref.img_orig_width, img_ref.img_orig_height))
#             img = Image.fromarray(img).convert("L") 
#             img_original_size = img.resize(
#                 (img_ref.img_orig_width, img_ref.img_orig_height), Image.ANTIALIAS)

            file_name = f'{img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)
#             img_original_size.save(file_path)
        print(f'Number of errors {counter}') 
                                  
    def get_score(self, img_refs):
        
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)
      
    def _is_keras_pixel_model(self):
        if self.config['model_name'] in ['nn', 'lr']:
#         if self.config['model_name'] in ['single_layer_nn', 'unet']:
            return True
        return False
    def _is_keras_img_model(self):
        if self.config['model_name'] in ['nn_img', 'unet']:
#         if self.config['model_name'] in ['single_layer_nn', 'unet']:
            return True
        return False_
             
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