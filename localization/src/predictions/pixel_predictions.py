import sys, os
sys.path.append('..')

import math
import gc
import cv2
import numpy as np
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from data_generators.csv_pixel_test_data_generator import CsvPixelTestDataGenerator
from data_generators.csv_pixel_train_data_generator import CsvPixelTrainDataGenerator
from data_generators.csv_nn_data_generator import CsvNnDataGenerator

class PixelPredictions():
    
    def __init__(self, config):
        self.config = config
        self.model_name = self.config["model_name"]
        img_downscale_factor = self.config['image_downscale_factor']
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir = FolderUtils.create_predictions_pixel_output_folder(
            self.model_name,
            self.config['data_prefix'],
            output_folder)
        
        self.my_logger = LogUtils.init_log(self.output_dir)
        self.patch_shape= self.config['patch_shape']
        img_ref_csv_path, self.ref_data_path, targets, indicators = PathUtils.get_paths(self.config)
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        self.train_data_size = self.config['train_data_size']
        self.test_data_size = self.ending_index - self.starting_index - self.train_data_size 
        self.train_batch_size = self.config['train_batch_size']
        
        irb = ImgRefBuilder(img_ref_csv_path)
        img_refs = irb.get_img_ref(self.starting_index, self.ending_index)
        irb.add_image_width_height(img_refs, self.config)
        self.train_img_refs = img_refs[:self.train_data_size]
        self.test_img_refs = img_refs[self.train_data_size:]
        
    def train_predict(self):
        score =[]
        for i in range(2):
            if i == 1:
                print('flipping train and test set')
                temp = self.train_img_refs
                self.train_img_refs = self.test_img_refs
                self.test_img_refs = temp
                temp = self.train_data_size
                self.train_data_size = self.test_data_size
                self.test_data_size = temp 
            train_gen, test_gen, valid_gen = self.get_data_generators()
            model = self.train_model(train_gen, valid_gen)
            self.predict(model, test_gen)
            score.append(self.get_score())
        avg_score = (score[0]*self.train_data_size + score[1]*self.test_data_size)/(self.train_data_size+ self.test_data_size)
        print(f'average score is : {avg_score}')
#         path = PathUtils.get_index_csv_path(self.config)
#         index_df = pd.read_csv(path) 
#         train_gen, test_gen = self.get_data_generators()
#         train_df = train_gen.__getitem__(0)
#         test_df = test_gen.__getitem__(0)
#         exclude = ['image_id', 'pixel_id', 'label']
#         indicators = [x for x in test_df.columns if x not in exclude]
#         X_train = train_df[indicators].values
#         y_train = train_df['label'].values
#         model =LogisticRegression() 
#         model = model.fit(X_train, y_train)
#         model = self.train_model(train_gen)
         
#         X_test = test_df[indicators].values
#         y_test = test_df['label'].values
#          
#         y_score = model.predict_proba(X_test)[:, 1]
#         test_df['y_score'] = y_score
        
#         self.predict(model, test_gen)
         
#         for i, (image_id, image_df) in enumerate(test_df.groupby('image_id')):
#             y = image_df['y_score'].values * 255
#             y = 255 - y.astype(np.uint8)
#             full_img_path = self.output_dir + image_id + '.png'
#             # get width and height and original width and height
#             r = index_df[index_df['image_id'] == image_id]
#             width, height = int(r['image_width'].values[0]), int(r['image_height'].values[0])
#             width_og, height_og = int(r['image_width_original'].values[0]), int(r['image_height_original'].values[0])
#          
#             # reshape predictions, then resize to its original dimensions
#             img = Image.fromarray(np.reshape(y, (height, width)))
#             img = img.resize((width_og, height_og), Image.ANTIALIAS)
#             img.save(full_img_path)
#         score = self.get_score()
#         print(score)
    
    def get_data_generators(self):
        
        csv_path = PathUtils.get_csv_data_path(self.config)
        df = pd.read_csv(csv_path)
        if self.model_name in ['unet']:
            
            train_gen = CsvNnDataGenerator(
                        data_size=self.train_data_size,
                        data = df,
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape
                        )
            test_gen = CsvNnDataGenerator(
                        data_size=self.test_data_size,
                        data= df,
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape
                        )
            valid_gen = CsvNnDataGenerator(
                        data_size=self.test_data_size,
                        data= df,
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape
                        )
        else:
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
#         q,w = test_gen.__getitem__(0)
#         a,b = train_gen.__getitem__(0)
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
            
        else:
            model = arch.get_model(self.patch_shape,x.shape[3])
#         model.fit(x,y)
#         class_weight = np.array([0.5,0.5])
#         model.fit_generator(generator = train_gen, epochs=epochs, validation_data= valid_gen, use_multiprocessing=self.config['multiprocessing'], workers = workers , class_weight=class_weight)
        model.fit_generator(generator = train_gen, epochs=epochs, validation_data= valid_gen, use_multiprocessing=self.config['multiprocessing'], workers = workers)
        return model
                         
    def predict(self, model, test_gen):
        predictions = []
        counter = 0
        for i in range(int(math.ceil(self.test_data_size / self.test_data_size))):
            x_list, y_list, ids = test_gen.__getitem__(i)
                
            for id, x in zip(ids, x_list):
                try:
                    if self._is_nn():
                        x = np.array(x)
                        pred = (model.predict(x), id)
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
            pred = 255 - np.array(MinMaxScaler((0, 255)).fit_transform(prediction.reshape(-1, 1))).flatten()
            try:
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
                                  
    def get_score(self):
        img_refs = self.test_img_refs
        data = MediforData.get_data(img_refs, self.output_dir, self.ref_data_path)
        scorer = Scoring()
        try:
            return scorer.start(data)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)
      
    def _is_nn(self):
        if self.config['model_name'] in ['nn', 'unet', 'lr']:
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
        return arch        