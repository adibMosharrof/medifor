import sys, os
sys.path.append('..')

import math
import gc
import cv2
import numpy as np
from PIL import Image
from shared.path_utils import PathUtils
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.json_loader import JsonLoader
from shared.image_utils import ImageUtils
from shared.medifordata import MediforData

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
        
        irb = ImgRefBuilder(img_ref_csv_path)
        img_refs = irb.get_img_ref(self.starting_index, self.ending_index)
        irb.add_image_width_height(img_refs, self.config)
        self.train_img_refs = img_refs[:self.train_data_size]
        self.test_img_refs = img_refs[self.train_data_size:]
        
    def train_predict(self):
        train_gen, test_gen = self.get_data_generators()

        model = self.train_model(train_gen)
        
        self.predict(model, test_gen)
        score = self.get_score()
    
    def get_data_generators(self):
        
        csv_path = PathUtils.get_csv_data_path(self.config)
        
        if self.model_name in ['unet']:
            
            train_gen = CsvNnDataGenerator(
                        data_size=self.train_data_size,
                        test_starting_index = self.starting_index,
                        csv_path = csv_path,
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape
                        )
            test_gen = CsvNnDataGenerator(
                        data_size=self.test_data_size,
                        test_starting_index = self.train_data_size+ self.starting_index,
                        csv_path = csv_path,
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape
                        )
        else:
            train_gen = CsvPixelTrainDataGenerator(
                         data_size=self.train_data_size,
                        csv_path = csv_path
                        )
 
            test_gen = CsvPixelTestDataGenerator(
                        test_starting_index = self.ending_index,
                        data_size=self.test_data_size,
                        csv_path = csv_path,
                        img_refs = self.test_img_refs
                        )
#         a,b = train_gen.__getitem__(0)
#         q,w = test_gen.__getitem__(0)
        return train_gen, test_gen
    
    def train_model(self, train_gen):
#         try:
#             model = load_model('my_model.h5')
#             return model
#         except:
#             model = None
            
        
#         x, y = train_gen.__getitem__(0)
        x, y = train_gen.__getitem__(0)
        arch = self._get_architecture()
        if self.model_name in ['lr']:
            model = arch.get_model(self.patch_shape,x.shape[1])
        else:
            model = arch.get_model(self.patch_shape,x.shape[3])
        model.fit(x,y)
        return model
                         
    def predict(self, model, test_gen):
        predictions = []
        counter = 0
        for i in range(int(math.ceil(self.test_data_size / self.test_data_size))):
            x_list, y_list = test_gen.__getitem__(i)
            
            for i, x in enumerate(x_list):
                try:
                    pred= model.predict_proba(x)[:,1]
#                     x = np.array(x)
#                     pred = model.predict(x)
                except:
                    counter +=1
                    pred = np.zeros(self.test_img_refs[i].img_height * self.test_img_refs[i].img_width)
                predictions.append(pred)
        print(f"Num of missing images {counter}")
        del model
        gc.collect()
        self._reconstruct(predictions)
        
    def _reconstruct(self, predictions):
        for prediction , img_ref in zip(predictions, self.test_img_refs):
            prediction = 255- (prediction*255)
#             prediction = prediction * 255
            img = prediction.reshape(img_ref.img_width, img_ref.img_height)
            img_original_size = cv2.resize(
                img, (img_ref.img_orig_width, img_ref.img_orig_height))
#             img = Image.fromarray(img).convert("L") 
#             img_original_size = img.resize(
#                 (img_ref.img_orig_width, img_ref.img_orig_height), Image.ANTIALIAS)

            file_name = f'{img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)
#             img_original_size.save(file_path)
                                  
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
        return arch        