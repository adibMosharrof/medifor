import sys, os
sys.path.append('..')

import math
import gc
import cv2
import numpy as np
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

class PixelPredictions():
    
    def __init__(self, config):
        self.config = config
        model_name = self.config["model_name"]
        img_downscale_factor = self.config['image_downscale_factor']
        output_folder = self.config["path"]["outputs"] + "predictions/"
        self.output_dir = FolderUtils.create_predictions_pixel_output_folder(
            model_name,
            self.config['data_prefix'],
            output_folder)
        
        self.my_logger = LogUtils.init_log(self.output_dir)

        img_ref_csv_path, self.ref_data_path, targets, indicators = PathUtils.get_paths(self.config)
        self.starting_index, self.ending_index = JsonLoader.get_data_size(self.config)
        self.train_data_size = self.config['train_data_size']
        self.test_data_size = self.ending_index - self.starting_index - self.train_data_size
        
        irb = ImgRefBuilder(img_ref_csv_path)
        img_refs = irb.get_img_ref(self.starting_index, self.ending_index)
        irb.add_image_width_height(img_refs, self.config)
        self.train_img_refs = img_refs[self.starting_index:self.train_data_size]
        self.test_img_refs = img_refs[self.train_data_size:self.ending_index]
        
    def get_data_generators(self):
        
        csv_path = PathUtils.get_csv_data_path(self.config)
        
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
        return train_gen, test_gen
    
    def train_model(self, train_gen):
#         try:
#             model = load_model('my_model.h5')
#             return model
#         except:
#             model = None
            
        arch = self._get_architecture()
        model = arch.get_model(None, None)
        
        x, y = train_gen.__getitem__(0)
        model= model.fit(x,y)
        return model
                         
    def predict(self, model, test_gen):
        predictions = []
        for i in range(int(math.ceil(self.test_data_size / self.test_data_size))):
            x_list, y_list = test_gen.__getitem__(i)
            for x in x_list:
                pred = model.predict_proba(x)[:,1]
                predictions.append(pred)
        del model
        gc.collect()
        self._reconstruct(predictions)
        
    def _reconstruct(self, predictions):
        for prediction , img_ref in zip(predictions, self.test_img_refs):
#             prediction = 255- (prediction*255)
            prediction = prediction * 255
            img = prediction.reshape(img_ref.img_height, img_ref.img_width)
            img_original_size = cv2.resize(
                img, (img_ref.img_orig_height, img_ref.img_orig_width))
            
            file_name = f'{img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)
                                  
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
            
        return arch        