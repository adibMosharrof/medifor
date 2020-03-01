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

from predictions import Predictions

class PixelPredictions(Predictions):
    
    def __init__(self, config):
        super().__init__(config)
        img_ref_csv_path, self.ref_data_path, self.targets_path, self.indicators_path = PathUtils.get_paths(self.config)
        self.indicator_directories = PathUtils.get_indicator_directories(self.indicators_path)
        
        self._prepare_img_refs(img_ref_csv_path)
        
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
                img_original_size = cv2.resize(
                    img, (img_ref.img_orig_width, img_ref.img_orig_height))
            except:
                counter +=1
                img_original_size = np.zeros((img_ref.img_orig_width, img_ref.img_orig_height))

            file_name = f'{img_ref.probe_file_id}.png'
            file_path = self.output_dir + file_name
            ImageUtils.save_image(img_original_size, file_path)
        print(f'Number of errors in reconstruction {counter}') 
                                  
    def _prepare_img_refs(self, img_ref_csv_path):
        irb = ImgRefBuilder(img_ref_csv_path)
        img_refs = irb.get_img_ref(self.starting_index, self.ending_index)
        ImgRefBuilder.add_image_width_height(img_refs, self.config)
        self.train_img_refs = img_refs[:self.train_data_size]
        self.test_img_refs = img_refs[self.train_data_size:]  
      
    def get_data_generators(self, missing_probe_file_ids):
        if self._is_keras_img_model():
            from data_generators.img_train_data_generator import ImgTrainDataGenerator
            from data_generators.img_test_data_generator import ImgTestDataGenerator

            targets_path = os.path.join(self.targets_path, "manipulation","mask")
            
            train_gen = ImgTrainDataGenerator(
                        data_size=len(self.train_img_refs),
                        img_refs = self.train_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.train_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = targets_path,
                        )
            test_gen = ImgTestDataGenerator(
                        data_size=len(self.test_img_refs),
                        img_refs = self.test_img_refs,
                        patch_shape = self.patch_shape,
                        batch_size = self.test_batch_size,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = targets_path,
                        missing_probe_file_ids = missing_probe_file_ids
                        )
            valid_gen = ImgTrainDataGenerator(
                        data_size=len(self.test_img_refs),
                        img_refs = self.test_img_refs,
                        batch_size = self.train_batch_size,
                        patch_shape = self.patch_shape,
                        indicator_directories = self.indicator_directories,
                        indicators_path = self.indicators_path,
                        targets_path = targets_path,
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
     