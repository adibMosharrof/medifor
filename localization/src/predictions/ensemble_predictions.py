
import sys, os
from pathlib import Path
import json
import logging
import cv2
import sys
import numpy as np
import multiprocessing
from datetime import datetime
import math
from tensorflow.keras.models import load_model
import itertools
import gc
import matplotlib.pyplot as plt

sys.path.append('..')
from scoring.img_ref_builder import ImgRefBuilder
from scoring.scoring import Scoring
from shared.image_utils import ImageUtils
from shared.path_utils import PathUtils
from shared.patch_utils import PatchUtils
from shared.timing import Timing
from shared.json_loader import JsonLoader
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from shared.medifordata import MediforData
from patches.patch_image_ref import PatchImageRefFactory
from patch_predictions import PatchPredictions
from pixel_predictions import PixelPredictions

from predictions import Predictions

class Ensemble_Predictions(Predictions):
    
    def __init__(self, config):
        super().__init__(config)
        output_folder = self.config["path"]["outputs"] + "predictions/ensemble/"
        self.ensemble_model_names = config['ensemble_model_names']
        ensemble_folder, self.results_path, self.graphs_path, model_folders = FolderUtils.create_ensemble_output_folder(
            config['data_prefix'], output_folder, self.ensemble_model_names)
        self.output_dir = self.results_path
        self.my_logger = LogUtils.init_log(ensemble_folder)
        _,self.ref_data_path,self.targets_path, self.indicators_path = PathUtils.get_paths(config)
        self.preds = []
        for ensemble_model_name, model_folder in zip(self.ensemble_model_names, model_folders):
            pred = self._get_preds(ensemble_model_name, model_folder)
            self.preds.append(pred)
        
        
    def train_predict(self):
        avg_scores = []
        all_models = []
        for it in range(self.config['iterations']):
            print(f'running iteration {it}')
            score =[]
            current_models ={} 
            for model_name in self.ensemble_model_names:
                current_models[model_name]=[]
            num_img_refs = []
            for i in range(2):
                for pred in self.preds:
                    missing_probe_file_ids = []
                    if i == 1:
                        pred._flip_train_test() 
                    pred.train_gen, pred.test_gen, pred.valid_gen = pred.get_data_generators(missing_probe_file_ids)
                    model = pred.train_model(pred.train_gen, pred.valid_gen)
                    current_models[pred.model_name].append(model)
                    pred.predict(model, pred.test_gen)
#                     self.test_img_refs=self._delete_missing_probe_file_ids(missing_probe_file_ids)
                img_refs_to_score = []
                self.predict(img_refs_to_score)
                score.append(self.get_score(img_refs_to_score))
                num_img_refs.append(len(img_refs_to_score))
            all_models.append(current_models)
#             avg_score = (score[0]*len(self.train_img_refs)+ score[1]*len(self.test_img_refs))/(len(self.test_img_refs)+ len(self.train_img_refs))
            avg_score = np.average(score, axis=0, weights=num_img_refs) 
            avg_scores.append(avg_score)
        if self.config['graphs'] == True:
            self.create_graphs(all_models)
        
        for i, avg_score in enumerate(avg_scores):
            print(f'average score for iteration {i} : {avg_score}')
        print(f'max score {max(avg_scores)}')
        print(f'avg score over iterations {np.mean(avg_scores)}')
           
    def predict(self, img_refs_to_score):
        #add list of img refs that have been scored
        a = []
        img_refs = []
        for p in self.preds:
            if len(img_refs) and len(p.test_img_refs) is not len(img_refs):
                print(f'the length of img refs for the two ensemble models dont match')
                exit
            img_refs = p.test_img_refs
            a.append(p.test_img_refs)
#         imgs = np.empty([len(img_refs), len(self.preds)])
        counter = 0
        for img_ref in img_refs:
            imgs = [None]*len(self.preds)
            try:
                for i,p in enumerate(self.preds):
                    img_path = f'{p.output_dir}{img_ref.probe_file_id}.png'
                    imgs[i] = ImageUtils.read_image(img_path).ravel()
                ensemble_img_raveled = np.average(imgs, axis=0, weights=self.config['ensemble_model_weights'])
                ensemble_img = ensemble_img_raveled.reshape(img_ref.img_orig_height, img_ref.img_orig_width)
                ImageUtils.save_image(ensemble_img, f'{self.results_path}{img_ref.probe_file_id}.png')
                img_refs_to_score.append(img_ref)
            except:
                counter+=1
                continue
        print(f'didnt ensemble {counter} images')
        
    def create_graphs(self, all_models):
        for (j, ensemble_models) in enumerate(all_models):
            counter = 1
            for i,ensemble_model in enumerate(ensemble_models):
                for model in ensemble_models[ensemble_model]:
                    self.create_graph(model, j,i)
    
    def _get_preds(self, model_name, output_dir):
        if model_name in ['unet']:
            pred = PatchPredictions(self.config, model_name=model_name, output_dir=output_dir)
        else:
            pred = PixelPredictions(self.config, model_name=model_name, output_dir=output_dir)
        return pred
            
            