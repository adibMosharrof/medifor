import os
from PIL import Image
import cv2 
import numpy as np
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from shared.medifordata import MediforData
from shared.image_utils import ImageUtils
import logging
import concurrent.futures

class Scoring(object):
    thresholds = []
    starting_time = None
    end_time = None
    my_logger = None
    image_utils = None
    
    processes = []
    
    def start(self, data, threshold_step):
        self.thresholds =  np.arange(0,1, threshold_step)
        avg_score = self.get_average_score(data)
        print(avg_score)
        logging.getLogger().info('The average Score of the whole run is :' + str(avg_score))
        return avg_score
    
    def get_average_score(self, data):
        scores = 0
        for d in data:
            bw = ImageUtils.get_black_and_white_image(d.ref)
            normalized_ref = self.normalize_ref(bw)
            noscore_img = self.get_noscore_image(normalized_ref)
            sys_image = ImageUtils.read_image(d.sys)
            scores += self.get_image_score(noscore_img.ravel(), normalized_ref.ravel(), np.array(sys_image).ravel())     
        return scores/len(data)
      
    def get_noscore_image(self, img):
        baseNoScore = self.boundary_no_score(img)
        return baseNoScore 
    
    def boundary_no_score(self, img):
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
        #NOTE: dilate and erode are "reversed" because dilation and erosion in this context is relative to white, not black.
        eImg = cv2.dilate(img, erosion_kernel,iterations=1)
        dImg = cv2.erode(img, dilation_kernel,iterations=1)
        _,bns=cv2.threshold(eImg-dImg,0,1,cv2.THRESH_BINARY_INV)
        return bns
    
    def unselected_no_score(self, img):
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
        #NOTE: dilate and erode are "reversed" because dilation and erosion in this context is relative to white, not black.
        eImg = cv2.erode(img, erosion_kernel,iterations=1)
        dImg = 1-cv2.dilate(img, dilation_kernel,iterations=1)

        dImg = dImg | eImg
        weights=dImg.astype(np.uint8)
        return weights
      
    def get_image_score(self, noscore_img, ref, sys):
    
        score_with_threshold = {}
        max_score = np.NINF
        dilated_score_with_threshold = {}
        vanilla_score_with_threshold = {}
        
        for t in self.thresholds:
            self.score_threshold(t, noscore_img, ref, sys,score_with_threshold )
            
        self.plot_threshold_with_scores(score_with_threshold, vanilla_score_with_threshold)
        max_score = max(score_with_threshold.values())
        return max_score
    
    def plot_threshold_with_scores(self, dilated_score_with_threshold, vanilla_score_with_threshold):    
        plt.plot((1-self.thresholds)*255, list(dilated_score_with_threshold.values()), marker='o', color='black', markerfacecolor='b',markeredgecolor='b')
        plt.xlabel('Binarization Threshold')
        plt.ylabel('MCC')
#         plt.show()
        
    def score_threshold(self, threshold, noscore_img, ref, sys, score_with_threshold):
        scoring_indexes = self.get_scoring_indexes(noscore_img);
        predictions = self.get_sys_normalized_predictions_from_indexes(sys, scoring_indexes, threshold)
        manipulations = ref[scoring_indexes]
        score = self.get_mcc_score(predictions, manipulations)
        score_with_threshold[threshold] = score
    
    def get_scoring_indexes(self, ref):
        result = np.where(ref != 0 )
        return result[0]
    
    def get_sys_normalized_predictions_from_indexes(self, sys, indexes, threshold):
        normalized = self.normalize_ref(sys)
        filtered = normalized[indexes]
        predictions =  np.where(filtered > threshold, 1.0, 0.0) #applying the threshold
        return predictions
        
    def normalize_flip_handlezeros(self, img):
        normalized = 1 - img/255
        return normalized
    
    def normalize_ref(self, img):
        return (255-img)/255
    
    def get_mcc_score(self, predictions, manipulations):
        return matthews_corrcoef(manipulations, predictions)
    
    def read_data(self, data_path):
        folders = self.get_folders(data_path)
        data = []
        for folder in folders:
            res = self.get_images(folder)
            data.append(MediforData(res['ref'], res['sys'], folder.split('/')[-1]))
        return data
    
    def get_folders(self, data_path):
        subfolders = [f.path for f in os.scandir(data_path) if f.is_dir() ]
        return subfolders
    
    def get_images(self, folder):
        ref_path = folder+'/refMask.png'
        sys_path = folder+'/sysMask.png'
        try:
            ref_image = Image.open(ref_path)
            ref_image = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY)
        except:
            print('failed to open: %s' % ref_path)
        try:
            sys_image = Image.open(sys_path)
        except:
            print('failed to open: %s' % sys_path)
            exit
        
        return {'ref':ref_image, 'sys':sys_image}     
         

 