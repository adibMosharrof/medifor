import os
import sys
sys.path.append('..')
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from shared.medifordata import MediforData
from shared.image_utils import ImageUtils
from shared.mcc_binarized import MccBinarized
import logging

class Scoring(object):
    thresholds = []
    starting_time = None
    end_time = None
    my_logger = None
    image_utils = None
    
    processes = []
    
    def start(self, data):
        self.thresholds =  np.arange(0,255, 1)
        avg_score = self.get_average_score(data)
        print(avg_score)
        logging.getLogger().info('The average Score of the whole run is :' + str(avg_score))
        return avg_score
    
    def get_average_score(self, data):
        scores = 0
        counter = 0
        for i, d in enumerate(data):
            try:
                sys_image = ImageUtils.read_image(d.sys)
                bw = ImageUtils.get_black_and_white_image(d.ref)
            except FileNotFoundError as err:
                print(f'scoring, couldnt read img with id {d.sys} at index {i}')

            normalized_ref = self.flip(bw)
            noscore_img = self.get_noscore_image(normalized_ref)
            score = self.get_image_score(noscore_img.ravel(), normalized_ref.ravel(), np.array(sys_image).ravel())     
            scores += score
            counter +=1
            print(f'running average {round(scores/(i+1),5)} current {round(score,5)}')
        return scores/counter
      
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
        scoring_indexes = self.get_scoring_indexes(noscore_img);
        normalized_pred = self.flip(sys)
        pred = normalized_pred[scoring_indexes].astype(int)
        manipulations = ref[scoring_indexes].astype(int)
        mcc_scores = MccBinarized.compute(pred, manipulations)
        max_score = max(mcc_scores)
#         self.plot_scores(mcc_scores)
        return max_score
      
    def plot_scores(self, scores):    
        plt.plot(range(len(scores)), scores)
        plt.xlabel('Binarization Threshold')
        plt.ylabel('MCC')
        plt.show()
        
    def get_scoring_indexes(self, ref):
        result = np.where(ref != 0 )
        return result[0]
    
    def flip(self, img):
#         return (255-img)/255
        return (255-img)
    