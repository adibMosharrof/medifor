import os
from PIL import Image
import cv2 
import numpy as np
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from medifordata import MediforData
from image_utils import ImageUtils
import logging
import concurrent.futures

class Metrics(object):
#     data_path = "../data/metrics/"
    thresholds = []
    starting_time = None
    end_time = None
    my_logger = None
    image_utils = None
    
    processes = []
    
    def __init__(self, logger, image_utils):
        self.my_logger = logger
        self.image_utils = image_utils
    
    def start(self, data, threshold_step):
        self.thresholds =  np.arange(0,1, threshold_step)
#         data = self.read_data(self.data_path)
        avg_score = self.get_average_score(data)
        print(avg_score)
        self.my_logger.info('The average Score of the whole run is :' + str(avg_score))
    
    def get_average_score(self, data):
        scores = 0
        for d in data:
            bw = self.image_utils.get_black_and_white_image(d.ref)
            normalized_ref = self.normalize_ref(bw)
            noscore_img = self.get_noscore_image(normalized_ref)
            sys_image = self.image_utils.read_image(d.sys)
#             noscore_img = self.get_server_dilated_image(self.data_path, d.folder_name)
            scores += self.get_image_score(noscore_img.ravel(), normalized_ref.ravel(), np.array(sys_image).ravel())     
        return scores/len(data)
    
    def get_server_dilated_image(self, path, folder_name):
        for file in os.listdir(path+folder_name):
            if file.endswith("bpm-bin.png"):
                dilated_image = Image.open(os.path.join(path+folder_name, file))
                bw = np.array(dilated_image.convert('L'))
                bw = np.where(bw ==0, 1, bw)
                bw = np.where(bw ==255, 1, bw)
                bw = np.where(bw == 225, 0, bw)
                return bw
        raise ValueError('found no file ending with bpm-bin.png')
        
    def get_noscore_image(self, img):
        baseNoScore = self.boundary_no_score(img)
        return baseNoScore
#         distractionNoScore = self.unselected_no_score(img)
#         wimg = cv2.bitwise_and(baseNoScore,distractionNoScore)
# #         plt.imshow(bns, cmap='gray')
# #         plt.show()
#         return wimg
    
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
            score = self.get_image_score_with_threshold(t, noscore_img, ref, sys, True,score_with_threshold )
            
#         for f in concurrent.futures.as_completed(self.processes):
#             print(f.result())
        self.plot_threshold_with_scores(score_with_threshold, vanilla_score_with_threshold)
        return max_score
    
    
    def plot_threshold_with_scores(self, dilated_score_with_threshold, vanilla_score_with_threshold):    
        plt.plot((1-self.thresholds)*255, list(dilated_score_with_threshold.values()), marker='o', color='black', markerfacecolor='b',markeredgecolor='b')
        plt.xlabel('Binarization Threshold')
        plt.ylabel('MCC')
#         plt.show()
        
    def get_image_score_with_threshold(self, threshold, noscore_img, ref, sys, should_dilate, score_with_threshold):
        scoring_indexes = self.get_scoring_indexes(noscore_img, should_dilate);
        predictions = self.get_sys_normalized_predictions_from_indexes(sys, scoring_indexes, threshold, should_dilate)
        manipulations = self.get_manipulations(ref, scoring_indexes, should_dilate)
        score = self.get_mcc_score(predictions, manipulations)
#         score_with_threshold[threshold] = score
        score_with_threshold[threshold] = 0
        
    
    
    def get_scoring_indexes(self, ref, should_dilate):
        if should_dilate:
            result = np.where(ref != 0 )
        else:
            result = np.where(ref != -1)
        return result[0]
    
    def get_sys_normalized_predictions_from_indexes(self, sys, indexes, threshold, should_dilate):
        normalized = self.normalize_ref(sys)
        filtered = self.filter_image_by_indexes(normalized, indexes, should_dilate)
        predictions =  np.where(filtered > threshold, 1.0, 0.0) #applying the threshold
        return predictions
        
    
    def get_manipulations(self, ref, indexes, should_dilate ):
#         normalized = self.normalize_flip_handlezeros(ref)
        filtered = self.filter_image_by_indexes(ref, indexes, should_dilate)
        return filtered
    
    def normalize_flip_handlezeros(self, img):
        normalized = 1 - img/255
        return normalized
    
    def normalize_ref(self, img):
        return (255-img)/255
    
    def filter_image_by_indexes(self, img, indexes, should_dilate):
        if(should_dilate):
            return img[indexes]
        return img
    
    def get_mcc_score(self, predictions, manipulations):
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             self.processes.append(executor.submit(matthews_corrcoef, manipulations, predictions))

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
         

 