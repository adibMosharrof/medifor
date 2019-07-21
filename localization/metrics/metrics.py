import os
from PIL import Image
import cv2 
import numpy as np
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

class Metrics(object):
    data_path = "../data/metrics/"
    thresholds = []
    def start(self):
        #self.thresholds = np.arange(0.1, 1, 0.1)
        self.thresholds = [0.5, 0.7]
        data = self.read_data(self.data_path)
        self.get_average_score(data)
        #self.get_image_with_noscore_region(data[0])
    
    def get_average_score(self, data):
        scores = 0
        for d in data:
            img_with_noscore = self.get_image_with_noscore_region(d)
            scores += self.get_image_score(img_with_noscore, d.sys)     
        return scores/len(data)
    
    def get_image_with_noscore_region(self, data):
        '''
            Dilates the image and returns a new image that has 3 types of pixels
            0 : manipulated regions
            100: no score region
            255: non-manipulated regions
        '''
        kernel = np.ones((5,5), np.uint8) 
        img_dilation = None
        
        img = np.array(data.ref.convert('L'))
        img = np.where(img < 255, 0, img)

        img_dilated = cv2.dilate(img, kernel, iterations=4)

        img_dilated_region = img_dilated - img
        img_dilated_region = np.where(img_dilated_region == 255, 100, 255)
        
        result = img + img_dilated_region
        result = np.where(result == 255, 0 , result)
        result = np.where(result == 510, 255 , result)
        
#         plt.imshow(result, cmap='gray')
#         plt.show()
        return result
      
    def get_image_score(self, ref, sys):
        max_score = np.NINF
        for t in self.thresholds:
            score = self.get_image_score_with_threshold(ref, sys, t)
            max_score = max(max_score, score)
        return max_score
        
    def get_image_score_with_threshold(self, ref, sys, threshold):
        manipulated_indexes = self.get_manipulated_indexes_from_ref(ref);
        predictions = self.get_sys_normalized_predictions_from_indexes(sys, manipulated_indexes, threshold)
        manipulations = [1] * len(manipulated_indexes)
        return self.get_mcc_score(predictions, manipulations)
    
    def get_manipulated_indexes_from_ref(self, ref):
        gray = ref.convert('L')
        bw = gray.point(lambda x: 0 if x<255 else 255, '1')
        img_data = np.asarray(bw,dtype="int32")
        
#         f = plt.figure()
#         f.add_subplot(1,2, 1)
        plt.imshow(bw)
#         f.add_subplot(1,2, 2)
#         plt.imshow(bw1)
        plt.show(block=True)
        indexes = []
        for x in range(len(img_data)):
            for y in range(len(img_data[x])):
                if img_data[x, y] == 0:
                    indexes.append((x,y))
        
        return indexes
    
    def get_sys_normalized_predictions_from_indexes(self, sys, indexes, threshold):
        
        predictions = []
        img_data = np.asarray(sys)
        for x,y in indexes:
            normalized_value_flipped = 1 - img_data[x,y]/255
            if(normalized_value_flipped > threshold):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
    
    def get_mcc_score(self, predictions, manipulations):
        return matthews_corrcoef(manipulations, predictions)

    
    
    def read_data(self, data_path):
        folders = self.get_folders(data_path)
        data = []
        for folder in folders:
            res = self.get_images(folder)
            data.append(MediforData(res['ref'], res['sys']))
        return data
    
    def get_folders(self, data_path):
        subfolders = [f.path for f in os.scandir(data_path) if f.is_dir() ]
        return subfolders
    
    def get_images(self, folder):
        ref_path = folder+'/refMask.png'
        sys_path = folder+'/sysMask.png'
        try:
            ref_image = Image.open(ref_path)
        except:
            print('failed to open: %s' % ref_path)
        try:
            sys_image = Image.open(sys_path)
        except:
            print('failed to open: %s' % sys_path)
        
        return {'ref':ref_image, 'sys':sys_image}     
    
    
   
            
class MediforData():
    ref = None
    sys = None
    
    def __init__(self, ref, sys):
        self.ref = ref
        self.sys = sys

metrics = Metrics()
metrics.start()  