import os
from PIL import Image
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
    
    def get_average_score(self, data):
        scores = 0
        for d in data:
            scores += self.get_image_score(d.ref, d.sys)     
        return scores/len(data)
    
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
    
    def get_folders(self, data_path):
        subfolders = [f.path for f in os.scandir(data_path) if f.is_dir() ]
        return subfolders
    
    def read_data(self, data_path):
        folders = self.get_folders(data_path)
        data = []
        for folder in folders:
            res = self.get_images(folder)
            data.append(MediforData(res['ref'], res['sys']))
        return data

        
    
    
   
            
class MediforData():
    ref = None
    sys = None
    
    def __init__(self, ref, sys):
        self.ref = ref
        self.sys = sys

metrics = Metrics()
metrics.start()  