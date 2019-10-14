import os
from PIL import Image
import cv2 
import numpy as np
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from metrics import Metrics
from _weakref import ref



# convert all images to grayscale and then do the work...need to revise the whole thing


class Mcctests(object):
    rows = cols = 6
    sys = None
    def start(self):
        #self.all_white_black()
#         self.plot_figure(self.get_sys())
#         self.black_left_two_rows()
#         self.black_half_rows()
#         self.white_right_two_rows()
#         self.top_left_box()
        self.left_center_box()
#         self.center_box()
#         m = Metrics()
        a = 1
    
    def get_sys(self):
        sys = []
        for r in range(self.rows):
            sys.append(np.arange(0., 1.0, 1.0/float(self.cols)))
        return np.array(sys)
     
    def get_sys_bottom(self):
        sys = []
        values = np.linspace(0, 1.0, num=self.cols)
        for r in range(self.rows):
            sys.append([values[r]]*self.cols)
        return np.array(sys)    
        
    def all_white_black(self):
        sys = self.get_sys()
        white = np.ones((self.rows, self.cols))
        black = np.zeros((self.rows, self.cols))
        self.calculate(sys, white, 'white')
        self.calculate(sys, black, 'black')   

    def black_left_two_rows(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[:,0:2] = 0
        self.plot_figure(ref)
        self.calculate(sys, ref, "left 2 rows black")
    
    def black_half_rows(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[:,0:int(self.cols/2)] = 0
        self.plot_figure(ref)
        self.calculate(sys, ref, "left half rows black")
    
    def white_right_two_rows(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[:,0:-2] = 0
        self.plot_figure(ref)
        self.calculate(sys, ref, "right 2 rows white")
        
    def top_left_box(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[0:2,0:2] = 0
        self.plot_figure(ref)
        self.calculate(sys, ref, "top left box")
        
    def left_center_box(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[int(self.rows/2-1):int(self.rows/2+1),0:2] = 0
#         self.plot_figure(ref)
        self.calculate(sys, ref, "left center box")
    
    def center_box(self):
        sys = self.get_sys()
        ref = np.ones((self.rows, self.cols))
        ref[int(self.rows/2-1):int(self.rows/2+1),int(self.rows/2-1):int(self.rows/2+1)] = 0
        self.plot_figure(ref)
        self.calculate(sys, ref, "center box")
    
    def calculate(self, sys, ref, name):
        sys_flipped = 1-sys.ravel()
        ref_flipped = 1-ref.ravel()
        threshold = np.arange(0,1, 0.1)
        white = np.ones(self.rows* self.cols)
        scores = {}
        for t in threshold:
            predictions = np.where(sys_flipped > t, 1.0, 0.0)
            scores[t] = matthews_corrcoef(ref_flipped, predictions)
        self.plot_single_score(scores, name)
        
    def plot_multiple_scores(self, black, white):
        plt.plot(black_scores, marker='o', color='black', markerfacecolor='b',markeredgecolor='b', label="black")
        plt.plot(white_scores, marker='o', color='g', markerfacecolor='r',markeredgecolor='r', label="white")    
        plt.legend()
        plt.show()
        
    def plot_single_score(self, score, name):
        plt.plot(list(score.keys()), list(score.values()), marker='o', color='black', markerfacecolor='b',markeredgecolor='b', label=name)
        plt.legend()
        plt.show()
    
    def plot_figure(self, data):
        plt.imshow(data, cmap='gray')
        plt.show()   
t = Mcctests()
t.start()  