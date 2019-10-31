import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import logging
import math
from shared.log_utils import LogUtils

class ImageUtils:
    
    my_logger = None
    
    @staticmethod
    def read_image(path, error_message=None, normalize=False):
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"No image found at the path {path}")
        except FileNotFoundError as err:
            error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
            logging.getLogger().debug(error_msg)
            raise err
        if normalize is True:
            a = ImageUtils.normalize(image)
            return a
        return image    
    
    @staticmethod
    def normalize(img):
        return np.divide(img, 255, dtype='float32')
#         return img/255
    
    @staticmethod
    def save_image(img, path, error_message=None):
        cv2.imwrite(path, img )
    
    @staticmethod
    def get_black_and_white_image(path):
        img = ImageUtils.read_image(path)
        return np.where(img < 255, 0, img)
    
    @staticmethod
    def display(img):
        plt.imshow(img, cmap='gray')
        plt.show()
        
    @staticmethod
    def resize_image(img, size_tuple):
        return cv2.resize(img, size_tuple)
    
    @staticmethod
    def display_two(img1, img2):
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(img1)
        f.add_subplot(1,2,2)
        plt.imshow(img2)
        plt.show()
        
    @staticmethod
    def display_multiple(*imgs):
        n = len(imgs)
        f = plt.figure()
        for i, img in enumerate(imgs):
            f.add_subplot(1,n,i+1)
            plt.imshow(img)
        plt.show()
        
    @staticmethod
    def read_image_add_border(path, patch_shape, vertical=None, horizontal=None):
#         img = ImageUtils.read_image(path, normalize=True)
        img1 = ImageUtils.read_image(path, normalize=True)
        img = ImageUtils.shrink_image(img1)
#         ImageUtils.display_multiple(img1, img)
        return ImageUtils.add_border(img, patch_shape, vertical=vertical, horizontal=horizontal)
        
    
    @staticmethod
    def add_border(img, patch_shape, vertical=None, horizontal=None):
        vertical = vertical or ImageUtils.get_border_pixels(img.shape[0], patch_shape[0])
        horizontal = horizontal or ImageUtils.get_border_pixels(img.shape[1], patch_shape[1])
        bordered = cv2.copyMakeBorder(img,top=vertical,bottom=vertical,left =horizontal,right=horizontal,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
        return bordered, vertical, horizontal

    @staticmethod     
    def get_border_pixels(img_size, patch_size):
        quotient = img_size / patch_size
        if quotient % 1 == 0:
            return 0
        border_size = (patch_size * math.ceil(quotient) - img_size)//2
        return border_size
        
    @staticmethod
    def shrink_image(img):
        return cv2.resize(img, (img.shape[1]//16, img.shape[0]//16))
        