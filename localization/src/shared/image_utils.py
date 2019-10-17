import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import logging

class ImageUtils:
    
    my_logger = None
    
    @staticmethod
    def read_image(path, error_message=None):
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"No image found at the path {path}")
        except ValueError as err:
            error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
            logging.getLogger().debug(error_msg)
            raise err
            
        return image

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