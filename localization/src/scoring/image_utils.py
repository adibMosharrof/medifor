from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2

class ImageUtils:
    
    my_logger = None
    
    def __init__(self, my_logger):
        self.my_logger = my_logger
    
    def read_image(self, path, error_message=None, grayscale=False):
        try:
            if(grayscale is True):
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"No image found at the path {path}")
            else:
                image = Image.open(path)
                image.load()
        except ValueError as err:
            error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            raise err
            
        return image
    
    def save_image(self, img, path, error_message=None):
        cv2.imwrite(path, img )
    
    def get_black_and_white_image(self, path):
        img = self.read_image(path)
        bw = np.array(img.convert('L'))
        return np.where(bw < 255, 0, bw)
    
    def display(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()