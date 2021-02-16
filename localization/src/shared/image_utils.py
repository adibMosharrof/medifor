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
            image_normalized = ImageUtils.normalize(image)
            return image_normalized 
        return image.astype(np.float32)
    
    @staticmethod
    def normalize(img):
        return np.divide(img, 255, dtype='float32')
#         return img/255
    
    @staticmethod
    def save_image(img, path, error_message=None):
        cv2.imwrite(path, img)
    
    @staticmethod
    def get_black_and_white_image(path):
        img = ImageUtils.read_image(path)
        return np.where(img < 255, 0, img)
    
    @staticmethod
    def resize_image(img, size_tuple):
        return cv2.resize(img, size_tuple)
    
    @staticmethod
    def display(*imgs):
        n = len(imgs)
        f = plt.figure()
        for i, img in enumerate(imgs):
            f.add_subplot(1, n, i + 1)
            plt.imshow(img)
        plt.show()
        
    @staticmethod
    def get_image_with_border(path, patch_shape, image_downscale_factor, border_value=[255,255,255], top=None, left=None):
#         img = ImageUtils.read_image(path, normalize=True)
        original_img = ImageUtils.read_image(path)
        img = ImageUtils.shrink_image(original_img, image_downscale_factor)
        bordered, top, left = ImageUtils.add_border(img, patch_shape, border_value=border_value, top=top, left=left)
        return bordered, top, left, original_img.shape
        
    @staticmethod
    def add_border(img, patch_shape, border_value=[255,255,255], top=None, left=None):
        top = top or ImageUtils.get_border_pixels(img.shape[0], patch_shape[0])
        left = left or ImageUtils.get_border_pixels(img.shape[1], patch_shape[1])
        bordered = cv2.copyMakeBorder(img, top=top, bottom=0, left=left, right=0, borderType=cv2.BORDER_CONSTANT, value=border_value)
        return bordered, top, left
    
    @staticmethod
    def remove_border(img, top, left):
        return img[top:, left:].copy()

    @staticmethod     
    def get_border_pixels(img_size, patch_size):
        quotient = img_size / patch_size
        if quotient % 1 == 0:
            return 0
        border_size = patch_size * math.ceil(quotient) - img_size
        return border_size
        
    @staticmethod
    def shrink_image(img, image_downscale_factor):
        return cv2.resize(img, (img.shape[1] // image_downscale_factor, img.shape[0] // image_downscale_factor))
    
    @staticmethod
    def get_shrunken_dimensions(height, width, image_downscale_factor):
        return height//image_downscale_factor, width//image_downscale_factor
           
    @staticmethod
    def dilate(img):
        kernel = np.ones((15,15),np.uint8)
        dilated_img = cv2.erode(img,kernel,iterations = 3)
        return dilated_img
    
    @staticmethod
    def get_image_dimensions(path):
        image = ImageUtils.read_image(path)
        return image.shape
            
    @staticmethod
    def is_image_extension(path):
        valid_extensions = ['png','jpeg','jpg']
        return path[-3:] in valid_extensions
    
    @staticmethod
    def get_color_value_percentage(path):
        image = ImageUtils.read_image(path)
        unique, counts = np.unique(image, return_counts=True)
        man = 0
        non_man = 0
        if len(unique) == 2:
            man = counts[0]
            non_man = counts[1]
        elif len(unique) == 1:
            if unique[0] == 0:
                man = counts[0]
            else:
                non_man = counts[0]
        return man*100/(man+non_man)
#         return dict(zip(unique,counts))
        
