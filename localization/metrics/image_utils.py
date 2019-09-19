from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ImageUtils:
    
    my_logger = None
    
    def __init__(self, my_logger):
        self.my_logger = my_logger
    
    def read_image(self, path, error_message=None):
        try:
            image = Image.open(path)
            image.load()
        except:
            error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)
            
        return image
    
    def get_black_and_white_image(self, path):
        img = self.read_image(path)
        bw = np.array(img.convert('L'))
        return np.where(bw < 255, 0, bw)
    
    def display(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()