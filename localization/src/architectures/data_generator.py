
import numpy as np
import keras
from tensorflow.python.keras.utils.data_utils import Sequence
import os

from shared.image_utils import ImageUtils

class DataGenerator(Sequence):

    def __init__(self, img_refs, targets_path, indicator_directories, indicators_path, batch_size=2, img_size=256,):
        self.img_refs = img_refs
        self.batch_size = batch_size
        self.img_size = img_size
        self.targets_path = targets_path
        self.indicator_directories= indicator_directories
        self.indicators_path = indicators_path
        self.metadata = []
        self.on_epoch_end()
        
    def __load__(self, img_ref):
        target_image_path = os.path.join(self.targets_path, "manipulation", "mask",img_ref.ref_mask_file_name)+".png"
        try:
            original_image = ImageUtils.read_image(target_image_path)
            target_image = ImageUtils.resize_image(original_image, (self.img_size, self.img_size))
            meta = MetaData(original_image.shape, img_ref.sys_mask_file_name)
            
        except ValueError as err:
            a = 1
            
        indicators = []
        for dir in self.indicator_directories:
            img_path = os.path.join(self.indicators_path, dir, "mask", img_ref.sys_mask_file_name) + ".png"
            try:
                img = ImageUtils.resize_image(ImageUtils.read_image(img_path), (self.img_size, self.img_size))
            except ValueError as err:
                img = self.handle_missing_indicator_image(target_image)
            finally:
                indicators.append(img)
        return indicators, target_image, meta

    def __getitem__(self, index, include_meta=False):
        if(index+1)*self.batch_size > len(self.img_refs):
            self.batch_size = len(self.img_refs) - index*self.batch_size

        img_refs = self.img_refs[index*self.batch_size:(index+1)*self.batch_size]   
        
        x = []
        y = []
        metas = []
        for img_ref in img_refs:
            indicators , target_image , meta = self.__load__(img_ref)
            x.append(indicators)
            y.append(target_image)
            metas.append(meta)
        x = np.array(x).reshape(-1, self.img_size, self.img_size, len(x[0]))
        x = x/255
        y = np.array(y).reshape(-1, self.img_size, self.img_size, 1)
        if include_meta is True:
            metas = np.array(metas)
            return x, y, metas
        return x, y
    
    
    def handle_missing_indicator_image(self, target_image):
        return np.zeros(target_image.shape)
    
    def __len__(self):
        return int(np.ceil(len(self.img_refs)/float(self.batch_size)))
    
    
    def on_epoch_end(self):    
        pass
    
class MetaData:
    def __init__(self, size, id):
        self.original_image_size = size
        self.probe_file_id = id
    
    