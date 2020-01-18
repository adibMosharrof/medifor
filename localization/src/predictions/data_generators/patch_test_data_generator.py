import numpy as np
from data_generators.test_data_generator import TestDataGenerator


class PatchTestDataGenerator(TestDataGenerator):
    
    def __init__(self,
                batch_size=3, indicator_directories=[],
                shuffle=False, patches_path="", patch_shape=128,
                data_size=8, patch_img_refs=[], patch_tuning=False):
        
       super().__init__(batch_size=batch_size,
                        indicator_directories=indicator_directories,
                        patches_path=patches_path,
                        patch_shape=patch_shape,
                        data_size=data_size,
                        patch_img_refs=patch_img_refs,
                        patch_tuning = patch_tuning) 
        
    def __getitem__(self, index):
        patch_img_refs = self.patch_img_refs[index * self.batch_size:(index + 1) * self.batch_size]   
        
        x = []
        y = []
        for patch_img_ref in patch_img_refs:
            _x, _y = super().__getitem__(patch_img_ref)
            _x = _x.reshape(-1, self.patch_shape, self.patch_shape, len(self.indicator_directories))
            _y = _y.reshape(-1, self.patch_shape, self.patch_shape, 1)
            x.append(_x)
            y.append(_y)
        return x, y

