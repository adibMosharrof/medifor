import numpy as np
from data_generators.test_data_generator import TestDataGenerator


class PixelTestDataGenerator(TestDataGenerator):
    
    def __init__(self,
                batch_size=3, indicator_directories=[],
                shuffle=False, patches_path="", patch_shape=128,
                data_size=8, patch_img_refs=[]):
        
        self.indicator_directories = indicator_directories
        self.patch_shape = patch_shape
        
        super().__init__(batch_size=batch_size,
                        indicator_directories=indicator_directories,
                        patches_path=patches_path,
                        patch_shape=patch_shape,
                        data_size=data_size,
                        patch_img_refs=patch_img_refs) 
        
    def __getitem__(self, index):
        patch_img_refs = self.patch_img_refs[index * self.batch_size:(index + 1) * self.batch_size]   
        
        x = []
        y = []
        for patch_img_ref in patch_img_refs:
            _x, _y = self._get_row(patch_img_ref)
            x.append(_x)
            y.append(_y)
        return x, y

    def _get_row(self, patch_img_ref):
        x_patches, y_patches = super().__getitem__(patch_img_ref)
        x_patches= np.array(x_patches).reshape(-1, len(self.indicator_directories), self.patch_shape, self.patch_shape)
        y_patches = y_patches.reshape(-1, self.patch_shape, self.patch_shape)
        x = []
        y = []
        for _x, _y in zip(x_patches, y_patches):
            for (i,j) , y_value in np.ndenumerate(_y):
                y.append(y_value)
                x_inds = []
                for ind in _x:
                    x_inds.append(ind[i,j])
                x.append(x_inds)
        return x,y