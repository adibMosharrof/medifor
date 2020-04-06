import numpy as np

from shared.image_utils import ImageUtils
from tensorflow.python.keras.utils.data_utils import Sequence

class PatchTestDataGenerator(Sequence):
    
    def __init__(self, batch_size=10, indicator_directories=[], 
                shuffle=False, patches_path="", patch_shape=128,
                num_patches=8,patch_tuning=None, img_refs=None, data_size = 100,
                targets_path = "", indicators_path="", missing_probe_file_ids = None
                ):
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.img_refs = img_refs
        self.data_size = data_size
        self.indicators_path = indicators_path
        self.indicator_directories = indicator_directories
        self.targets_path = targets_path
        self.missing_probe_file_ids = missing_probe_file_ids
        
    def __getitem__(self, index):
        starting_index = index * self.batch_size
        ending_index = (index + 1) * self.batch_size
        
        img_refs = self.img_refs[starting_index:ending_index]

        target_imgs = []
        target_imgs = self._read_images_from_directory(self.targets_path, img_refs)
        
        indicator_imgs = [None]*len(img_refs)
        for i,img_ref in enumerate(img_refs):
            indicator_imgs[i] = self._read_indicators(img_ref)

        if len(indicator_imgs) ==0:
            indicator_imgs = np.empty([0,len(self.indicator_directories)])
        return np.array(indicator_imgs), np.array(target_imgs), [i.probe_file_id for i in img_refs]
        
    def _read_indicators(self, img_ref):
        
        num_images = img_ref.patch_window_shape[0] * img_ref.patch_window_shape[1]
        all_indicators = [None]*num_images
        for j in range(num_images):
            #indicators = []
            indicators = np.empty([self.patch_shape, self.patch_shape, len(self.indicator_directories)])
            for i, indicator_name in enumerate(self.indicator_directories):
                indicator_path = f'{self.indicators_path}{indicator_name}/{img_ref.probe_file_id}_{j}.png'
                try:
                    img = ImageUtils.read_image(indicator_path)
                    ind_img = 255 - img 
                except FileNotFoundError as err:
                    ind_img = np.zeros([self.patch_shape, self.patch_shape])
                indicators[:,:,i] = ind_img
            all_indicators[j] =indicators
        return all_indicators
    
    def _read_images_from_directory(self, dir_path, img_refs):
        all_imgs = [None]*len(img_refs)
        for (i,img_ref) in enumerate(img_refs):
            num_images = img_ref.patch_window_shape[0] * img_ref.patch_window_shape[1]
            imgs = [None]*num_images
            for j in range(num_images):
                img_path = f'{dir_path}/{img_ref.probe_file_id}_{j}.png'
                try:
                    img = ImageUtils.read_image(img_path)
                    flipped_img = 255-img
                    thresholded_img = np.where(flipped_img > 127, 1,0)
                    imgs[j] = thresholded_img.reshape(self.patch_shape, self.patch_shape, 1)
                except FileNotFoundError as err:
                    print(f'could not find file with id {img_ref.probe_file_id}')
                    self.missing_probe_file_ids.append(img_ref.probe_file_id)
            all_imgs[i]=imgs
        return all_imgs
    
    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))

