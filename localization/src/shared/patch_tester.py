import numpy as np
from patch_utils import PatchUtils

def my_print(item):
   print(item)
   print(item.shape) 
   print('*'*40)
   
patch_size = (2,2)
img = np.arange(16).reshape(4,4)
print(img)
print('*'*40)
patch_windows = PatchUtils.get_patches(img, patch_size)
my_print(patch_windows)
patches = patch_windows.reshape(-1,patch_size[0], patch_size[1])
my_print(patches)
recon_patch_windows = patches.reshape(patch_windows.shape)
recon = PatchUtils.get_image_from_patches(recon_patch_windows, img.shape)
my_print(recon)

cc = np.concatenate((patches, patches), axis=0)
my_print(cc)

