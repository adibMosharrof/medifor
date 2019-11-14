from sklearn.feature_extraction import image
import numpy as np
from numpy.lib import stride_tricks
import tensorflow as tf
from skimage.util.shape import view_as_windows
from patchify import patchify, unpatchify 


class PatchUtils:
    @staticmethod
    def get_patches(img, patch_shape):
        try:
            patch_windows = view_as_windows(img, patch_shape, patch_shape)
        except ValueError as err:
            raise
        patches = patch_windows.reshape(-1,patch_shape[0], patch_shape[1])
        return patches, patch_windows.shape
      
    @staticmethod
    def get_image_from_patches(patches, new_image_shape, patch_window_shape):
        recon_patch_windows = patches.reshape(patch_window_shape)
        try:
            return unpatchify(recon_patch_windows, new_image_shape)
        except ZeroDivisionError as err:
            return patches[0]