from datetime import datetime
import os

class FolderUtils:
    @staticmethod
    def create_output_folder(model_name, output_path):
        model_dir = '{}{}/'.format(output_path, model_name)
        output_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir+output_folder_name
        return FolderUtils.make_dir(output_dir)
    
    @staticmethod
    def create_patch_output_folder(patch_shape, output_path):
        output_dir = '{}{}/'.format(output_path, patch_shape)
        return FolderUtils.make_dir(output_dir)
    
    @staticmethod
    def make_dir(path):
        os.makedirs(path)
        return path + '/'