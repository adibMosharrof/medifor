from datetime import datetime
import os

class FolderUtils:
    @staticmethod
    def create_output_folder(model_name, output_path):
        model_dir = '{}{}/'.format(output_path, model_name)
        output_folder_name = FolderUtils._get_timestamp() 
        output_dir = model_dir+output_folder_name
        return FolderUtils.make_dir(output_dir)
    
    @staticmethod
    def create_patch_output_folder(patch_shape,img_downscale_factor,  output_path, indicators):
        output_dir_path = '{}{}_{}'.format(output_path, patch_shape, img_downscale_factor)
        output_dir =  FolderUtils.make_dir(output_dir_path)
        for indicator in indicators:
            FolderUtils.make_dir(FolderUtils.make_dir(output_dir+ indicator))
        FolderUtils.make_dir(output_dir+ 'target_image')
        return output_dir
    
    @staticmethod
    def create_predictions_output_folder(model_name, patch_shape, img_downscale_factor, output_path):
        timestamp = FolderUtils._get_timestamp()
        output_dir_path = f'{output_path}{model_name}/{patch_shape}_{img_downscale_factor}/{timestamp}'
        return FolderUtils.make_dir(output_dir_path)
        
    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def make_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError as err:
            pass
        return path + '/'