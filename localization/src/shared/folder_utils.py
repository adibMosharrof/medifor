from datetime import datetime
import os
from shutil import copyfile

class FolderUtils:
    @staticmethod
    def create_output_folder(model_name, output_path):
        model_dir = '{}{}/'.format(output_path, model_name)
        output_folder_name = FolderUtils._get_timestamp() 
        output_dir = model_dir+output_folder_name
        return FolderUtils.make_dir(output_dir)
    
    @staticmethod
    def create_predictions_pixel_output_folder(model_name, data_prefix, output_path):
        timestamp = FolderUtils._get_timestamp()
        output_dir_path = f'{output_path}{model_name}/{data_prefix}/{timestamp}'
        graphs_path = f'{output_dir_path}/graphs'
        return FolderUtils.make_dir(output_dir_path), FolderUtils.make_dir(graphs_path)
    
    @staticmethod
    def create_ensemble_output_folder(data_prefix, output_path, model_names):
        timestamp = FolderUtils._get_timestamp()
        output_dir_path = f'{output_path}{data_prefix}/{timestamp}'
        results_path = f'{output_dir_path}/results'
        graphs_path = f'{results_path}/graphs'
        model_paths = []
        for model_name in model_names:
            model_path = f'{output_dir_path}/{model_name}/'
#             FolderUtils.make_dir(model_path)
            FolderUtils.make_dir(f'{model_path}graphs')
            model_paths.append(model_path)
        return FolderUtils.make_dir(output_dir_path), FolderUtils.make_dir(results_path), FolderUtils.make_dir(graphs_path), model_paths
    
    
    @staticmethod
    def create_patch_output_folder(patch_shape,img_downscale_factor,  output_path, indicators, tuning, data_year, data_prefix):
        output_dir_path = FolderUtils._get_patch_output_folder_name(output_path, patch_shape, img_downscale_factor, tuning, data_year, data_prefix)
        output_dir =  FolderUtils.make_dir(output_dir_path)
        for indicator in indicators:    
            FolderUtils.make_dir(FolderUtils.make_dir(output_dir+ indicator))
        FolderUtils.make_dir(output_dir+ 'target_image')
        return output_dir
    
    @staticmethod
    def create_predictions_output_folder(model_name, patch_shape, img_downscale_factor, output_path):
        timestamp = FolderUtils._get_timestamp()
        output_dir_path = f'{output_path}{model_name}/{patch_shape}_{img_downscale_factor}/{timestamp}'
        FolderUtils.make_dir(output_dir_path)
        return output_dir_path
    
    @staticmethod
    def create_csv_to_image_output_folder(output_path,data_prefix, data_year,indicators):
        output_dir_path = f'{output_path}{data_prefix}{data_year}'
        output_dir = FolderUtils.make_dir(output_dir_path)
        indicators_dir = FolderUtils.make_dir(output_dir_path+"indicators")
        for indicator in indicators:
            FolderUtils.make_dir(indicators_dir+ indicator+"/mask/")
        FolderUtils.make_dir(f'{output_dir}targets/manipulation/mask')
        index_dir = FolderUtils.make_dir(f'{output_dir}indexes/')
        img_ref_dir = FolderUtils.make_dir(f'{output_dir}reference/manipulation')
        return output_dir, index_dir, img_ref_dir
    
    @staticmethod
    def csv_to_image_copy_csv_files(data_path, index_dir, img_ref_dir):
        copyfile(f'{data_path}reference/manipulation/image_ref.csv', f'{img_ref_dir}image_ref.csv')
        copyfile(f'{data_path}indexes/index.csv', f'{index_dir}index.csv')
    
    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def _get_patch_output_folder_name(output_path, patch_shape,img_downscale_factor, tuning,data_year, data_prefix):
        name = ''
        for key, value in tuning.items():
            if value is True:
                name = key +'/'
        return '{}{}{}{}{}_{}'.format(output_path,data_prefix, data_year,name, patch_shape, img_downscale_factor)
        
    
    @staticmethod
    def make_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError as err:
            pass
        return path + '/'