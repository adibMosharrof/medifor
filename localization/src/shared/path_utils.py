import os

class PathUtils:
    
    @staticmethod
    def get_paths(config):
        config_path = config['path']
        data_path = config_path['data'] + config['data_prefix']+ config['data_year']
        image_ref_csv, ref_data = PathUtils.get_image_ref_paths(config_path, data_path)
        
        targets = f"{config_path['data']}{config['data_prefix']}{config['data_year']}targets/"
        indicators = f"{config_path['data']}{config['data_year']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
    
    @staticmethod
    def get_image_ref_paths(config_path, data_path):
        image_ref_csv = data_path+ config_path['image_ref_csv']
        ref_data = '{}{}'.format(data_path, config_path["target_mask"])
        return image_ref_csv, ref_data
    
    @staticmethod
    def get_paths_for_patches(config):
        path = config['path']
        patches = f"{path['outputs']}patches/{config['patch_data_type']}{config['patch_shape']}_{config['image_downscale_factor']}/"
        patch_img_ref_csv = f"{patches}patch_image_ref.csv"
        indicators = f"{path['data']}{config['data_prefix']}{config['data_year']}indicators/"
        data_path = path['data'] + config['data_prefix'] + config['data_year']
        img_ref_csv, ref_data = PathUtils.get_image_ref_paths(path, data_path)
        return patches, patch_img_ref_csv, indicators, img_ref_csv, ref_data 
    
    @staticmethod
    def get_indicator_directories(indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    