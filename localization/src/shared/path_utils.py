import os

class PathUtils:
    
    @staticmethod
    def get_paths(config_json, env_json):
        env = env_json['path']
        image_ref_csv, ref_data = PathUtils.get_image_ref_paths(config_json, env)
        
        targets = f"{env['data']}{config_json['default']['data']}targets/"
        indicators = f"{env['data']}{config_json['default']['data']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
    
    @staticmethod
    def get_image_ref_paths(config_json, env_path):
        current_data = PathUtils._get_current_data(config_json, env_path) 
        image_ref_csv = current_data+ env_path['image_ref_csv']
        ref_data = '{}{}'.format(current_data, env_path["target_mask"])
        
        return image_ref_csv, ref_data
    
    @staticmethod
    def get_paths_for_patches(config_json,env_json):
        patches = f"{env_json['path']['patches']}{env_json['patch_shape']}_{env_json['image_downscale_factor']}/"
        patch_img_ref_csv = f"{patches}patch_image_ref.csv"
        indicators = f"{env_json['path']['data']}{config_json['default']['data']}indicators/"
        return patches, patch_img_ref_csv, indicators
    
    @staticmethod
    def get_indicator_directories(indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    
    @staticmethod
    def _get_current_data(config_json, env_path):
        return env_path['data'] + config_json["default"]["data"]
        