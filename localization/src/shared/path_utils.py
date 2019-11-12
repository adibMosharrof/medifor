import os

class PathUtils:
    
    @staticmethod
    def get_paths(env_json):
        env_path = env_json['path']
        data_path = env_path['data'] + env_json['data']
        image_ref_csv, ref_data = PathUtils.get_image_ref_paths(env_path, data_path)
        
        targets = f"{env_path['data']}{env_json['data']}targets/"
        indicators = f"{env_path['data']}{env_json['data']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
    
    @staticmethod
    def get_image_ref_paths(env_path, data_path):
        image_ref_csv = data_path+ env_path['image_ref_csv']
        ref_data = '{}{}'.format(data_path, env_path["target_mask"])
        return image_ref_csv, ref_data
    
    @staticmethod
    def get_paths_for_patches(env_json):
        env_path = env_json['path']
        patches = f"{env_path['outputs']}patches/{env_json['patch_shape']}_{env_json['image_downscale_factor']}/"
        patch_img_ref_csv = f"{patches}patch_image_ref.csv"
        indicators = f"{env_path['data']}{env_json['data']}indicators/"
        data_path = env_path['data'] + env_json['data']
        img_ref_csv, ref_data = PathUtils.get_image_ref_paths(env_path, data_path)
        return patches, patch_img_ref_csv, indicators, img_ref_csv, ref_data 
    
    @staticmethod
    def get_indicator_directories(indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    