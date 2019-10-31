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
    def get_image_ref_paths(config_json, env):
        current_data = env['data'] + config_json["default"]["data"]
        image_ref_csv = current_data+ env['image_ref_csv']
        ref_data = '{}{}'.format(current_data, env["target_mask"])
        
        return image_ref_csv, ref_data
    
    @staticmethod
    def get_indicator_directories(indicators_path):
        return [name for name in os.listdir(indicators_path)
            if os.path.isdir(os.path.join(indicators_path, name))]    
    