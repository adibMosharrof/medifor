

class PathUtils:
    
    @staticmethod
    def get_paths_for_architecture_runner(config_json, env_json):
        env = env_json['path']
        image_ref_csv, ref_data = PathUtils._image_ref_paths(config_json, env)
        
        targets = f"{env['data']}{config_json['default']['data']}targets/"
        indicators = f"{env['data']}{config_json['default']['data']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
    
    @staticmethod
    def get_image_ref_paths(config_json, env):
        current_data = env['data'] + config_json["default"]["data"]
        image_ref_csv = current_data+ env['image_ref_csv']
        ref_data = '{}{}'.format(current_data, env["target_mask"])
        
        return image_ref_csv, ref_data