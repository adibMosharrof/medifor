

class PathUtils:
    
    @staticmethod
    def get_paths(config_json, env_json):
        env = env_json['path']
        current_data = env['data'] + config_json["default"]["data"]
        
        image_ref_csv = current_data+ env['image_ref_csv']
        ref_data = '{}{}'.format(current_data, env["target_mask"])
        
        targets = f"{env['data']}{config_json['default']['data']}targets/"
        indicators = f"{env['data']}{config_json['default']['data']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
        
