import sys
sys.path.append('../..')
from shared.json_loader import JsonLoader
import argparse

class ConfigLoader():
    
    @staticmethod
    def get_config():
        json_config = JsonLoader.load_config("patches")
        
        parser = argparse.ArgumentParser(description='Break images into patches')
        
        parser.add_argument('-si','--starting_index', type=int,
                    default=json_config['starting_index'],help='Starting Index')

        parser.add_argument('-ei','--ending_index', type=int,
                    default=json_config['ending_index'],help='Ending Index')
        
        parser.add_argument('-dy','--dilate_y', type=str,
                    default=json_config['tuning']['dilate_y'],help='Dilate y')

        parser.add_argument('-bb','--black_border_y', type=str,
                    default=json_config['tuning']['black_border_y'],help='Black Border y')    

        parser.add_argument('-dybb','--dilate_y_black_border_y', type=str,
                    default=json_config['tuning']['dilate_y_black_border_y'],help='Dilate y and Black Border y')    

        parser.add_argument('-d','--data_year', type=str,
                    default=json_config['data_year'],help='Dataset')
        
        parser.add_argument('-dp','--data_prefix', type=str,
                    default=json_config['data_prefix'],help='Data Prefix')

        parser.add_argument('-idf','--image_downscale_factor', type=int,
                    default=json_config['image_downscale_factor'],help='Image Downscale Factor')
        
        parser.add_argument('-ps','--patch_shape', type=int,
                    default=json_config['patch_shape'],help='Patch Shape')

        config = vars(parser.parse_args())
        
        if config['csv_to_image']:
            json_config['path']['data'] = json_config['path']['outputs']+"csv_to_image/"
            
        config['tuning'] = {'dilate_y':bool(config['dilate_y']), 'black_border_y':bool(config['black_border_y']), 'dilate_y_black_border_y':bool(config['dilate_y_black_border_y'])}
        for param in ["black_border_y", "dilate_y", 'dilate_y_black_border_y']:
            del config[param]
            
        json_config.update(config)
        ConfigLoader.print_config(json_config)
        return json_config, None
    
    @staticmethod
    def print_config(config):
        print(f"data prefix {config['data_prefix']}")
        print(f"data size {config['ending_index'] - config['starting_index']}" )
        print(f"Image downscale factor {config['image_downscale_factor']}")
        print(f"Tuning {config['tuning']}")
        