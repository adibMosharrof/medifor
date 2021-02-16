import sys
sys.path.append('../..')
from shared.json_loader import JsonLoader
import argparse

class ConfigLoader():
    
    @staticmethod
    def get_config():
        json_config = JsonLoader.load_config("data_analysis")
        
        parser = argparse.ArgumentParser(description='Break images into patches')
        
        parser.add_argument('-si','--starting_index', type=int,
                    default=json_config['starting_index'],help='Starting Index')

        parser.add_argument('-ei','--ending_index', type=int,
                    default=json_config['ending_index'],help='Ending Index')
        
        parser.add_argument('-d','--data_year', type=str,
                    default=json_config['data_year'],help='Dataset')
        
        parser.add_argument('-dp','--data_prefix', type=str,
                    default=json_config['data_prefix'],help='Data Prefix')

        parser.add_argument('-idf','--image_downscale_factor', type=int,
                    default=json_config['image_downscale_factor'],help='Image Downscale Factor')
        
        parser.add_argument('-ps','--patch_shape', type=int,
                    default=json_config['patch_shape'],help='Patch Shape')

        config = vars(parser.parse_args())
            
        json_config.update(config)
        ConfigLoader.print_config(json_config)
        return json_config, None
    
    @staticmethod
    def print_config(config):
        print(f"data prefix {config['data_prefix']}")
        print(f"data size {config['ending_index'] - config['starting_index']}" )
        print(f"Image downscale factor {config['image_downscale_factor']}")
        