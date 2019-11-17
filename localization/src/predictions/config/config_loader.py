import sys
sys.path.append('../..')
from shared.json_loader import JsonLoader
import argparse

class ConfigLoader():
    
    @staticmethod
    def get_config():
        json_config, email = JsonLoader.load_env_email("predictions")
        
        parser = argparse.ArgumentParser(description='Train a model make predictions')
        
        parser.add_argument('-si','--starting_index', type=int,
                    default=json_config['starting_index'],help='Starting Index')

        parser.add_argument('-ei','--ending_index', type=int,
                    default=json_config['ending_index'],help='Ending Index')
        
        parser.add_argument('-mn','--model_name', type=str,
                    default=json_config['model_name'],help='Architecture Model Name')
 
        parser.add_argument('-e','--epochs', type=int,
                    default=json_config['epochs'],help='Num Epochs')

        parser.add_argument('-w','--workers', type=int,
                    default=json_config['workers'],help='Num Worker')
        
        parser.add_argument('-t','--threshold_step', type=int,
                    default=json_config['threshold_step'],help='Threshold Step')
        
        parser.add_argument('-trbs','--train_batch_size', type=int,
                    default=json_config['train_batch_size'],help='Training Batch Size')
        
        parser.add_argument('-ttbs','--test_batch_size', type=int,
                    default=json_config['test_batch_size'],help='Test Batch Size')

        parser.add_argument('-tds','--train_data_size', type=int,
                    default=json_config['train_data_size'],help='Number of images to train on')

        parser.add_argument('-d','--data', type=str,
                    default=json_config['data'],help='Dataset')

        parser.add_argument('-idf','--image_downscale_factor', type=int,
                    default=json_config['image_downscale_factor'],help='Image Downscale Factor')
        
        parser.add_argument('-ps','--patch_shape', type=int,
                    default=json_config['patch_shape'],help='Patch Shape')

        config = vars(parser.parse_args())
        json_config.update(config)
        return json_config, email