import sys
sys.path.append('../..')
from shared.json_loader import JsonLoader
import argparse

class ConfigLoader():
    
    @staticmethod
    def get_config():
        json_config, email = JsonLoader.load_config_email("predictions")
        
        parser = argparse.ArgumentParser(description='Train a model make predictions')
        
        parser.add_argument('-si','--starting_index', type=int,
                    default=json_config['starting_index'],help='Starting Index')

        parser.add_argument('-ei','--ending_index', type=int,
                    default=json_config['ending_index'],help='Ending Index')
        
        parser.add_argument('-mn','--model_name', type=str,
                    default=json_config['model_name'],help='Architecture Model Name')
 
        parser.add_argument('-e','--epochs', type=int,
                    default=json_config['epochs'],help='Num Epochs')

        parser.add_argument('-m','--multiprocessing', type=bool,
                    default=json_config['multiprocessing'],help='Multiprocessing')

        parser.add_argument('-w','--workers', type=int,
                    default=json_config['workers'],help='Num Worker')
        
        parser.add_argument('-trbs','--train_batch_size', type=int,
                    default=json_config['train_batch_size'],help='Training Batch Size')
        
        parser.add_argument('-ttbs','--test_batch_size', type=int,
                    default=json_config['test_batch_size'],help='Test Batch Size')

        parser.add_argument('-tds','--train_data_size', type=int,
                    default=json_config['train_data_size'],help='Number of images to train on')

        parser.add_argument('-dy','--dilate_y', type=str,
                    default=json_config['patch_tuning']['dilate_y'],help='Dilate y')

        parser.add_argument('-pb','--patch_black', type=str,
                    default=json_config['patch_tuning']['patch_black'],help='Patch Black')    

        parser.add_argument('-d','--data_year', type=str,
                    default=json_config['data_year'],help='Dataset')

        parser.add_argument('-idf','--image_downscale_factor', type=int,
                    default=json_config['image_downscale_factor'],help='Image Downscale Factor')
        
        parser.add_argument('-ps','--patch_shape', type=int,
                    default=json_config['patch_shape'],help='Patch Shape')
        
        parser.add_argument('-pdt','--patch_data_type', type=str,
                    default=json_config['patch_data_type'],help='Patch Data Type')

        parser.add_argument('-dp','--data_prefix', type=str,
                    default=json_config['data_prefix'],help='Data Prefix')

        parser.add_argument('-cd','--csv_data', type=str,
                    default=json_config['csv_data'],help='Csv Data')

        parser.add_argument('-nnl','--nn_layers', type=int,
                    default=json_config['nn_layers'],help='Patch Shape')

        parser.add_argument('-unl','--unet_layers', type=int,
                    default=json_config['unet_layers'],help='Patch Shape')

        parser.add_argument('-i','--iterations', type=int,
                    default=json_config['iterations'],help='Patch Shape')
        
        parser.add_argument('-r','--regularization', type=float,
                    default=json_config['regularization'],help='Patch Shape')

        parser.add_argument('-lr','--learning_rate', type=float,
                    default=json_config['learning_rate'],help='Patch Shape')
        
        parser.add_argument('-g','--graphs', type=bool,
                    default=json_config['graphs'],help='Patch Shape')

        parser.add_argument('-dt','--data_type', type=str,
                    default=json_config['data_type'],help='Patch Shape')

        config = vars(parser.parse_args())
        
        config['patch_data_type'] = ConfigLoader._get_patch_data_type(config["patch_data_type"])
        
        config['patch_tuning'] = {'dilate_y':bool(config['dilate_y']), 'patch_black':bool(config['patch_black'])}
        for param in ["patch_black", "dilate_y"]:
            del config[param]
            
        json_config.update(config)
        ConfigLoader.print_config(json_config)
        return json_config, email
    
    @staticmethod
    def print_config(config):

#         print(f"patch size_image downscale {config['patch_shape']}_{config['image_downscale_factor']}" )
#         print(f"Patch data type {config['patch_data_type'] or 'default'}" )
#         print(f"CSV Data {config['csv_data']}" )
        print(f"Data prefix {config['data_prefix']}" )
        
        
        print(f"training batch size {config['train_batch_size']}" )
        print(f"training data size {config['train_data_size']}" )
        print(f"test data size {config['ending_index'] - config['starting_index'] - config['train_data_size']}" )
        print(f"Model name {config['model_name']}")
        print(f"Nn layers {config['nn_layers']}")
        print(f"Unet layers {config['unet_layers']}")
        print(f"Num iterations {config['iterations']}" )
        print(f"Regularization {config['regularization']}" )
        print(f"Learning Rate {config['learning_rate']}" )
        print(f"Graphs {config['graphs']}" )
        print(f"Data type {config['data_type']}" )
        print(f"Image Downscale Factor {config['image_downscale_factor']}" )
        print(f"Patch Shape {config['patch_shape']}" )
    
    @staticmethod
    def _get_patch_data_type(code):
        dict = {
            "dy":"dilate_y/",
            "bb":"black_border_y/",
            "dybb":"dilate_y_black_border_y/",
            "":""
        }
        return dict[code]
        