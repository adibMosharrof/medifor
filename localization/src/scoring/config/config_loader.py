import sys
sys.path.append('../..')
from shared.json_loader import JsonLoader
import argparse

class ConfigLoader():
    
    @staticmethod
    def get_config():
        json_config, email = JsonLoader.load_config_email("scoring")
        
        parser = argparse.ArgumentParser(description='Scoring predictions')
        
        parser.add_argument('-si','--starting_index', type=int,
                    default=json_config['starting_index'],help='Starting Index')

        parser.add_argument('-ei','--ending_index', type=int,
                    default=json_config['ending_index'],help='Ending Index')
        
        parser.add_argument('-mn','--model_name', type=str,
                    default=json_config['model_name'],help='Architecture Model Name')
 
        parser.add_argument('-t','--threshold_step', type=int,
                    default=json_config['threshold_step'],help='Threshold Step')
        
        parser.add_argument('-d','--data', type=str,
                    default=json_config['data'],help='Dataset')
        
        parser.add_argument('-p','--predictions', type=str,
                    default=json_config['predictions'],help='Predictions Path')
        
        config = vars(parser.parse_args())
        json_config.update(config)
        return json_config, email