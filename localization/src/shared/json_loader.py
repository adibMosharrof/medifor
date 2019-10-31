import json
import socket
import logging
from json.decoder import JSONDecodeError


class JsonLoader:

    @staticmethod
    def load_config_env_email(config_path):
        hostname, file_name_suffix = JsonLoader._get_hostname_file_name_suffix(config_path)
        
        config, env = JsonLoader.load_config_env(config_path)
        email = JsonLoader.load(config_path + file_name_suffix + ".email.config.json")
            
        return config, env, email    

    @staticmethod
    def load_config_env(config_path):
        hostname, file_name_suffix = JsonLoader._get_hostname_file_name_suffix(config_path)
        config = JsonLoader.load(config_path + file_name_suffix + ".config.json")
        
        env_file_name = config['hostnames'][hostname]
        env = JsonLoader.load(config_path + file_name_suffix + '.' + env_file_name)
        return config, env
    
    @staticmethod
    def get_data_size(env_json):
        try:
          starting_index = int(env_json["data_size"]["starting_index"])
        except ValueError:
          starting_index = 0
        try:
          ending_index = int(env_json["data_size"]["ending_index"])
        except ValueError:
          ending_index = None   
        return starting_index, ending_index  

    @staticmethod
    def _get_hostname_file_name_suffix(config_path):
        return socket.gethostname(), config_path.split("/")[-2]
        


    @staticmethod
    def load(path):
        try:
            with open(path) as json_file:
                return json.load(json_file)
        except FileNotFoundError as err:
            logging.getLogger().info(f'No json file found in {path}')
            raise
        except JSONDecodeError as err:
            logging.getLogger().info(err)
            raise
            
