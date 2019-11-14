import json
import socket
import logging
from json.decoder import JSONDecodeError
import re


class JsonLoader:

    @staticmethod
    def load_env_email(module_name):
        env_name = JsonLoader.get_env_name()
        env = JsonLoader.load_env(module_name)
        email = JsonLoader.load(f"config/{module_name}.email.config.json")
            
        return env, email    

    @staticmethod
    def load_env(module_name):
        env_name = JsonLoader.get_env_name()
        env = JsonLoader.load(f"config/{module_name}.{env_name}.json")
        return env
    
    @staticmethod
    def get_data_size(env_json):
        try:
          starting_index = int(env_json["data_size"]["starting_index"])
        except ValueError:
          starting_index = 0
        try:
          ending_index = int(env_json["data_size"]["ending_index"])
        except ValueError:
          ending_index = -1   
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
                        
    @staticmethod
    def get_env_name():
        hostname = socket.gethostname()
        if hostname == "LAPTOP-DNMQC5VO":
            return "local"
        elif "uomlmedifor" in hostname:
            return "openstack" 
        elif re.search("(?<=n).\d+", x) is not None:
            return "talapas" 
        else:
            ValueError(f"Need to configure the new hostname {hostname}")
