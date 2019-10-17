import json
import socket

class JsonLoader:
    @staticmethod
    def load_config_env_email(config_path):
        file_name_suffix = config_path.split("/")[-2]
        hostname = socket.gethostname()
        with open(config_path+ file_name_suffix +".config.json") as json_file:
            config = json.load(json_file)
        
        env_file_name = config['hostnames'][hostname]
        with open(config_path+file_name_suffix+'.'+env_file_name) as json_file:
            env = json.load(json_file)
            
        with open(config_path+file_name_suffix+".email.config.json") as json_file:
            email = json.load(json_file)
            
        return config, env, email    
    
