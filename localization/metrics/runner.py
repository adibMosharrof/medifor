from metrics import Metrics
from email_sender import EmailSender
from img_ref_builder import ImgRefBuilder
from types import SimpleNamespace as Namespace
import socket
import json
import datetime
import os
import logging
import sys

class Runner():
    config_path = "../configurations/"
    config_json = None
    env_json = None
    email_json = None
    my_logger = None
    email_sender = None
    
    def __init__(self):
        #logging.info("starting run")
        json_files = self.load_json_files(self.config_path)
        self.env_json = json_files['env']
        self.config_json = json_files['config']
        self.email_json = json_files['email']

        output_dir = self.create_folder_for_output()
        self.my_logger = self.initiate_log(output_dir)
        self.emailsender = EmailSender(self.my_logger)
        
    
    def metric_scoring(self):
        data_path = "../data/metrics/"
        metrics = Metrics()
        data = metrics.read_data(data_path)
        metrics.start(data[2:3]) 
    
    def model_scoring(self):
        irb = ImgRefBuilder(self.config_json, self.env_json, self.my_logger)
        data = irb.get_img_ref_data()
        metrics = Metrics(self.my_logger)
        try:
            metrics.start(data, self.env_json["threshold_step"])
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
        self.emailsender.send(self.email_json)
    
    def load_json_files(self, config_path):
        hostname = socket.gethostname()
        with open(config_path+"environment.config.json") as json_file:
            config = json.load(json_file)
        
        env_file_name = config['hostnames'][hostname]
        with open(config_path+env_file_name) as json_file:
            env = json.load(json_file)
            
        with open(config_path+"email.config.json") as json_file:
            email = json.load(json_file)
            
        return {'env': env, 'config': config, 'email':email}
    
    def create_folder_for_output(self):
        model_type = self.config_json["default"]["model_type"]
        model_dir = '{}{}/'.format(self.env_json["path"]["outputs"], model_type)
        output_folder_name = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
        output_dir = model_dir+output_folder_name
        os.makedirs(output_dir)
        return output_dir
        
    def initiate_log(self, output_dir):
        logging.basicConfig(filename='{}/app.txt'.format(output_dir), level=logging.INFO, format='%(message)s')
#         logging.debug('This message should appear on the console')
#         logging.info('So should this')
#         logging.warning('And this, too')
        my_logger = logging.getLogger()
        return my_logger
        
r = Runner()
r.model_scoring()
