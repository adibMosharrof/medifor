import sys

sys.path.append('..')

from types import SimpleNamespace as Namespace
import socket
import json
from datetime import datetime
import os
import logging

from img_ref_builder import ImgRefBuilder
from scoring import Scoring
from shared.email_sender import EmailSender
from shared.image_utils import ImageUtils
from shared.json_loader import JsonLoader
from shared.folder_utils import FolderUtils
from shared.log_utils import LogUtils
from shared.medifordata import MediforData
from shared.timing import Timing 

class Runner():
    config_path = "../../configurations/scoring/"
    config_json = None
    env_json = None
    email_json = None
    my_logger = None
    my_timing = None
    email_sender = None
    model_name = None
    
    def __init__(self):
        self.config_json, self.env_json, self.email_json= JsonLoader.load_config_env_email(self.config_path)

        self.model_name = self.config_json["default"]["model_name"]
        output_dir = FolderUtils.create_output_folder(self.model_name,self.env_json["path"]["outputs"])
        self.my_logger = LogUtils.init_log(output_dir)
        self.my_timing = Timing(self.my_logger)
        self.log_configs()
        self.emailsender = EmailSender(self.my_logger)
        
    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
     
    def log_configs(self):
        self.my_logger.info("Threshold step: {0}".format(self.env_json["threshold_step"]))

    def start(self):
        env_path = self.env_json['path']
        sys_data_path = '{}{}'.format(env_path["model_sys_predictions"], env_path["model_name"][self.model_name])
        ref_data_path, image_ref_csv_path = PathUtils.get_image_ref_paths(self.config_json, self.env_json)

        starting_index, ending_index = self.get_data_size(self.env_json)
        irb = ImgRefBuilder(image_ref_csv_path)
        img_refs = irb.get_img_ref(starting_index, ending_index)
        data = MediforData.get_data(img_refs, sys_data_path, ref_data_path)
        self.model_scoring(data)
    
    def model_scoring(self, data):
        self.my_logger.info("Data Size {0}".format(len(data)))
        scorer = Scoring()
        try:
            scorer.start(data, self.env_json["threshold_step"])
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)

    def get_data_size(self, env_json):
        try:
          starting_index = int(env_json["data_size"]["starting_index"])
        except ValueError:
          starting_index= 0
        try:
          ending_index = int(env_json["data_size"]["ending_index"])
        except ValueError:
          ending_index = None   
        return starting_index, ending_index  
    
    def create_folder_for_output(self, model_name):
        model_dir = '{}{}/'.format(self.env_json["path"]["outputs"], model_name)
        output_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir+output_folder_name
        os.makedirs(output_dir)
        return output_dir
        
if __name__ == '__main__':
    r = Runner()
    r.start()
    r.at_exit()
