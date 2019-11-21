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
from shared.path_utils import PathUtils
from config.config_loader import ConfigLoader

class ScoringRunner():
    config_path = "../../configurations/scoring/"
    config_json = None
    config = None
    email_json = None
    my_logger = None
    my_timing = None
    email_sender = None
    model_name = None
    
    def __init__(self):
        self.config , self.email_json = ConfigLoader.get_config()
        self.model_name = self.config["model_name"]
        output_dir = FolderUtils.create_output_folder(self.model_name,self.config["path"]["outputs"])
        self.my_logger = LogUtils.init_log(output_dir)
        self.my_timing = Timing(self.my_logger)
        self.log_configs()
        self.emailsender = EmailSender(self.my_logger)
        
    def at_exit(self):
        self.my_timing.endlog()
        #self.emailsender.send(self.email_json)
     
    def log_configs(self):
        self.my_logger.info("Threshold step: {0}".format(self.config["threshold_step"]))

    def start(self):
        env_path = self.config['path']
        sys_data_path = f"{env_path['predictions']}{self.config['predictions']}/"
        image_ref_csv_path, ref_data_path, _, _ = PathUtils.get_paths(self.config)

        starting_index, ending_index = JsonLoader.get_data_size(self.config)
        irb = ImgRefBuilder(image_ref_csv_path)
        img_refs = irb.get_img_ref(starting_index, ending_index)
        data = MediforData.get_data(img_refs, sys_data_path, ref_data_path)
        self.model_scoring(data)
    
    def model_scoring(self, data):
        self.my_logger.info("Data Size {0}".format(len(data)))
        scorer = Scoring()
        try:
            scorer.start(data)
        except:
            error_msg = 'Program failed \n {} \n {}'.format(sys.exc_info()[0], sys.exc_info()[1])
            self.my_logger.debug(error_msg)
            sys.exit(error_msg)

    def create_folder_for_output(self, model_name):
        model_dir = '{}{}/'.format(self.config["path"]["outputs"], model_name)
        output_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir+output_folder_name
        os.makedirs(output_dir)
        return output_dir
        
if __name__ == '__main__':
    r = ScoringRunner()
    r.start()
    r.at_exit()
