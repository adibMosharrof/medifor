import sys
sys.path.append('..')

import pandas as pd
from shared.log_utils import LogUtils
from shared.folder_utils import FolderUtils
from shared.json_loader import JsonLoader
from shared.path_utils import PathUtils

from data_exploration import DataExploration
from config.config_loader import ConfigLoader

class DataAnalysisRunner:
    def __init__(self):
        a=1
        
    def start(self):
        
        de = DataExploration()
        de.explore()
        
#         output_dir, index_dir, img_ref_dir = FolderUtils.create_csv_to_image_output_folder(
#             output_path,self.config['data_prefix'], 
#             self.config['data_year'],indicators)
#         FolderUtils.csv_to_image_copy_csv_files(data_path, index_dir, img_ref_dir)
        
#         LogUtils.init_log(output_dir)
                

if __name__ == '__main__':
    pr = DataAnalysisRunner()
    pr.start()
