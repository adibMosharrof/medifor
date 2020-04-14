import sys
sys.path.append('..')

import pandas as pd
from shared.log_utils import LogUtils
from shared.folder_utils import FolderUtils
from shared.json_loader import JsonLoader
from shared.path_utils import PathUtils

from scoring.img_ref_builder import ImgRefBuilder
from config.config_loader import ConfigLoader
from csv_to_image_generator import CsvToImageGenerator

class CsvToImageRunner:
    def __init__(self):
        self.config , email_json = ConfigLoader.get_config()
        
    def start(self):
        data_path = f"{self.config['path']['data']}{self.config['data_prefix']}{self.config['data_year']}"
        csv_path = PathUtils.get_csv_data_path(self.config)
        df = pd.read_csv(csv_path)
        
        indicators = df.columns[:-3]
        output_path = self.config['path']['outputs']+"csv_to_image/"
        output_dir, index_dir, img_ref_dir = FolderUtils.create_csv_to_image_output_folder(
            output_path,self.config['data_prefix'], 
            self.config['data_year'],indicators)
        FolderUtils.csv_to_image_copy_csv_files(data_path, index_dir, img_ref_dir)
        
        LogUtils.init_log(output_dir)
        
        cg = CsvToImageGenerator(
            config = self.config,
            output_dir=output_dir,
            df = df,
            indicators=indicators,
            index_dir = index_dir,
            img_ref_dir=img_ref_dir)
        
        cg.generate_images()

if __name__ == '__main__':
    pr = CsvToImageRunner()
    pr.start()
