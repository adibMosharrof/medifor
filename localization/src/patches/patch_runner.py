import sys
sys.path.append('..')

from shared.log_utils import LogUtils
from shared.folder_utils import FolderUtils
from shared.json_loader import JsonLoader
from shared.path_utils import PathUtils

from scoring.img_ref_builder import ImgRefBuilder
from patch_generator import PatchGenerator


class PatchRunner:
    config_path = "../../configurations/patches/"
    
    def __init__(self):
        self.config_json, self.env_json = JsonLoader.load_config_env(self.config_path) 
        self.patch_shape = int(self.env_json["patch_shape"])
        self.output_dir = FolderUtils.create_patch_output_folder(self.patch_shape, self.env_json["path"]["outputs"])
        self.my_logger = LogUtils.init_log(self.output_dir)
        
    def start(self):
        
        image_ref_csv_path, ref_data_path, targets_path, indicators_path = PathUtils.get_paths(self.config_json, self.env_json)
        irb = ImgRefBuilder(image_ref_csv_path)
        
        starting_index, ending_index = JsonLoader.get_data_size(self.env_json)
        indicator_directories = PathUtils.get_indicator_directories(indicators_path)
        img_refs = irb.get_img_ref(starting_index, ending_index)

        pg = PatchGenerator(
            img_refs=img_refs,
            output_dir=self.output_dir,
            targets_path=targets_path,
            indicator_directories=indicator_directories,
            indicators_path=indicators_path,
            patch_shape=self.patch_shape)
        
        pg.start()

            
if __name__ == '__main__':
    pr = PatchRunner()
    pr.start()
