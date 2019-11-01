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
        self.img_downscale_factor = int(self.env_json['image_downscale_factor'])
        
    def start(self):
        img_ref_csv_path, ref_data_path, targets_path, indicators_path = PathUtils.get_paths(self.config_json, self.env_json)
        irb = ImgRefBuilder(img_ref_csv_path)

        starting_index, ending_index = JsonLoader.get_data_size(self.env_json)
        img_refs = irb.get_img_ref(starting_index, ending_index)
        
        output_dir = FolderUtils.create_patch_output_folder(
            self.patch_shape, 
            self.img_downscale_factor, self.env_json["path"]["outputs"], 
            PathUtils.get_indicator_directories(indicators_path))
        
        LogUtils.init_log(output_dir)
        
        pg = PatchGenerator(
            output_dir=output_dir,
            indicators_path=indicators_path,
            img_downscale_factor = self.img_downscale_factor,
            patch_shape=self.patch_shape)
        
        pg.create_img_patches(img_refs, targets_path)

            
if __name__ == '__main__':
    pr = PatchRunner()
    pr.start()
