import sys
sys.path.append('..')

import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from shared.log_utils import LogUtils
from shared.folder_utils import FolderUtils
from shared.json_loader import JsonLoader
from shared.path_utils import PathUtils
from shared.image_utils import ImageUtils

from scoring.img_ref_builder import ImgRefBuilder
from config.config_loader import ConfigLoader

plt.style.use('ggplot')

class DataExploration:
    def __init__(self):
        self.config , email_json = ConfigLoader.get_config()
        img_ref_csv_path, self.ref_data_path, self.targets_path, self.indicators_path = PathUtils.get_paths(self.config)
        self.targets_path = os.path.join(self.targets_path,"manipulation","mask")

        output_path = self.config['path']['outputs']+"data_analysis/"
        self.out_folder = FolderUtils.create_data_exploration_output_folder(output_path, self.config['data_prefix'], self.config['data_year'])
        self.starting_index = self.config['starting_index']
        self.ending_index = self.config['ending_index']
        a=1
        
    def explore(self):
       self._image_distribution()
       
    def _image_distribution(self):
        dimensions= []
        for file_name in os.listdir(self.targets_path)[self.starting_index:self.ending_index]:
            path = os.path.join(self.targets_path,file_name)
            if not ImageUtils.is_image_extension(file_name):
                continue
            dimensions.append(ImageUtils.get_image_dimensions(path=path))
        df = pd.DataFrame({"shape":dimensions})
        counts = df['shape'].value_counts()
        
        x = [str(i) for i in counts.index]
        plt.figure(figsize=(12.8,9.6))
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
        plt.bar(x,counts.values, color='green')
        plt.xlabel("Image Dimensions")
        plt.ylabel("Frequency")
        plt.title('Image Dimensions Distribution')
        plt.xticks(x[::5],  rotation='vertical')
        
        plt.savefig(os.path.join(self.out_folder, f'image_distribution_{self.starting_index}_{self.ending_index}.png'))
        
        fig, ax = plt.subplots(figsize=(12.8,9.6))
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
#         fig.figure(figsize=(12.8,9.6))
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
        
        num_items = len(counts)//5
        table_values = np.stack((counts.index[self.starting_index:self.starting_index+num_items], counts.values[self.starting_index:self.starting_index+num_items]), axis=-1)
        ax.table(cellText=table_values, colLabels=['Dimensions', 'Frequency'], loc='center')
#         fig.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'image_distribution_table_{self.starting_index}_{self.ending_index}.png'))
#         plt.show()
        

        
        