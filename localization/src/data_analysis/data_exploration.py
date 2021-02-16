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
        self._image_dimensions()
        self._image_manipulation_fraction()
       
    def _image_manipulation_fraction(self):
        fractions = self._loop_over_images(Operations.Fractions)
        cut_bins = [0,10,20,30,50,70,100]
        labels = []
        for i, v in enumerate(cut_bins):
            if i is len(cut_bins)-1:
                continue
            text = f'{cut_bins[i]}-{cut_bins[i+1]}'
            labels.append(text)
        bins = pd.cut(fractions, bins=cut_bins, labels=labels, ordered=False).value_counts()
        y = bins.values
        
        fig1, ax1 = plt.subplots(figsize=(12.8,9.6))
        wedges, texts, autotexts = ax1.pie(y, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        total = sum(y)
        pie_lables = []
        for v, l in zip(y, labels):
            value = round(v*100/total,1)
            pie_lables.append(f'{l}, {value}%')
            
        ax1.legend(wedges, pie_lables,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")
        
        ax1.set_title("Manipulation Fractions")
        plt.savefig(os.path.join(self.out_folder, f'manipulation_fractions_{self.starting_index}_{self.ending_index}.png'))
        plt.show()
    
    def _image_dimensions(self):        
        dimensions = self._loop_over_images(Operations.Dimensions)
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
        
    def _loop_over_images(self, operation, color=0):
        output= []
        for file_name in os.listdir(self.targets_path)[self.starting_index:self.ending_index]:
            path = os.path.join(self.targets_path,file_name)
            if not ImageUtils.is_image_extension(file_name):
                continue
            if operation is Operations.Dimensions:
                output.append(ImageUtils.get_image_dimensions(path))
            elif operation is Operations.Fractions:
                percentage = ImageUtils.get_color_value_percentage(path)
                output.append(percentage)
        return output

from enum import Enum
class Operations(Enum):
    Dimensions = 1
    Fractions = 2