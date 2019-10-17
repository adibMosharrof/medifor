from datetime import datetime
import os

class FolderUtils:
    @staticmethod
    def create_output_folder(model_name, output_path):
        model_dir = '{}{}/'.format(output_path, model_name)
        output_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir+output_folder_name
        os.makedirs(output_dir)
        return output_dir + '/'