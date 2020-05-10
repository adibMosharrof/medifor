import csv
from ast import literal_eval
import math
import sys
sys.path.append('..')
from scoring.img_ref_builder import ImgRefs

class PatchImageRef(ImgRefs):

    def __init__(self, id, bordered_img_shape, patch_window_shape, 
                probe_mask_file_name, original_img_shape,
                border_top, border_left):
        self.bordered_img_shape = bordered_img_shape
        self.patch_window_shape = patch_window_shape
        self.probe_file_id = id
        self.probe_mask_file_name = probe_mask_file_name
        self.original_img_shape = original_img_shape
        self.border_top = border_top
        self.border_left = border_left

        
    def __iter__(self):
        return iter([self.probe_file_id, self.bordered_img_shape,
                    self.patch_window_shape, self.probe_mask_file_name, 
                    self.original_img_shape, self.border_top, self.border_left])
    
    
class PatchImageRefFactory():

    @staticmethod
    def create_img_ref(id, bordered_img_shape, patch_window_shape,
                       probe_mask_file_name, original_img_shape,
                       border_top, border_left):
        return PatchImageRef(id,bordered_img_shape, patch_window_shape,
                            probe_mask_file_name, original_img_shape,
                            border_top, border_left)
    
    @staticmethod
    def get_img_refs_from_csv(csv_path, starting_index, ending_index, target_index=-1):
        if ending_index is -1:
            ending_index = math.inf
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            patch_img_refs = []
            ti = 3 if target_index is -1 else 0
            for i, row in enumerate(reader):
                if i >= starting_index and i < ending_index:
                    patch_img_refs.append(PatchImageRefFactory.create_img_ref(
                        row[0], 
                        literal_eval(row[1]), 
                        literal_eval(row[2]),
                        row[ti],
                        literal_eval(row[4]),
                        int(row[5]),
                        int(row[6])
                    ))
                if i is ending_index:
                    break
        if ending_index == math.inf:
            ending_index = len(patch_img_refs)
        return patch_img_refs, ending_index
