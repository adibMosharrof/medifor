import csv
from ast import literal_eval


class PatchImageRef():

    def __init__(self, id, shape, patch_window_shape):
        self.original_image_shape = shape
        self.patch_window_shape = patch_window_shape
        self.probe_file_id = id
        
    def __iter__(self):
        return iter([self.probe_file_id, self.original_image_shape, self.patch_window_shape])
    
    
class PatchImageRefFactory():

    @staticmethod
    def get_img_ref(id, shape, patch_window_shape):
        return PatchImageRef(id,shape, patch_window_shape)
    
    @staticmethod
    def get_img_refs_from_csv(csv_path, starting_index, ending_index):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            patch_img_refs = []
            for i, row in enumerate(reader):
                if i >= starting_index and i < ending_index:
                    patch_img_refs.append(PatchImageRefFactory.get_img_ref(
                        row[0], literal_eval(row[1]), literal_eval(row[2])))
                if i is ending_index:
                    break
        return patch_img_refs    
