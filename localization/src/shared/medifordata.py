class MediforData():
    ref = None
    sys = None
    
    def __init__(self, img_ref, sys_data_path, ref_data_path):
        self.sys = sys_data_path + img_ref.sys_mask_file_name + ".png"
        self.ref = ref_data_path + img_ref.ref_mask_file_name +".png"
        
    @staticmethod    
    def get_data(img_refs, sys_data_path, ref_data_path):
        return [MediforData(img_ref, sys_data_path, ref_data_path) for img_ref in img_refs]