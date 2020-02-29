import os

class PathUtils:
    
    @staticmethod
    def get_paths(config):
        config_path = config['path']
        data_path = config_path['data'] + config['data_prefix']+ config['data_year']
        image_ref_csv, ref_data = PathUtils.get_image_ref_paths(config_path, data_path)
        
        targets = f"{config_path['data']}{config['data_prefix']}{config['data_year']}targets/"
        indicators = f"{config_path['data']}{config['data_prefix']}{config['data_year']}indicators/"
        
        return image_ref_csv, ref_data, targets, indicators
    
    @staticmethod
    def get_image_ref_paths(config_path, data_path):
        image_ref_csv = data_path+ config_path['image_ref_csv']
        ref_data = '{}{}'.format(data_path, config_path["target_mask"])
        return image_ref_csv, ref_data
    
    @staticmethod
    def get_paths_for_patches(config):
        path = config['path']
        patches = f"{path['outputs']}patches/{config['data_prefix']}/{config['data_year']}/{config['patch_data_type']}{config['patch_shape']}_{config['image_downscale_factor']}/"
        patch_img_ref_csv = f"{patches}patch_image_ref.csv"
        data_path = path['data'] + config['data_prefix'] + config['data_year']
        img_ref_csv, ref_data = PathUtils.get_image_ref_paths(path, data_path)
        return patches, patch_img_ref_csv, patches, img_ref_csv, ref_data 
    
    @staticmethod
    def get_indicator_directories(indicators_path):
#         all_inds =  [name for name in os.listdir(indicators_path)
#             if os.path.isdir(os.path.join(indicators_path, name))]    
#         indicators = list(filter(PathUtils.filter_indicators, all_inds))
#         a=1
        return ['p-uscisigradbased02a_0_2a_mediforsystem', 'p-binghamtonaggc_1_0_mediforsystem', 'p-purduepolimita11c_2_0_mediforsystem', 'p-noiseprintblindhq_1_1_mediforsystem', 'p-cresamplingdetector2st9wcm1mc_1_0_mediforsystem', 'p-kitwareberkeleyselfconsistency_84cd062_mediforsystem', 'p-ta11c_1_0_mediforsystem', 'p-purduepolimita11d_2_0_mediforsystem', 'p-cresamplingdetector2st8wcm1mc_1_0_mediforsystem', 'p-noiseprintblindpm_1_0_mediforsystem', 'p-kitwarereflectionauthentication_dcd941a2_mediforsystem', 'p-pprnudrfgclatest1_1_0_mediforsystem', 'p-umdrgbn_1_0_mediforsystem', 'p-cacontrarioresampling1st91cl_1_0_mediforsystem', 'p-ta11d_1_0_mediforsystem', 'p-fourigh_3_0_mediforsystem', 'p-plinearpattern1_1_0_mediforsystem', 'p-unifimod3_1_1_mediforsystem', 'p-kitwaredartmouthjpegdimples_0db8e4c_mediforsystem', 'p-purdueta11bmfcn6april2018_1_0_mediforsystem', 'p-cresamplingdetector1st91mc_1_0_mediforsystem', 'p-unifimod3py_1_2_mediforsystem', 'p-featuresimilarity_2_0_mediforsystem', 'p-noiseprintblindall_1_0_mediforsystem', 'p-cscmod4a1mc_1_0_mediforsystem', 'p-sriprita1imgmdlprnubased_1_0_mediforsystem', 'p-noiseprintblind_1_1_mediforsystem']
            
                    
    @staticmethod
    def filter_indicators(indicator):
        exclude = ['max','min','average', 'nb_max','nb_min','nb_average']
        for ex in exclude:
            if indicator.endswith(ex):
                return False
        return True
    
    @staticmethod
    def get_csv_data_path(config):
        return f"{config['path']['data']}{config['data_prefix']}{config['data_year']}csv_data/{config['data_year'][:-1]}{config['csv_data']}.csv"
    
    @staticmethod
    def get_index_csv_path(config):
        return f"{config['path']['data']}{config['data_prefix']}{config['data_year']}indexes/index.csv"