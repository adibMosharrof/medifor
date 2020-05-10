import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader
from patch_predictions import PatchPredictions
from pixel_predictions import PixelPredictions
from ensemble_predictions import Ensemble_Predictions
from shared.path_utils import PathUtils

class PredictionRunner():

    def start(self):
        config , email_json = ConfigLoader.get_config()
        #patch_pred = PatchPredictions(config)
        if config['ensemble']:
            pred = Ensemble_Predictions(config) 
        elif config['model_name'] in ['unet']:
            pred = PatchPredictions(config)
        else:
            pred = PixelPredictions(config)
        
        if config['change_targets']:
            ind_dirs = PathUtils.get_indicator_directories(None)
            target_ids = list(range(-1,len(ind_dirs)))
#             target_ids = list(range(0,3))
            scores = {}
            for id in target_ids:
                print(f'target id {id}')
                pred.set_target_paths(id)
                max_score, avg_score = pred.train_predict()
                scores[id] = {"max_score":round(max_score,5), "avg_score":round(avg_score,5)}
            
            for key,value in scores.items():
                print(f"For target id {key} the max score is {value['max_score']} and avg score is {value['avg_score']}")
        else:
#         train_gen, test_gen = pred.get_data_generators()
            pred.train_predict()
#         pred.predict(model, train_gen)
#         score = pred.get_score()

if __name__ == '__main__':
    m = PredictionRunner()
    m.start()
