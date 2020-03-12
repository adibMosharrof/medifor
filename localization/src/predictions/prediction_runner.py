import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader
from patch_predictions import PatchPredictions
from pixel_predictions import PixelPredictions
from ensemble_predictions import Ensemble_Predictions


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
        
#         train_gen, test_gen = pred.get_data_generators()
        pred.train_predict()
#         pred.predict(model, train_gen)
#         score = pred.get_score()

if __name__ == '__main__':
    m = PredictionRunner()
    m.start()
