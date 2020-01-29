import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader
from patch_predictions import PatchPredictions
from pixel_predictions import PixelPredictions


class PredictionRunner():

    def start(self):
        config , email_json = ConfigLoader.get_config()
        #patch_pred = PatchPredictions(config)
        if config['model_name'] in ['unet', 'single_layer_nn']:
            pred = PatchPredictions(config)
        else:
            pred = PixelPredictions(config)
        
        train_gen, test_gen = pred.get_data_generators()
        model = pred.train_model(train_gen)
        
        pred.predict(model, test_gen)
        score = pred.get_score()

if __name__ == '__main__':
    m = PredictionRunner()
    m.start()
