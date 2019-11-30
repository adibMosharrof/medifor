import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader
from patch_predictions import PatchPredictions


class PredictionRunner():

    def start(self):
        config , email_json = ConfigLoader.get_config()
        patch_pred = PatchPredictions(config)
        
        train_gen, test_gen = patch_pred.get_data_generators()
        model = patch_pred.train_model(train_gen)
        patch_pred.predict(model, test_gen)
        score = patch_pred.get_score()
        a = 1


if __name__ == '__main__':
    m = PredictionRunner()
    m.start()
