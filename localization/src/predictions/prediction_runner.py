import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader


class PredictionRunner():

    def _get_handler(self, config):
        model_name = config['model_name']
        if model_name in ["unet", "single_layer_nn"]:
            from patch_predictions import PatchPredictions
            return PatchPredictions(config)
        return None
    
    def start(self):
        config , email_json = ConfigLoader.get_config()
        prediction_handler = self._get_handler(config)
        
        train_gen, test_gen = prediction_handler.get_data_generators()
        model = prediction_handler.train_model(train_gen)
        prediction_handler.predict(model, test_gen)
        score = prediction_handler.get_score()
        a = 1


if __name__ == '__main__':
    m = PredictionRunner()
    m.start()
