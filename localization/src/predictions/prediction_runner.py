import sys, os
sys.path.append('..')

from config.config_loader import ConfigLoader

class PredictionRunner():
    def run(self):
        config , email_json = ConfigLoader.get_config()
        prediction_handler = self._get_handler(config)
        return prediction_handler.run()
 
    def _get_handler(self, config):
        model_name = config['model_name']
        if model_name in ["unet", "single_layer_nn"]:
            from patch_predictions import PatchPredictions
            return PatchPredictions(config)
        return None

if __name__ == '__main__':
    m = PredictionRunner()
    m.run()
