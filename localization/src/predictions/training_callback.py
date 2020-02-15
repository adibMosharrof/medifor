import tensorflow as tf

class TrainingCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, pp=None, scores=None):
        self.pp = pp
        
    def on_epoch_end(self, epoch, logs=None):
        self.pp.predict(self.model, self.pp.test_gen)
        score = self.pp.get_score()
        scores = logs.get('mcc_scores', [])
        scores.append(score)
        logs.setdefault('mcc_scores', scores)
        a=1
            