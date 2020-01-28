from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class Mlp():
    
    def get_model(self, image_size, num_indicators, layers=1):
        hidden_layer_sizes = [(100,), (200,), (100, 100), (200, 200)]
        activation = ['relu']
        solver = ['adam']
        
        mlp_grid = {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver}
        model = MLPClassifier((100,))
        model = GridSearchCV(model, mlp_grid, cv=2, scoring=None, verbose=2)
        return model