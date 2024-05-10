from model.classical_methods.knn import KnnMethod
from copy import deepcopy
import os.path as ops
import pickle

class RandomForestMethod(KnnMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if self.is_regression:
            self.model = RandomForestRegressor(**model_config,random_state=self.args.seed)
        else:
            self.model = RandomForestClassifier(**model_config,random_state=self.args.seed)
    
    