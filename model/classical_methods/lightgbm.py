from model.classical_methods.xgboost import XGBoostMethod
from copy import deepcopy
import os.path as ops
import pickle

class LightGBMMethod(XGBoostMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from lightgbm import LGBMClassifier, LGBMRegressor
        if self.is_regression:
            self.model = LGBMRegressor(**model_config,random_state=self.args.seed)
        else:
            self.model = LGBMClassifier(**model_config,random_state=self.args.seed)
    