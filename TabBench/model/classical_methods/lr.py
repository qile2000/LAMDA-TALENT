from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
import time
import sklearn.metrics as skm
import numpy as np
class LinearRegressionMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(is_regression)
        assert(not args.tune)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(**model_config)
    
    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        tic = time.time()
        self.model.fit(self.N['train'], self.y['train'])
        self.trlog['best_res'] = self.model.score(self.N['val'], self.y['val'])
        time_cost = time.time() - tic
        with open(ops.join(self.args.save_path , 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)
        return time_cost
        
    
    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path , 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        test_logit = self.model.predict(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit
    
    def metric(self, predictions, labels, y_info):
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy()
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae,r2,rmse), ("MAE", "R2", "RMSE")
        else:
            raise ValueError("Unknown tabular task type")