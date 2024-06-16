from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
from model.lib.data import (
    Dataset,
)
from model.utils import (
    get_device
)
import numpy as np
import time
from sklearn.metrics import accuracy_score, mean_squared_error

class CatBoostMethod(classical_methods):
    def __init__(self, args, is_regression):
        self.args = args
        print(args.config)
        self.is_regression = is_regression
        self.D = None
        self.args.device = get_device()
        self.trlog = {}
        assert(args.cat_policy == 'indices')

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        
        model_config = None
        if config is not None:
            self.reset_stats_withconfig(config)
            model_config = config['model']
        
        if model_config is None:
            model_config = self.args.config['model']
        self.data_format(is_train = True)
        from catboost import CatBoostClassifier, CatBoostRegressor
        
        cat_features = list(range(self.n_num_features, self.n_num_features + self.n_cat_features))
        if self.C is None:
            X_train,X_val = self.N['train'],self.N['val']
        elif self.N is None:
            X_train,X_val = self.C['train'],self.C['val']
        else:
            X_train = np.concatenate([self.N['train'], self.C['train'].astype(str)], axis=1)
            X_val = np.concatenate([self.N['val'], self.C['val'].astype(str)], axis=1) 
        self.model = CatBoostRegressor(**model_config, random_state=self.args.seed, cat_features=cat_features, allow_writing_files=False) if self.is_regression else CatBoostClassifier(**model_config, random_state=self.args.seed, cat_features=cat_features, allow_writing_files=False)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        fit_config = deepcopy(self.args.config['fit'])
        fit_config.pop('n_bins')
        fit_config['eval_set'] = (X_val, self.y['val'])
        tic = time.time()
        self.model.fit(X_train, self.y['train'],**fit_config)
        if not self.is_regression:
            y_pred_val = self.model.predict(X_val)
            self.trlog['best_res'] = accuracy_score(self.y['val'], y_pred_val) 
        else:
            y_pred_val = self.model.predict(X_val)
            self.trlog['best_res'] = mean_squared_error(self.y['val'], y_pred_val, squared=False)*self.y_info['std']
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
        if self.C_test is None:
            test_data = self.N_test
        elif self.N_test is None:
            test_data = self.C_test.astype(str)
        else:
            test_data = np.concatenate([self.N_test, self.C_test.astype(str)], axis=1)
        if self.is_regression:
            test_logit = self.model.predict(test_data)
        else:
            test_logit = self.model.predict_proba(test_data)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit