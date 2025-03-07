from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process
)
import time

class HyperFastMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(is_regression == False)
        assert(args.normalization == 'none')
        assert(args.cat_policy == 'indices')
        assert(args.num_policy == 'none')
        assert(args.tune != True)


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.criterion = F.cross_entropy
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']

    def construct_model(self, model_config = None):
        from TALENT.model.models.hyperfast import HyperFastClassifier
        self.model = HyperFastClassifier(device = self.args.device, seed = self.args.seed)  

    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        self.construct_model()

        sampled_Y = self.y['train']
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.C['train'], self.N['train']),axis=1)
            cat_features = list(range(self.C['train'].shape[1]))
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_features = list(range(self.C['train'].shape[1]))
        else:
            sampled_X = self.N['train']
            cat_features = []
        tic = time.time()
        self.model.fit(sampled_X, sampled_Y, cat_features)
        time_cost = time.time() - tic
        return time_cost
    
    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test,self.C_test),axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        test_logit = self.model.predict_proba(Test_X)
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit, dtype=torch.double),torch.tensor(test_label, dtype=torch.long)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit

