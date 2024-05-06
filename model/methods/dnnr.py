from model.methods.base import Method
import torch
import numpy as np
import torch.nn.functional as F

from model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_norm_process,
    data_label_process
)


class DNNRMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')
        assert(is_regression)
        

    def construct_model(self, model_config = None):
        from model.models.dnnr import DNNR
        if model_config is None:
            model_config = self.args.config['model']
        self.model = DNNR(
            **model_config
        )


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)

            self.d_out = 1
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
            self.criterion = F.mse_loss
        else:
            self.N_test, self.C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            self.y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            self.N_test, self.C_test, _, _, _ = data_enc_process(self.N_test, self.C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            self.N_test, _ = data_norm_process(self.N_test, self.args.normalization, self.args.seed, self.normalizer)



    def fit(self, N, C, y, info, train = True, config = None):
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
            
            self.data_format(is_train = True)
            if config:
                self.reset_stats_withconfig(config)
            self.construct_model()
        
        assert(self.C is None and self.N is not None)

        self.model.fit(np.array(self.N['train']), np.array(self.y['train']))
        
        test_logit = self.model.predict(np.array(self.N['val']))
        test_label = self.y['val']
        
        test_logit = torch.from_numpy(test_logit)
        test_label = torch.from_numpy(test_label)
        
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        self.trlog['best_res'] = vres[0]


    def predict(self, N, C, y, info, model_name):
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        
        self.data_format(False, N, C, y)
        
        assert(self.C_test is None and self.N_test is not None)

        test_logit = self.model.predict(np.array(self.N_test['test']))
        test_label = self.y_test['test']
    
        test_logit = torch.from_numpy(test_logit)
        test_label = torch.from_numpy(test_label)
        
        vl = self.criterion(test_logit, test_label).item()

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))

        return vl, vres, metric_name, test_logit