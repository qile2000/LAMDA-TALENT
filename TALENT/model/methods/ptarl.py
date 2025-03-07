from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch
import os.path as osp
from TALENT.model.lib.ptarl.utils import (
    fit_Ptarl,
    test,
    generate_topic
)

from TALENT.model.lib.data import (
    Dataset
)
import time

class PTARLMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        from TALENT.model.models.ptarl import PTARL
        if model_config is None:
            model_config = self.args.config['model']
        self.model = PTARL(
            input_num=self.d_in,
            out_dim=self.d_out,
            categories=self.categories,
            cluster_centers_=self.cluster_centers_,
            model_type=self.model_type1,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()
        self.model_config = model_config

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        
        
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        cluster_centers_ = np.zeros([self.args.config['model']['n_clusters'], 1])
        self.cluster_centers_ = cluster_centers_
        self.model_type1 = self.args.model_type
        self.construct_model()
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        tic = time.time()
        best_model,_ = fit_Ptarl(self.args,self.model, self.train_loader, self.val_loader, self.criterion, 
                                 self.args.model_type, self.args.config, self.args.config['model']['regularize'], 
                                 self.is_regression, self.args.config['general']['ot_weight'], self.args.config['general']['diversity_weight'], 
                                 self.args.config['general']['r_weight'], self.args.config['general']['diversity'],self.args.seed, self.args.save_path)
        cluster_centers_ = generate_topic(best_model, self.train_loader,self.args.config['model']['n_clusters'])
        self.cluster_centers_ = cluster_centers_
        np.save(osp.join(self.args.save_path, 'cluster-centers-{}.npy'.format(str(self.args.seed))), cluster_centers_)
        self.model_type1 = self.args.model_type + '_ot'
        self.construct_model(self.model_config)
        best_model,best_loss = fit_Ptarl(self.args,self.model, self.train_loader, self.val_loader, self.criterion, 
                                         self.args.model_type+'_ot', self.args.config, self.args.config['model']['regularize'], 
                                         self.is_regression, self.args.config['general']['ot_weight'], self.args.config['general']['diversity_weight'], 
                                         self.args.config['general']['r_weight'], self.args.config['general']['diversity'],self.args.seed, self.args.save_path)
        time_cost = time.time() - tic
        self.model = best_model
        self.trlog['best_res'] = best_loss
        return time_cost
        
    def predict(self, data, info, model_name):
        N,C,y = data
        self.model_type1 = self.args.model_type + '_ot'
        self.cluster_centers_ = np.load(osp.join(self.args.save_path, 'cluster-centers-{}.npy'.format(str(self.args.seed))), allow_pickle=True)
        self.construct_model(self.model_config)
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        self.model.eval()
        self.data_format(False, N, C, y)

        test_logit,test_label = test(self.model, self.test_loader,self.args)
        
        vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        
        return vl, vres, metric_name, test_logit
