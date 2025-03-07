from TALENT.model.methods.base import Method
import time
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from TALENT.model.utils import (
    Averager
)
from typing import Optional
from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_norm_process,
    data_label_process,
    data_loader_process
)


def make_random_batches(
    train_size: int, batch_size: int, device: Optional[torch.device] = None
) :
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]


class TabRMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'tabr_ohe')
        assert(args.num_policy == 'none')


    def construct_model(self, model_config = None):
        from TALENT.model.models.tabr import TabR
        if model_config is None:
            model_config = self.args.config['model']
        self.model = TabR(
            n_num_features = self.n_num_features,
            n_cat_features = self.C_features,
            n_classes = self.d_out,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()
    
    
    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.n_num_features = self.N['train'].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C['train'].shape[1] if self.C is not None else 0
                    
            
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.C_features = self.C['train'].shape[1] if self.C is not None else 0
            self.N, self.C, self.y, self.train_loader, self.val_loader, self.criterion = data_loader_process(self.is_regression, (self.N, self.C), self.y, self.y_info, self.args.device, self.args.batch_size, is_train = True,is_float=self.args.use_float)
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            _, _, _, self.test_loader, _ =  data_loader_process(self.is_regression, (N_test, C_test), y_test, self.y_info, self.args.device, self.args.batch_size, is_train = False,is_float=self.args.use_float)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
            
            self.data_format(is_train = True)
        if config is not None:
            self.reset_stats_withconfig(config)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        self.train_size = self.N['train'].shape[0] if self.N is not None else self.C['train'].shape[0]
        self.train_indices = torch.arange(self.train_size, device=self.args.device)
        self.context_size = 96
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        
        time_cost = 0
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost


    def predict(self, data, info, model_name):
        N,C,y = data
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()
        
        self.data_format(False, N, C, y)
        
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None  
                candidate_x_num = self.N['train'] if self.N is not None else None
                candidate_x_cat = self.C['train'] if self.C is not None else None
                candidate_y = self.y['train']
                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None
                    candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
                    candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
                    if self.is_regression:
                        candidate_y = candidate_y.float()
                pred = self.model(
                    x_num = X_num,
                    x_cat = X_cat,
                    y = None,
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        
        return vl, vres, metric_name, test_logit


    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        i = 0
        for batch_idx in make_random_batches(self.train_size, self.args.batch_size, self.args.device):
            self.train_step = self.train_step + 1
            
            X_num = self.N['train'][batch_idx] if self.N is not None else None
            X_cat = self.C['train'][batch_idx] if self.C is not None else None
            y = self.y['train'][batch_idx]

            candidate_indices = self.train_indices
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]

            candidate_x_num = self.N['train'][candidate_indices] if self.N is not None else None
            candidate_x_cat = self.C['train'][candidate_indices] if self.C is not None else None
            candidate_y = self.y['train'][candidate_indices]
            if self.args.use_float:
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
                candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
                candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
                if self.is_regression:
                    candidate_y = candidate_y.float()
                    y = y.float()
            pred = self.model(
                x_num = X_num,
                x_cat = X_cat,
                y=y,
                candidate_x_num=candidate_x_num,
                candidate_x_cat=candidate_x_cat,
                candidate_y=candidate_y,
                context_size=self.context_size,
                is_train=True,
            ).squeeze(-1)
            
            loss = self.criterion(pred, y)
            
            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
            i += 1

        tl = tl.item()
        self.trlog['train_loss'].append(tl)    


    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            
                
                candidate_x_num = self.N['train'] if self.N is not None else None
                candidate_x_cat = self.C['train'] if self.C is not None else None
                candidate_y = self.y['train']
                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None
                    candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
                    candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
                    if self.is_regression:
                        candidate_y = candidate_y.float()
                pred = self.model(
                    x_num = X_num,
                    x_cat = X_cat,
                    y = None,
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()          

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))