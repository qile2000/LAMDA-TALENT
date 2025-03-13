import abc
import torch
import numpy as np
import time
import os.path as osp
from tqdm import tqdm
import sklearn.metrics as skm
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from TALENT.model.utils import (
    Timer,
    Averager,
    set_seeds,
    get_device
)

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
    data_loader_process,
    get_categories
)
from TALENT.model.methods.base import Method

def check_softmax(logits):
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits
  
class ExcelFormerMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        from TALENT.model.models.excelformer import ExcelFormer
        if model_config is None:
            model_config = self.args.config['model']
        self.model = ExcelFormer(
            d_numerical=self.d_in,
            d_out=self.d_out,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def fit(self, data, info, train = True, config = None):
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        N,C,y = data
        self.D = Dataset(N, C, y, info)
        mi_func = mutual_info_regression if self.D.is_regression else mutual_info_classif

        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        mi_scores = mi_func(self.N['train'].cpu().numpy(), self.y['train'].cpu().numpy())
        mi_ranks = np.argsort(-mi_scores)
        self.sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).to(torch.float64).to(self.args.device)
        
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        self.mix_type = self.args.config['training']['mix_type']
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


    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        for i, (X, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            if self.N is not None and self.C is not None:
                X_num, X_cat = X[0], X[1]
            elif self.C is not None and self.N is None:
                X_num, X_cat = None, X
            else:
                X_num, X_cat = X, None
            if self.mix_type == 'none':
                loss = self.criterion(self.model(X_num, X_cat,mix_up=False), y)
            else:
                preds, feat_masks, shuffled_ids = self.model(X_num, X_cat,mix_up=True)
                if self.mix_type == 'feat_mix':
                    lambdas = (self.sorted_mi_scores * feat_masks).sum(1) # bs
                    lambdas2 = 1 - lambdas
                elif self.mix_type == 'hidden_mix':
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                elif self.mix_type == 'niave_mix':
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                if self.is_regression:
                    mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                    loss = self.criterion(preds, mix_y)
                else:
                    loss = lambdas * self.criterion(preds, y) + lambdas2 * self.criterion(preds, y[shuffled_ids])
                    loss = loss.mean()
            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)  