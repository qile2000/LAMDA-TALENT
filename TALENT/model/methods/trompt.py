import abc
import torch
import numpy as np
import time
import os.path as osp
from tqdm import tqdm
import sklearn.metrics as skm

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
    
class TromptMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')
    
    def construct_model(self, model_config = None):
        from TALENT.model.models.trompt import Trompt
        if model_config is None:
            model_config = self.args.config['model']
        self.model = Trompt(
                n_num_features=self.d_in,
                cat_cardinalities=self.categories,
                d_out=self.d_out,
                **model_config
                ).to(self.args.device) 
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

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
            # print(self.model(X_num, X_cat).shape)
            output = self.model.forward_for_training(X_num, X_cat)
            # print(output.shape)

            n_cycles = output.shape[1]
            output = output.view(-1, self.d_out)

            # print(output.shape)
            y = y.repeat_interleave(n_cycles)

            # print(output.shape, y.shape)
            loss = self.criterion(output, y)

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