from TALENT.model.methods.base import Method
import torch

from TALENT.model.utils import (
    Averager
)

class TangosMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        from TALENT.model.models.tangos import Tangos
        if model_config is None:
            model_config = self.args.config['model']
        self.model = Tangos(
                d_in=self.d_in,
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
            # categorical features are encoded to X_num
            assert(X_num is not None)
            loss1 = self.criterion(self.model(X_num, X_cat), y)
            spec_loss,orth_loss = self.model.cal_tangos_loss(X_num)
            loss = loss1 + self.model.lambda1 * spec_loss + self.model.lambda2 * orth_loss
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


