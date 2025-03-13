from TALENT.model.methods.base import Method
import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch

from TALENT.model.lib.data import (
    Dataset
)
import time


class GrowNetMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')
        if is_regression:
            self.loss_f1 = torch.nn.MSELoss()
            self.loss_f2 = torch.nn.MSELoss()
        else:
            self.loss_f1 = torch.nn.MSELoss(reduction='none')
            self.loss_f2 = torch.nn.CrossEntropyLoss()
        self.sub_model_config = None

    def construct_model(self, model_config = None):
        if model_config == None:
            model_config = self.args.config['model']
        self.sub_model_config = model_config
        self.ensemble_model_config = self.args.config['ensemble_model']
        from TALENT.model.models.grownet import DynamicNet
        self.model = DynamicNet(
            categories = self.categories,
            **self.ensemble_model_config
        )
        self.model.to_cuda()

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()
        
        if self.sub_model_config == None:
            self.sub_model_config = self.args.config['model']
        if self.C:
            self.sub_model_config['feat_d'] = self.d_in + self.C['train'].shape[1] * self.ensemble_model_config['d_embedding']
        else:
            self.sub_model_config['feat_d'] = self.d_in

        self.sub_model_config['dim_out'] = self.d_out
        
        if not train:
            return

        from TALENT.model.models.grownet import MLP_2HL
        training_config = self.args.config['training']
        learning_rate = training_config['lr']
        weight_decay = training_config['weight_decay']
        tic = time.time()
        for s in range(self.args.max_epoch):
            m = MLP_2HL.get_model(s,argparse.Namespace(**self.sub_model_config)).to(self.args.device)
            
            self.optimizer = torch.optim.AdamW(
                m.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            self.model.to_train()
            self.model.to_cuda()
            if not self.args.use_float:
                self.model.to_double()
            for epoch in range(training_config['epochs_per_stage']):
                for i, (X, y) in enumerate(self.train_loader, 1):
                    if self.N is not None and self.C is not None:
                        X_num, X_cat = X[0], X[1]
                    elif self.C is not None and self.N is None:
                        X_num, X_cat = None, X
                    else:
                        X_num, X_cat = X, None
                    middle_feat, out = self.model.forward(X_num, X_cat)
                    out = torch.as_tensor(out, dtype=torch.float64).cuda().view(-1, 1)
                    if self.is_regression:
                        grad_direction = -(out - y)
                        _, out = m(self.model.embed_input(X_num, X_cat), middle_feat)
                        out = torch.as_tensor(out, dtype=torch.float64).cuda().view(-1, 1)
                        loss = self.loss_f1(self.model.boost_rate * out, grad_direction)
                    else:
                        h = 1 / ((1 + torch.exp(y * out)) * (1 + torch.exp(-y * out)))
                        grad_direction = y * (1.0 + torch.exp(-y * out))
                        out = torch.as_tensor(out)
                        _, out = m(self.model.embed_input(X_num, X_cat), middle_feat)
                        out = torch.as_tensor(out, dtype=torch.float64).cuda().view(-1, 1)
                        loss = self.loss_f1(self.model.boost_rate * out, grad_direction) # T
                        loss = loss * h
                        loss = loss.mean()
                    m.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            self.model.add(m)
            if s > 0:
                lr_scaler = training_config['lr_scaler']
                if s % 30 == 0:
                    learning_rate /= 2
                    weight_decay /= 2
                self.optimizer = torch.optim.AdamW(
                    m.parameters(), 
                    lr=learning_rate/ lr_scaler, 
                    weight_decay=weight_decay
                )
                for e in range(training_config["correct_epoch"]):
                    for i, (X, y) in enumerate(self.train_loader, 1):
                        if self.N is not None and self.C is not None:
                            X_num, X_cat = X[0], X[1]
                        elif self.C is not None and self.N is None:
                            X_num, X_cat = None, X
                        else:
                            X_num, X_cat = X, None
                        _, out = self.model.forward_grad(X_num, X_cat)
                        out = torch.as_tensor(out, dtype=torch.float64).cuda()
                        loss = self.loss_f2(out, y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    print(f"Corrected epoch {e} done with loss {loss.item()} for stage {s} with lr {self.optimizer.param_groups[0]['lr']}")
            self.validate(s)
            if not self.continue_training:
                break
        
        time_cost = time.time() - tic
        return time_cost

    
    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.to_eval()
        self.model.to_cuda()
        if not self.args.use_float:
                self.model.to_double()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                _,pred = self.model.forward(X_num,X_cat)
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
            self.model.to_file(self.args.save_path + "/final-{}.pt".format(str(self.args.seed)))
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False


    def predict(self, data, info, model_name):
        N,C,y = data
        from TALENT.model.models.grownet import DynamicNet,MLP_2HL
        self.model = DynamicNet.from_file(
            self.args.save_path + "/final-{}.pt".format(str(self.args.seed)),
            lambda stage: MLP_2HL.get_model(stage, argparse.Namespace(**self.sub_model_config)),
        )
        self.model.to_eval()
        self.model.to_cuda()
        if not self.args.use_float:
                self.model.to_double()
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
                        
                _,pred = self.model.forward(X_num,X_cat)

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
