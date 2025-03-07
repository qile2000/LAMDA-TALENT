from TALENT.model.methods.base import Method
import time
import torch
import torch.nn.functional as F
import os.path as osp
from tqdm import tqdm
import numpy as np
import sklearn.metrics as skm
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

class ProtoGateMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        self.device = args.device
        assert(args.cat_policy != 'indices')
        assert(is_regression == False)

    def construct_model(self, model_config = None):
        from TALENT.model.models.protogate import GatingNet
        if model_config is None:
            model_config = self.args.config['model']
        self.model = GatingNet(
            input_dim = self.d_in,
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
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.X_train = self.N['train'] 
        self.y_neighbour = torch.tensor(self.y['train'])
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        self.feature_selection = self.args.config['model']['feature_selection'] if 'feature_selection' in self.args.config['training'].items() else True
        self.pred_k = self.args.config['training']['pred_k']
        self.lam = self.args.config['training']['lam']
        self.sigma = self.args.config['model']['sigma']
        self.a = self.args.config['model']['a']
        self.l1_coef = self.args.config['training']['l1_coef']
        self.pred_coef = self.args.config['training']['pred_coef']
        from TALENT.model.models.protogate import KNNNet
        self.proto_layer = KNNNet(k=self.pred_k, tau=self.args.config['training']['sorting_tau'])

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
                        
                x_selected, self.alpha, self.stochastic_gate = self.model(X_num)
                x_neighbour_selected, _, _ = self.model(torch.tensor(self.X_train).to(x_selected.device))
                y_neighbour = self.y_neighbour.to(x_selected.device)
                y_neighbour = F.one_hot(y_neighbour, num_classes=self.d_out)

                pred, neighbour_list = proto_predict(x_selected, x_neighbour_selected, y_neighbour, self.pred_k)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = 1.0   

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))

        
        return vl, vres, metric_name, test_logit

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
            x_selected, self.alpha, self.stochastic_gate = self.model(X_num)
            x_neighbour_selected = x_selected
            y_neighbour = y
            y_true = F.one_hot(y, num_classes=self.d_out)
            y_neighbour = F.one_hot(y_neighbour, num_classes=self.d_out)

            y_pred, neighbour_list = proto_predict(x_selected, x_neighbour_selected, y_neighbour, self.pred_k)
            loss = self.compute_pred_loss(x_selected, x_neighbour_selected, y_true, y_neighbour)['total']

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
                x_selected, self.alpha, self.stochastic_gate = self.model(X_num)
                x_neighbour_selected, _, _ = self.model(torch.tensor(self.X_train).to(x_selected.device))
                y_neighbour = self.y_neighbour.to(x_selected.device)
                y_neighbour = F.one_hot(y_neighbour, num_classes=self.d_out)

                pred, neighbour_list = proto_predict(x_selected, x_neighbour_selected, y_neighbour, self.pred_k)
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        # vl = self.criterion(test_logit, test_label).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)


        print('epoch {}, val, {} result={:.4f}'.format(epoch,  task_type, vres[0]))
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
        elif self.is_binclass:
            accuracy = skm.accuracy_score(labels, predictions)
            avg_recall = skm.balanced_accuracy_score(labels, predictions)
            avg_precision = skm.precision_score(labels, predictions, average='macro')
            f1_score = skm.f1_score(labels, predictions, average='binary')
            return (accuracy, avg_recall, avg_precision, f1_score), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1")
        elif self.is_multiclass:
            accuracy = skm.accuracy_score(labels, predictions)
            avg_recall = skm.balanced_accuracy_score(labels, predictions)
            avg_precision = skm.precision_score(labels, predictions, average='macro')
            f1_score = skm.f1_score(labels, predictions, average='macro')
            return (accuracy, avg_recall, avg_precision, f1_score), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1")
        
        
    def compute_pred_loss(self, x_query, x_cand, y_query, y_neighbor):
        losses = {}
        losses['l0_norm'] = torch.zeros(1, device=self.device)
        losses['l1_norm'] = torch.zeros(1, device=self.device)
        losses['pred_loss'] = torch.zeros(1, device=self.device)

        if self.feature_selection:
            losses['l0_norm'] = self.compute_sparsity_loss(self.alpha)
            losses['l1_norm'] = self.l1_coef * torch.norm(self.model.embed.fn0.weight, p=1)

        losses['pred_loss'] = self.pred_k + \
                              pred_loss(self.proto_layer, x_query, x_cand, y_query, y_neighbor)
        losses['pred_loss'] = self.pred_coef * losses['pred_loss']

        losses['total'] = losses['pred_loss'] + losses['l1_norm'] + losses['l0_norm']

        return losses

    def compute_sparsity_loss(self, input2cdf):
        # gates regularization
        reg = 0.5 - 0.5 * torch.erf((-input2cdf) / (self.sigma * np.sqrt(2)))
        loss_reg_gates = self.lam * torch.mean(torch.sum(reg, dim=1))

        return loss_reg_gates

def pred_loss(proto_layer, query, neighbors, query_label, neighbor_labels, method='deterministic'):
    # query: batch_size x p
    # neighbors: 10k x p
    # query_labels: batch_size x [10] one-hot
    # neighbor_labels: n x [10] one-hot
    if method == 'deterministic':
        # top_k_ness is the sum of top-k row of permutation matrix
        top_k_ness = proto_layer(query, neighbors)  # (B*512, N*512) => (B, N)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)  # (B, N)
        correct_in_top_k = (correct * top_k_ness).sum(-1)  # [B]
        loss = -correct_in_top_k
        # loss = 1 / correct_in_top_k
        return loss.mean()
    elif method == 'stochastic':
        top_k_ness = proto_layer(query, neighbors)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)
        correct_in_top_k = (correct.unsqueeze(0) * top_k_ness).sum(-1)
        loss = -correct_in_top_k
        return loss.mean()
    else:
        raise ValueError(method)

def proto_predict(query, neighbors, neighbor_labels, k):
    '''
    query: p
    neighbors: n x p
    neighbor_labels: n x num_classes
    '''
    query, neighbors = query.detach(), neighbors.detach()
    diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
    # squared_diffs = diffs**2
    # norms = squared_diffs.sum(-1)
    norms = torch.norm(diffs, p=2, dim=-1)
    indices = torch.argsort(norms, dim=-1).to(neighbor_labels.device)
    labels = neighbor_labels[indices[:, :k]]  # n x k x num_classes
    label_counts = labels.sum(dim=1)  # n x num_classes
    prediction = torch.argmax(label_counts, dim=1)  # n

    return prediction, indices[:, :k]