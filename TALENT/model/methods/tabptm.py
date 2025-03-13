from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from TALENT.model.lib.tabptm.utils_tabptm import *
from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process,
    data_norm_process,
)
from sklearn.feature_selection import  mutual_info_classif, mutual_info_regression
from TALENT.model.utils import Averager
from tqdm import tqdm
import os.path as osp

class TabPTMMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.normalization == 'standard')
        assert(args.cat_policy == 'ohe')
        assert(args.tune != True)
        assert(args.num_policy == 'none')
        self.distance = 'maneucbra'
        self.args.centers_num = 10000

        if is_regression:
            self.numK = 16
        else:
            self.numK = 32


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)

            if self.is_regression:
                self.criterion = F.mse_loss
                self.d_out = 1
                self.centers = prepare_meta_feature_regression(self.N, self.y, self.args)
                mi = mutual_info_regression(self.N['train'], self.y['train'])
                mi = mi / mi.sum()
                self.mi = torch.from_numpy(mi).double().to(self.args.device)
            else:
                self.criterion = F.cross_entropy
                self.d_out = len(np.unique(self.y['train']))
                self.centers = prepare_meta_feature(self.N, self.y, self.args)
                mi = mutual_info_classif(self.N['train'], self.y['train'])
                mi = mi / mi.sum()
                self.mi = torch.from_numpy(mi).double().to(self.args.device)
            print("mutual information weights: ", self.mi)

            X = {k: v.to(self.args.device) for k, v in to_tensors(self.N).items()}
            Y = {k: v.to(self.args.device) for k, v in to_tensors(self.y).items()}
            self.N = {k: v.double() for k, v in X.items()}
            if self.is_regression:
                self.y = {k: v.double() for k, v in Y.items()}
            else:
                self.y = {k: v.long() for k, v in Y.items()}
            if torch.cuda.is_available():
                self.centers = [torch.from_numpy(c.astype(np.float64)).cuda() for c in self.centers]    
            else:
                self.centers = [torch.from_numpy(c.astype(np.float64)) for c in self.centers]
            trainset = TabPTMData(self.D, self.N, self.y, self.y_info, 'train')
            valset = TabPTMData(self.D, self.N, self.y, self.y_info, 'val')
            self.train_loader = DataLoader(dataset=trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)        
            self.val_loader = DataLoader(dataset=valset, batch_size=2048, shuffle=False, num_workers=0) 
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)


            X = {k: v.to(self.args.device) for k, v in to_tensors(N_test).items()}
            Y = {k: v.to(self.args.device) for k, v in to_tensors(y_test).items()}
            self.N_test = {k: v.double() for k, v in X.items()}
            if self.is_regression:
                self.y_test = {k: v.double() for k, v in Y.items()}
            else:
                self.y_test = {k: v.long() for k, v in Y.items()}
            testset = TabPTMData(self.D, self.N_test, self.y_test, self.y_info, 'test')
            self.test_loader = DataLoader(dataset=testset, batch_size=2048, shuffle=False, num_workers=0)

    def construct_model(self, model_config = None):
        from TALENT.model.models.tabptm import TabPTM
        if model_config is None:
            model_config = self.args.config['model']
        self.model = TabPTM(
            distance=self.distance,
            is_regression=self.is_regression,
            d_in=self.numK,
            d_out=self.d_out,
            **model_config
        ).to(self.args.device)
        if self.is_regression:
            self.model.load_state_dict(torch.load('model/models/models_tabptm/metaregC-numK16-Reweight-LR0.001-maneucbra-log.pth')['params'])  
        else:
            self.model.load_state_dict(torch.load('model/models/models_tabptm/metaclsA-numK32-Reweight-LR0.001-maneucbra-log.pth')['params'])
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def fit(self, data, info, train = True, config = None):
        N,C,y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
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
                if self.is_regression:
                    X_meta = self.get_meta_feature_regression(X, y)

                else:
                    X_meta = self.get_meta_feature_allclass(X, y)
                pred = self.model(X_meta)                
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


    def bray_curtis_dist(self, X, centers_c, weights = None):
        if weights != None:
            weighted_numerator = torch.sum(weights[None, None, :] * torch.abs(X[:, None, :] - centers_c[None, :, :]), dim=-1)
            weighted_denominator = torch.sum(weights[None, None, :] * torch.abs(X[:, None, :] + centers_c[None, :, :]), dim=-1)
            return weighted_numerator / weighted_denominator
        else:
            numerator = torch.sum(torch.abs(X[:, None, :] - centers_c[None, :, :]), dim=-1)
            denominator = torch.sum(torch.abs(X[:, None, :] + centers_c[None, :, :]), dim=-1)
            return numerator / denominator

    def canberra_dist(self, X, centers_c, weights = None):
        if weights != None:
            weighted_numerator = weights[None, None, :] *  torch.abs(X[:, None, :] - centers_c[None, :, :])
            weighted_denominator = weights[None, None, :] * (torch.abs(X[:, None, :]) + torch.abs(centers_c[None, :, :])) + 1e-10
            return torch.sum(weighted_numerator / weighted_denominator, dim=-1)
        else:
            numerator = torch.abs(X[:, None, :] - centers_c[None, :, :])
            denominator = torch.abs(X[:, None, :]) + torch.abs(centers_c[None, :, :])
            return torch.sum(numerator / denominator, dim=-1)
    
    def get_meta_feature_regression(self, X, y=None, is_training=False):
        distance_list = [self.distance[i:i+3] for i in range(0, len(self.distance), 3)]
        distance_pos_list = []
        label_pos_list = []
        # get pairwise distance matrix
        numK = self.numK
        # convert a list of tensor to tensor
        if isinstance(self.centers, list):
            self.centers = torch.stack(self.centers)
        for distance_type in distance_list:
            centers_c = self.centers[:,:-1]
            labels_c = self.centers[:,-1]

            if distance_type == 'euc':
                distance_pos_c = torch.cdist(X * torch.sqrt(self.mi), centers_c * torch.sqrt(self.mi), p=2.0)
            elif distance_type == 'man':
                distance_pos_c = torch.cdist(X * self.mi, centers_c * self.mi, p=1.0)
            elif distance_type == 'cos':
                distance_pos_c = 1.0 - torch.mm(F.normalize(X, dim=-1), F.normalize(centers_c, dim=-1).t())
            elif distance_type == 'che':
                distance_pos_c = torch.cdist(X, centers_c, p=float('inf'))
            elif distance_type == 'bra':
                distance_pos_c = self.bray_curtis_dist(X, centers_c, self.mi)
            elif distance_type == 'can':
                distance_pos_c = self.canberra_dist(X, centers_c, self.mi)

            if is_training:           
                topk_pos = torch.topk(distance_pos_c, min(numK+1, centers_c.shape[0]), largest=False)[0]
                topk_pos_labels = labels_c[torch.topk(distance_pos_c, min(numK + 1, centers_c.shape[0]), largest=False)[1]]
                zero_mask = (topk_pos[:, 0] < 1e-6)
                if zero_mask.any():
                    topk_pos = torch.where(zero_mask[:, None].expand_as(topk_pos[:, 1:]), topk_pos[:, 1:1+numK], topk_pos[:, :numK])
                    topk_pos_labels = torch.where(zero_mask[:, None].expand_as(topk_pos_labels[:, 1:]), topk_pos_labels[:, 1:1+numK], topk_pos_labels[:, :numK])
                else:
                    topk_pos = topk_pos[:, :numK]
                    topk_pos_labels = topk_pos_labels[:, :numK]
            else:
                topk_pos = torch.topk(distance_pos_c, min(numK, centers_c.shape[0]), largest=False)[0]    
                topk_pos_labels = labels_c[torch.topk(distance_pos_c, min(numK, centers_c.shape[0]), largest=False)[1]]

            if topk_pos.shape[1] < numK:
                num_diff = numK - topk_pos.shape[1]
                topk_pos = torch.cat([topk_pos, torch.tile(torch.max(topk_pos, axis=1, keepdims=True)[0], [1, num_diff])], axis=-1)
                topk_pos_labels = torch.cat([topk_pos_labels, torch.tile(topk_pos_labels[:,-1:], [1, num_diff])], axis=-1)

            distance_pos_list.append(topk_pos)
            label_pos_list.append(topk_pos_labels)

        # distance_num x batch x (numK * 2)
        distance_and_label = torch.cat([torch.stack(distance_pos_list), torch.stack(label_pos_list)], dim=2)
        X_meta = distance_and_label.permute([1, 0, 2]).contiguous()   

        # num_inst x 1 x len(distance_list) x (numK * 2) 
        X_meta = torch.unsqueeze(X_meta, 1)

        return X_meta

    def get_meta_feature_allclass(self, X, y=None, is_training=False):
        # get pairwise distance matrix
        distance_list = [self.distance[i:i+3] for i in range(0, len(self.distance), 3)]
        numK = self.numK
        distance_pos_list = []

        for distance_type in distance_list:
            distance_pos_c_list = []
            for c_index, centers_c in enumerate(self.centers):
                if distance_type == 'euc':
                    distance_pos_c = torch.cdist(X * torch.sqrt(self.mi), centers_c * torch.sqrt(self.mi), p=2.0)
                elif distance_type == 'man':
                    distance_pos_c = torch.cdist(X * self.mi, centers_c * self.mi, p=1.0)
                elif distance_type == 'cos':
                    distance_pos_c = 1.0 - torch.mm(F.normalize(X, dim=-1), F.normalize(centers_c, dim=-1).t())
                elif distance_type == 'che':
                    distance_pos_c = torch.cdist(X, centers_c, p=float('inf'))
                elif distance_type == 'bra':
                    distance_pos_c = self.bray_curtis_dist(X, centers_c, self.mi)
                elif distance_type == 'can':
                    distance_pos_c = self.canberra_dist(X, centers_c, self.mi)

                if is_training:           
                    topk_pos_c = torch.topk(distance_pos_c, min(numK+1, centers_c.shape[0]), largest=False)[0]
                    nn_mask_pos = torch.zeros_like(topk_pos_c).masked_fill(torch.unsqueeze(y == c_index, -1), 1)
                    nn_mask_neg = torch.zeros_like(topk_pos_c).masked_fill(torch.unsqueeze(y != c_index, -1), 1) 
             
                    zero_mask = (topk_pos_c[:, 0] < 1e-6)
                    if zero_mask.any():
                        topk_pos_c = (topk_pos_c * nn_mask_pos)[:, 1:] + (topk_pos_c * nn_mask_neg)[:, :min(numK, centers_c.shape[0]-1)]
                    else:
                        topk_pos_c = (topk_pos_c * nn_mask_pos)[:, :min(numK, centers_c.shape[0]-1)] + (topk_pos_c * nn_mask_neg)[:, :min(numK, centers_c.shape[0]-1)]
                else:
                    topk_pos_c = torch.topk(distance_pos_c, min(numK, centers_c.shape[0]), largest=False)[0]

                if topk_pos_c.shape[1] < numK:
                    num_diff = numK - topk_pos_c.shape[1]
                    topk_pos_c = torch.cat([topk_pos_c, torch.tile(torch.max(topk_pos_c, axis=1, keepdims=True)[0], [1, num_diff])], axis=-1)

                distance_pos_c_list.append(topk_pos_c)

            distance_pos_list.append(torch.stack(distance_pos_c_list))

        X_meta = torch.stack(distance_pos_list).permute([2, 1, 0, 3]).contiguous()  # num_inst x num_class x len(distance_list) x numK       

        batch_size, num_class, num_dist, numK = X_meta.size()
        X_meta_dist_list = []
        label_dist_list = []
        for i in range(num_dist):
            X_meta_dist = X_meta[:,:,i,:]

            num_per_subset = numK // num_class
            remainder = numK % num_class
            X_meta_new = torch.zeros(batch_size, numK).cuda()
            label_dist = torch.zeros(batch_size, numK).cuda()
            for i in range(num_class):
                subset = X_meta_dist[:, i, :].topk(num_per_subset, dim=1, largest=False).values
                start_idx = i * num_per_subset
                end_idx = start_idx + num_per_subset
                X_meta_new[:, start_idx:end_idx] = subset
                label_dist[:, start_idx:end_idx] = i 

            if remainder > 0:
                non_zero_values, non_zero_indices = torch.nonzero(X_meta_new).unbind(1)
                _, min_indices = non_zero_values.topk(remainder, largest=False)
                
                selected_indices = non_zero_indices[min_indices]
                selected_values = X_meta_new[:, selected_indices]
                X_meta_new[:, -remainder:] = selected_values

                selected_labels = label_dist[:, selected_indices]
                label_dist[:, -remainder:] = selected_labels
            
            X_meta_new, sorted_indices = X_meta_new.sort(dim=1)
            label_dist = label_dist.gather(1, sorted_indices)
            X_meta_dist_list.append(X_meta_new)
            label_dist_list.append(label_dist)

        X_meta = torch.stack(X_meta_dist_list).permute([1, 0, 2]).unsqueeze(1).expand(-1, num_class, -1, -1)
        label = torch.stack(label_dist_list).permute([1, 0, 2]).unsqueeze(1).expand(-1, num_class, -1, -1)
        mask_tensor = torch.ones_like(label).cuda() * -1.0

        mask_tensor[label == torch.arange(num_class).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)] = 1.0
        X_meta = torch.cat((X_meta,mask_tensor),dim=-1)
        return X_meta.double()

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        for i, (X, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            # get meta-feature
            if self.is_regression:
                X_meta = self.get_meta_feature_regression(X, y, is_training=True)
                pred = self.model(X_meta)
                loss = self.criterion(pred, y.to(torch.float64).squeeze())
            else:
                X_meta = self.get_meta_feature_allclass(X, y, is_training=True)
                pred = self.model(X_meta, X)
                loss = self.criterion(pred, y)

            # X: batch x num_class x 3 x numK
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
                if self.is_regression:
                    X_meta = self.get_meta_feature_regression(X, y)
                    pred = self.model(X_meta)
                    test_logit.append(pred)
                    test_label.append(y.to(torch.float64))
                else:
                    X_meta = self.get_meta_feature_allclass(X, y, is_training=True)
                    pred = self.model(X_meta, X)
                    test_logit.append(pred)
                    test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label.squeeze()).item()       

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