import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from .utils import *

#Source: https://github.com/AI-sandbox/HyperFast/blob/main/hyperfast/model.py
class HyperFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_dims = cfg.n_dims
        self.max_categories = cfg.max_categories
        self.rf_size = cfg.rf_size
        self.torch_pca = cfg.torch_pca
        self.clip_data_value = cfg.clip_data_value
        self.hn_n_layers = cfg.hn_n_layers
        self.hn_hidden_size = cfg.hn_hidden_size
        self.main_n_layers = cfg.main_n_layers

        middle_layers = []
        for n in range(self.hn_n_layers - 2):
            middle_layers.append(nn.Linear(self.hn_hidden_size, self.hn_hidden_size))
            middle_layers.append(nn.ReLU())
        self.num_input_features_hn = self.n_dims + self.max_categories

        self.hypernetworks = nn.ModuleList()
        self.hn_emb_to_weights = nn.ModuleList()

        for n in range(self.main_n_layers - 1):
            if n > 0:
                self.num_input_features_hn = self.n_dims * 2 + self.max_categories
            num_input_features_hn = self.num_input_features_hn + self.n_dims * 2

            hn_layers = []
            hn_layers.append(nn.Linear(num_input_features_hn, self.hn_hidden_size))
            hn_layers.append(nn.ReLU())
            hn_layers = hn_layers + middle_layers

            self.hypernetworks.append(nn.Sequential(*hn_layers))
            self.output_size_hn = (self.n_dims + 1) * self.n_dims
            self.hn_emb_to_weights.append(
                nn.Linear(self.hn_hidden_size, self.output_size_hn)
            )

        hn_layers = []
        last_hn_output_size = self.n_dims + 1
        self.num_input_features_hn += self.n_dims * 2

        hn_layers.append(nn.Linear(self.num_input_features_hn, self.hn_hidden_size))
        hn_layers.append(nn.ReLU())
        hn_layers = hn_layers + middle_layers
        hn_layers.append(nn.Linear(self.hn_hidden_size, last_hn_output_size))
        self.hypernetworks.append(nn.Sequential(*hn_layers))
        self.nn_bias = nn.Parameter(torch.ones(2))

    def forward(self, X, y, n_classes):
        X = X.flatten(start_dim=1)
        rf_linear = nn.Linear(X.shape[1], self.rf_size, bias=True) # random feature
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.kaiming_uniform_(rf_linear.weight, mode="fan_out", nonlinearity="relu") # test
        nn.init.normal_(rf_linear.bias, 0, 0.01)
        rf_linear.weight.requires_grad = False
        rf = nn.Sequential(rf_linear, nn.ReLU()).to(X.device)
        with torch.no_grad():
            X = rf(X)
        if self.torch_pca: # PCA
            self.pca = TorchPCA(n_components=self.n_dims)
        else:
            self.pca = PCA(n_components=self.n_dims)
        if self.torch_pca:
            X = self.pca.fit_transform(X)
        else:
            X = torch.from_numpy(self.pca.fit_transform(X.cpu().numpy())).to(X.device)
        X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)

        out = X
        pca_global_mean = torch.mean(out, axis=0)
        pca_perclass_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                class_mean = torch.mean(out[y == lab], dim=0, keepdim=True)
            else:
                class_mean = torch.mean(out, dim=0, keepdim=True)
            pca_perclass_mean.append(class_mean)
        pca_perclass_mean = torch.cat(pca_perclass_mean)

        pca_concat = []
        for ii, lab in enumerate(y):
            if pca_perclass_mean.ndim == 1:
                pca_perclass_mean = pca_perclass_mean.unsqueeze(0)
            if out.ndim == 1:
                out = out.unsqueeze(0)

            lab_index = lab.item() if torch.is_tensor(lab) else lab
            lab_index = min(lab_index, pca_perclass_mean.size(0) - 1)

            row = torch.cat((out[ii], pca_global_mean, pca_perclass_mean[lab_index]))
            pca_concat.append(row)
        pca_output = torch.vstack(pca_concat)
        y_onehot = F.one_hot(y, self.max_categories)

        main_network = []
        for n in range(self.main_n_layers - 1):
            if n > 0:
                data = torch.cat((out, pca_output, y_onehot), dim=1)
            else:
                data = torch.cat((pca_output, y_onehot), dim=1)
            if n % 2 == 0:
                residual_connection = out

            weights = get_main_weights(
                data, self.hypernetworks[n], self.hn_emb_to_weights[n]
            )
            out, main_linear_layer = forward_linear_layer(out, weights, self.n_dims)
            if n % 2 == 0:
                out = F.relu(out)
            else:
                out = out + residual_connection
                out = F.relu(out)
            main_network.append(main_linear_layer)
        data = torch.cat((out, pca_output, y_onehot), dim=1)
        weights_per_sample = get_main_weights(data, self.hypernetworks[-1])

        weights = []
        last_input_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                w = torch.mean(weights_per_sample[y == lab], dim=0, keepdim=True)
                input_mean = torch.mean(out[y == lab], dim=0, keepdim=True)
            else:
                w = torch.mean(weights_per_sample, dim=0, keepdim=True)
                input_mean = torch.mean(out, dim=0, keepdim=True)
            weights.append(w)
            last_input_mean.append(input_mean)
        weights = torch.cat(weights)
        last_input_mean = torch.cat(last_input_mean)
        weights[:, :-1] = weights[:, :-1] + last_input_mean
        weights = weights.T
        out, last_linear_layer = forward_linear_layer(out, weights, n_classes)
        main_network.append(last_linear_layer)

        return rf, self.pca, main_network
