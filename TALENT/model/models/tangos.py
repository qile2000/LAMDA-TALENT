import math
import typing as ty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# adapted from https://github.com/alanjeffares/TANGOS
# %%
class Tangos(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int, 
        d_layers: ty.List[int],    
        dropout: float,
        lambda1: float,
        lambda2: float,
        subsample: int=50,
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.d_out = d_out 
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.lambda1 = lambda1
        self.lambda2= lambda2
        self.subsample=subsample


    def forward(self, x, x_cat):

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return  logit
    
    def cal_representation(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
            if self.dropout and i!=len(self.layers)-1:
                x = F.dropout(x, self.dropout, self.training)
        return x

    
    def cal_tangos_loss(self,x):
        # x: batch_size x num_features
        batch_size=x.shape[0]
        jacobian=torch.vmap(torch.func.jacrev(self.cal_representation),randomness='different')(x)
        neuron_attr = jacobian.swapaxes(0,1)
        h_dim = neuron_attr.shape[0]    
        if len(neuron_attr.shape) > 3:
            # h_dim x batch_size x features
            neuron_attr = neuron_attr.flatten(start_dim=2)

        # calculate specialization loss component
        spec_loss = torch.norm(neuron_attr, p=1)/(batch_size*h_dim*neuron_attr.shape[2])
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        orth_loss = torch.tensor(0., requires_grad=True).to(x.device)
        # apply subsampling routine for orthogonalization loss
        if self.subsample > 0 and self.subsample < h_dim*(h_dim-1)/2:
            tensor_pairs = [list(np.random.choice(h_dim, size=(2), replace=False)) for i in range(self.subsample)]
            for tensor_pair in tensor_pairs:
                pairwise_corr = cos(neuron_attr[tensor_pair[0], :, :], 
                                    neuron_attr[tensor_pair[1], :, :]).norm(p=1)
                orth_loss = orth_loss + pairwise_corr

            orth_loss = orth_loss/(batch_size*self.subsample)

        else:
            for neuron_i in range(1, h_dim):
                for neuron_j in range(0, neuron_i):
                    pairwise_corr = cos(neuron_attr[neuron_i, :, :],
                                        neuron_attr[neuron_j, :, :]).norm(p=1)
                    orth_loss = orth_loss + pairwise_corr
            num_pairs = h_dim*(h_dim-1)/2
            orth_loss = orth_loss/(batch_size*num_pairs)

        return spec_loss, orth_loss
