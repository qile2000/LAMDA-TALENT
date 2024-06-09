import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib.tabr.utils import make_module
from typing import Optional, Union
class Residual_block(nn.Module):
    def __init__(self,d_in,d,dropout):
        super().__init__()
        self.linear0=nn.Linear(d_in,d)
        self.Linear1=nn.Linear(d,d_in)
        self.bn=nn.BatchNorm1d(d_in)
        self.dropout=nn.Dropout(dropout)
        self.activation=nn.ReLU()
    def forward(self, x):
        z=self.bn(x)
        z=self.linear0(z)
        z=self.activation(z)
        z=self.dropout(z)
        z=self.Linear1(z)
        # z=x+z
        return z

class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num:int,
        d_out: int,
        dim:int,
        dropout:int,
        d_block:int,
        n_blocks:int,
        num_embeddings: Optional[dict],
        temperature:float=1.0,
        sample_rate:float=0.8
        ) -> None:

        super().__init__()
        self.d_in = d_in if num_embeddings is None  else d_num*num_embeddings['d_embedding']+d_in-d_num      
        self.d_out = d_out
        self.d_num=d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T=temperature
        self.sample_rate=sample_rate
        if n_blocks >0:
            self.post_encoder=nn.Sequential()
            for i in range(n_blocks):
                name=f"ResidualBlock{i}"
                self.post_encoder.add_module(name,self.make_layer())
            self.post_encoder.add_module('bn',nn.BatchNorm1d(dim))
        self.encoder = nn.Linear(self.d_in, dim)
        # self.bn=nn.BatchNorm1d(dim)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block=Residual_block(self.dim,self.d_block,self.dropout)
        return block
            

    def forward(self, x, y,
                candidate_x, candidate_y,
                context_size, is_train,
                ):
        if is_train:
            data_size=candidate_x.shape[0]
            retrival_size=int(data_size*self.sample_rate)
            sample_idx=torch.randperm(data_size)[:retrival_size]
            candidate_x=candidate_x[sample_idx]
            candidate_y =candidate_y[sample_idx]
        if self.num_embeddings is not None and self.d_num >0:
            x_num,x_cat=x[:,:self.d_num],x[:,self.d_num:]
            candidate_x_num,candidate_x_cat=candidate_x[:,:self.d_num],candidate_x[:,self.d_num:]
            x_num=self.num_embeddings(x_num).flatten(1)
            candidate_x_num=self.num_embeddings(candidate_x_num).flatten(1)
            x=torch.cat([x_num,x_cat],dim=-1)
            candidate_x=torch.cat([candidate_x_num,candidate_x_cat],dim=-1)
        x=x.double()
        candidate_x=candidate_x.double()
        if self.n_blocks > 0:
            candidate_x =self.post_encoder(self.encoder(candidate_x))
            x = self.post_encoder(self.encoder(x))          
        else:         
            candidate_x = self.encoder(candidate_x)
            x = self.encoder(x)
        if is_train:
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).double()
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1)

        # calculate distance
        distances = torch.cdist(x, candidate_x, p=2)
        distances=distances/self.T
        # remove the label of training index
        if is_train:
            distances = distances.clone().fill_diagonal_(torch.inf)     
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps=1e-7
        if self.d_out>1:
            logits=torch.log(logits+eps)
        return logits.squeeze()
