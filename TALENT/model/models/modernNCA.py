import torch
import torch.nn as nn
import torch.nn.functional as F
from TALENT.model.lib.tabr.utils import make_module
from typing import Optional, Union

class MLP_Block(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d_in)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

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
            self.post_encoder = nn.Sequential(*[
                MLP_Block(dim, d_block, dropout)
                for _ in range(n_blocks)
            ], nn.BatchNorm1d(dim))
        self.encoder = nn.Linear(self.d_in, dim)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block=MLP_Block(self.dim,self.d_block,self.dropout)
        return block
            
    def forward(self, x, y,
                candidate_x, candidate_y, is_train,
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
        # x=x.double()
        # candidate_x=candidate_x.double()
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)
        if self.n_blocks > 0:
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)
        if is_train: 
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1)

        # calculate distance
        # default we use euclidean distance, however, cosine distance is also a good choice for classification.
        # Using cosine distance, you need to tune the temperature. You can add "temperature":["loguniform",1e-5,1] in the configs/opt_space/modernNCA.json file.
        distances = torch.cdist(x, candidate_x, p=2)
        # following is the code for cosine distance
        # x=F.normalize(x,p=2,dim=-1)   
        # candidate_x=F.normalize(candidate_x,p=2,dim=-1)
        # distances=torch.mm(x,candidate_x.T)
        # distances=-distances
        distances=distances/self.T
        # remove the label of training index
        if is_train:
            distances = distances.fill_diagonal_(torch.inf)     
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps=1e-7
        if self.d_out>1:
            # if task type is classification, since the logit is already normalized, we calculate the log of the logit 
            # and use nll_loss to calculate the loss
            logits=torch.log(logits+eps)
        return logits.squeeze(-1)
