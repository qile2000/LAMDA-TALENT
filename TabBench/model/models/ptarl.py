import math
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# adapted from https://github.com/HangtingYe/PTaRL
class MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        self.categories = categories

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor(np.insert(categories[:-1],0,0)).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if self.categories is not None:
            x.append(
                self.category_embeddings((x_cat + self.category_offsets[None]).long()).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

class PTARL(nn.Module):
    def __init__(self, input_num, model_type, out_dim,  categories, n_clusters , cluster_centers_, d_layers,dropout,d_embedding, regularize) -> None:
        super().__init__()

        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.topic_num = n_clusters
        self.cluster_centers_ = cluster_centers_
        self.categories = categories
        self.d_layers = d_layers
        self.dropout = dropout
        self.d_embedding = d_embedding
        self.regularize = regularize

        self.build_model()



    def build_model(self):
        
        self.topic = nn.Parameter(torch.tensor(self.cluster_centers_), requires_grad=True)
        self.weight_ = nn.Parameter(torch.tensor(0.5))

        self.encoder = MLP(self.input_num, self.d_layers, self.dropout, self.categories, self.d_embedding)
        self.head = nn.Linear(self.d_layers[-1], self.out_dim)
        self.reduce = nn.Sequential(
                    nn.Linear(self.d_layers[-1], self.d_layers[-1]),
                    nn.GELU(),
                    nn.Dropout(0.1), 
                    nn.Linear(self.d_layers[-1], self.d_layers[-1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.d_layers[-1], self.d_layers[-1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.d_layers[-1], self.topic_num)
                )


    def forward(self, inputs_n, inputs_c):
        inputs_ = self.encoder(inputs_n, inputs_c)
        r_ = self.reduce(inputs_)
        if self.model_type[-2:] == 'ot':
            hid = torch.mm(r_, self.topic)
            return self.head(hid), torch.softmax(r_, dim=1), inputs_, torch.sigmoid(self.weight_)+0.01
        else:
            return self.head(inputs_)
        