import numpy
import torch

from einops import rearrange, repeat

import torch
import random
import numpy as np
# Source: https://github.com/MAGICS-LAB/BiSHop/blob/main/models/embed.py
class NumEmb(torch.nn.Module):
  """
  Numerical embedding for tabular data

  Description: 
    TO BE FILLED

  Parameters:
  ----------
  n_num   : Number of categorical features.
  emb_dim : Number of classes in categorical features.
  """
  def __init__(self, n_num, emb_dim):
    super(NumEmb, self).__init__()
    assert emb_dim >= 1, "embed_dim should be greater then 0"
    self.n_num = n_num
    self.n_bins = emb_dim

    self.bins = torch.linspace(0, 1, steps=emb_dim+1)
    self.quantiles = torch.zeros((n_num, emb_dim+1))
  

  def get_bins(self, data, identifier='num'):
    if isinstance(data, torch.utils.data.DataLoader):
      num_data = [batch[identifier] for batch in data]
      num_data = torch.cat(num_data)
    elif isinstance(data, torch.Tensor):
      num_data = data
    else:
      raise TypeError('Input should be DataLoader or torch tensor')

    for i in range(self.n_num):
      self.quantiles[i] = torch.quantile(num_data[:, i], self.bins).to(num_data.device)

  def forward(self, x):
    batch_size, n_num = x.shape
    assert n_num == self.n_num, "Feature dimension should match number of numerical features"

    x_expanded = repeat(x, 'b f -> b f e', e=self.n_bins)
    bins_expanded = repeat(self.quantiles, 'f e-> b f e', b=batch_size)
    
    left_bins = bins_expanded[:, :, :-1].to(x_expanded.device)
    right_bins = bins_expanded[:, :, 1:].to(x_expanded.device)
    
    double_one = torch.tensor(1.0, dtype=torch.float)  # Convert 1.0 to double
    double_one = double_one.to(x_expanded.device)
    double_zero = torch.tensor(0.0, dtype=torch.float)  # Convert 1.0 to double
    double_zero = double_zero.to(x_expanded.device)
    x_expanded = x_expanded.float()            # Convert x_expanded to double
    right_bins = right_bins.float()    
    left_bins = left_bins.float()    

    # print(x.shape, x_expanded.dtype, right_bins.dtype, left_bins.dtype)
    embeddings = torch.where(x_expanded < left_bins, double_zero,
                             torch.where(x_expanded >= right_bins, double_one, 
                                         (x_expanded - left_bins) / (right_bins - left_bins)))
    # embeddings = torch.where(x_expanded < left_bins, 0,
    #                          torch.where(x_expanded >= right_bins, 1, 
    #                                      (x_expanded - left_bins) / (right_bins - left_bins)))
    #                                          double_one = torch.tensor(1.0, dtype=torch.float)  # Convert 1.0 to double

    return embeddings.to(x.dtype)
  
  def _to(self, device):
    self.bins = self.bins.to(device)
    self.quantiles = self.quantiles.to(device)

class FullEmbDropout(torch.nn.Module):
  def __init__(self, dropout: float=0.1):
    super(FullEmbDropout, self).__init__()
    self.dropout=dropout

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    dist = torch.bernoulli(torch.ones(X.size(1), 1) * (1 - self.dropout)).to(X.device)
    mask = dist.expand_as(X) / (1 - self.dropout)
    return mask * X
        
def _trunc_normal_(x, mean=0., std=1.):
  "Truncated normal initialization (approximation)"
  # From fastai.layers
  # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
  return x.normal_().fmod_(2).mul_(std).add_(mean)


class _Embedding(torch.nn.Embedding):
  "Embedding layer with truncated normal initialization"
  # From fastai.layers
  def __init__(self, ni, nf, std=0.01):
    super(_Embedding, self).__init__(ni, nf)
    _trunc_normal_(self.weight.data, std=std)
        
class SharedEmbedding(torch.nn.Module):
  def __init__(self, n_class, emb_dim, share=True, share_add=False, share_div=8):
    super().__init__()
    if share:
      if share_add:
        shared_embed_dim = emb_dim
        self.embed = _Embedding(n_class, emb_dim)
      else:
        shared_embed_dim = emb_dim // share_div
        self.embed = _Embedding(n_class, emb_dim - shared_embed_dim)
      self.share = torch.nn.Parameter(torch.empty(1, 1, shared_embed_dim))
      _trunc_normal_(self.share.data, std=0.01)
      self.share_add = share_add
    else: 
      self.embed = _Embedding(n_class, emb_dim)
      self.share = None

  def forward(self, x):
    out = self.embed(x).unsqueeze(1)
    if self.share is None: return out
    if self.share_add:
      out += self.share
    else:
      share = self.share.expand(out.shape[0], -1, -1)
      out = torch.cat((out, share), dim=-1)
    return out

# class SharedEmbedding(torch.nn.Module):
#   def __init__(self, n_class, emb_dim, share=True, share_add=False, share_div=8):
#     super().__init__()
#     if share:
#       if share_add:
#         shared_embed_dim = emb_dim
#         self.embed = torch.nn.Embedding(n_class, emb_dim)
#       else:
#         # shared_embed_dim = emb_dim // share_div
#         shared_embed_dim = 0
#         self.embed = torch.nn.Embedding(n_class, emb_dim - shared_embed_dim)
#       self.share = torch.nn.Parameter(torch.empty(1, 1, shared_embed_dim))
#       _trunc_normal_(self.share.data, std=0.01)
#       self.share_add = share_add
#     else: 
#       self.embed = torch.nn.Embedding(n_class, emb_dim)
#       self.share = None

#   def forward(self, x):
#     out = self.embed(x).unsqueeze(1)
#     if self.share is None: return out
#     if self.share_add:
#       out += self.share
#     else:
#       share = self.share.expand(out.shape[0], -1, -1)
#       out = torch.cat((out, share), dim=-1)
#     return out

class CatEmb(torch.nn.Module):
  """
  Categorical embedding for tabular data

  Description: 

  Parameters
  ----------
  n_cat        : Number of categorical features.
  n_class      : Number of classes in categorical features.
  emb_dim      : Embedding dimensions for categorical features.
  share        : Include shared embedding or not.
  share_div    : Shared embedding division, shared embedding dimenion = emb_dim/share_div
  share_add    : Add shared embedding to indentical embedding or concate
  full_dropout : Full embedding dropout or not
  emb_dropout  : Dropout ratio for categoircal embeddding
  n_heads      : Number of heads in the multi-head attention models in GSH (Generalized Sparse Hopfield module).
  """
  def __init__(self, n_cat, emb_dim, n_class, share=True, share_add=False, share_div=8, full_dropout=False, emb_dropout=0.1):
    super(CatEmb, self).__init__()
    self.full_dropout = full_dropout
    self.share = share
    if share:
      self.embeddings = torch.nn.ModuleList([SharedEmbedding(n_class, emb_dim, share, share_add, share_div) 
                                             for _ in range(n_cat)])
    else:
      self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(n_class, emb_dim) for _ in range(n_cat)])

    if full_dropout:
      self.embedding_dropout = FullEmbDropout(emb_dropout)
    else:
      self.embedding_dropout = torch.nn.Dropout(emb_dropout)  

  def forward(self, x):
    _, n_cat = x.shape
    x = torch.cat([self.embeddings[i](x[:, i]) for i in range(n_cat)], dim=1)
    if not self.share and self.embedding_dropout is not None:
      x = self.embedding_dropout(x)
    return x
  
class PatchEmb(torch.nn.Module):
  """
  Patch embedding in aggregating different features in to ones.

  Description: 
  ----------
  It begins by dividing the input features 'x' into 'n' patches, each containing 'patch_dim' features. 
  Subsequently, a linear layer is employed to transform each patch within the batch into a hidden 
  representation with a dimensionality of 'd_model'.

  Parameters:
  ----------
  patch_dim : 
  d_model   : Expected number of features in attention machnism

  Input: 
  ----------
  Embedded tabular data: (batch size, feature dimension, embedding dimension)
    
  Output:
  ----------
  Patched embedded tabular data: (batch size, embedding dimension, number of patches, d_model)
  """
  def __init__(self, patch_dim, d_model):
    super(PatchEmb, self).__init__()
    self.patch_dim = patch_dim
    self.linear = torch.nn.Linear(patch_dim, d_model)

  def forward(self, x):
    batch_size, feat_dim, emb_dim = x.shape
    x_patch = rearrange(x, 'b (n_patch patch_dim) d -> (b d n_patch) patch_dim', patch_dim=self.patch_dim)
    x_emb = self.linear(x_patch)
    x_emb = rearrange(x_emb, '(b d n_patch) d_model -> b d n_patch d_model', b=batch_size, d=emb_dim)
    return x_emb
  
  
