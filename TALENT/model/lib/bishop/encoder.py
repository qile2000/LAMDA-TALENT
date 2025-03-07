import math
import torch

from TALENT.model.lib.bishop.attn import BAModule
from einops import repeat

import torch
import random
import numpy as np
# Source: https://github.com/MAGICS-LAB/BiSHop/blob/main/models/encoder.py
class PatchMerge(torch.nn.Module):
  """
  Merge adjecent patch together .
  
  Description
  ----------
  TO BE FILLED
  
  Parameters
  ----------
  d_model : Number of features.
  n_agg   : number of sperate patched to aggregate together

  Input
  ----------
  
  Output
  ----------
  """
  def __init__(self, d_model=512, n_agg=4):
    super(PatchMerge, self).__init__()
    self.d_model = d_model
    self.n_agg = n_agg
    self.linear = torch.nn.Linear(n_agg*d_model, d_model)
    self.pre_norm = torch.nn.LayerNorm(n_agg*d_model)
    

  def forward(self, x):
    batch_size, emb_dim, n_patch, d_model = x.shape

    if n_patch < self.n_agg:
      # print(Warning('Please consider use a smaller n_patch'))
      x = repeat(x, 'b e n d -> b e (repeat n) d', repeat=math.ceil(self.n_agg / n_patch))
      x = x[:, :, :self.n_agg, :]
    else:
      pad_num = n_patch % self.n_agg
      if pad_num != 0:
        pad_num = self.n_agg - pad_num
        x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
    
    patch_merge = []
    for i in range(self.n_agg):
      patch_merge.append(x[:, :, i::self.n_agg, :])
      # print(x[:, :, i::self.n_agg, :].shape, self.n_agg, 'x to merge')
    x = torch.cat(patch_merge, -1)
    
    x = self.pre_norm(x)
    x = self.linear(x)

    return x
    
class EncoderLayer(torch.nn.Module):
  """
  The encoder layer.
  
  Description
  ----------
  TO BE FILLED
  
  Parameters
  ----------
  d_model : Number of features.
  n_agg   : number of sperate patched to aggregate together

  Input
  ----------
  
  Output
  ----------
  """
  def __init__(self, n_agg=4, n_pool=10, factor=10, actv='entmax', hopfield=True, d_model=512, n_heads=8, d_ff=1024, dropout=0.2,):
    super(EncoderLayer, self).__init__()

    if n_agg > 1:
      self.merge = PatchMerge(d_model=d_model, n_agg=n_agg)
    else:
      self.merge = None
    
    self.BAModule = BAModule(n_pool=n_pool, factor=factor, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, actv=actv, hopfield=hopfield)

  def forward(self, x):
    _, emb_dim, _, _ = x.shape
    if self.merge:
      x = self.merge(x)
    x = self.BAModule(x)
    return x
  
class Encoder(torch.nn.Module):
  """
  Merge adjecent patch together .
  
  Description
  ----------
  TO BE FILLED
  
  Parameters
  ----------
  d_model : Number of features.
  n_agg   : number of sperate patched to aggregate together

  Input
  ----------
  
  Output
  ----------
  """
  def __init__(self, e_layers=3, n_agg=4, d_model=512, n_heads=8, d_ff=1024, dropout=0.2, n_pool=10, factor=10, actv='entmax', hopfield=True):
    super(Encoder, self).__init__()
    self.encoder_blocks = torch.nn.ModuleList()
    self.encoder_blocks.append(EncoderLayer(n_agg=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,  n_pool=n_pool, factor=factor, actv=actv, hopfield=hopfield))
    for i in range(1, e_layers):
      self.encoder_blocks.append(EncoderLayer(n_agg=n_agg, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,  n_pool=math.ceil(n_pool/n_agg**i), factor=factor, actv=actv, hopfield=hopfield))

  def forward(self, x):
    encode_x = []
    encode_x.append(x)
    for block in self.encoder_blocks:
      x = block(x)
      encode_x.append(x)

    return encode_x