import math
import torch

from einops import rearrange, repeat

from TALENT.model.lib.bishop.embed import PatchEmb
from TALENT.model.lib.bishop.encoder import Encoder
from TALENT.model.lib.bishop.decoder import Decoder

import torch
import random
import numpy as np
# Source: https://github.com/MAGICS-LAB/BiSHop/blob/main/models/module.py
def ifnone(a, b):
  # From fastai.fastcore
  "`b` if `a` is None else `a`"
  return b if a is None else a

class MLP(torch.nn.Module):
  """
  MLP layer for final prediction.

  Descripition: 

  input_dim    : Input data dimension.
  output_dim   : Output data dimension, e.g., 1 for regression.
  actv         : Activation function in MLP.
  bn           : Add batch normalization after each layer or not.
  bn_final     : Add batch normalization in the final layer.
  dropout      : Dropout ratio in MLP.
  hidden       : MLP hidden layers (exclude final layer).
  skip_connect : Add skip connection in MLP or not.
  softmax      : Whether to use softmax in the last layer of MLP (for classification) or not.
  """
  def __init__(self, input_dim, output_dim,  actv=None, bn=True, bn_final=False, dropout=0.2, 
               hidden=(4, 2, 1), skip_connect=False, softmax=False):
    super().__init__()

    layers = []
    
    hidden_configs = list(map(lambda t: input_dim * t, hidden))
    mlp_configs = [input_dim, *hidden_configs, output_dim]
    dim_pairs = list(zip(mlp_configs[:-1], mlp_configs[1:]))
    # print(dim_pairs)
    for i, (dim_in, dim_out) in enumerate(dim_pairs):
      is_last = i > (len(hidden) - 1)
      if bn and (not is_last or bn_final): layers.append(torch.nn.BatchNorm1d(dim_in))
      if dropout and not is_last:
        layers.append(torch.nn.Dropout(dropout))
      layers.append(torch.nn.Linear(dim_in, dim_out))
      if is_last:
        break
      layers.append(ifnone(actv, torch.nn.ReLU()))

    self.mlp = torch.nn.Sequential(*layers)
    self.shortcut = torch.nn.Linear(mlp_configs[0], mlp_configs[-1]) if skip_connect else None

    self.softmax = torch.nn.Softmax(dim=1) if softmax and (output_dim>1) else None

  def forward(self, x):
    y = self.mlp(x)
    if self.shortcut: 
      y += self.shortcut(x) 
    if self.softmax:
      # print(y.shape)
      return self.softmax(y)
    return y


class BAModel(torch.nn.Module):
  """
  BAModel includes patch embedding, encoder and decoder.



  """
  def __init__(self, feat_dim, emb_dim=32, out_dim=24, patch_dim=8, factor=8, n_agg=4, actv='entmax', 
               hopfield=True, d_model=512, d_ff=1024, n_heads=8, e_layer=3, d_layer=4, dropout=0.2):
    super(BAModel, self).__init__()

    self.feat_dim = feat_dim
    self.emb_dim = emb_dim
    self.patch_dim = patch_dim

    # Padding to original data to avoid indivisable
    # pad_feat_dim: the padded feature dimension
    # pad_out_dim:
    # pad_feat_add: dimensions need to add to make the feat_dim divisable by patch_dim
    self.pad_feat_dim = math.ceil(1.0 * feat_dim / patch_dim) * patch_dim
    self.pad_out_dim = math.ceil(1.0 * out_dim / patch_dim) * patch_dim
    self.pad_feat_add = self.pad_feat_dim - feat_dim


    self.PatchEmb = PatchEmb(patch_dim, d_model)
    self.pos_enc_encoder = torch.nn.Parameter(torch.randn(1, emb_dim, self.pad_feat_dim // patch_dim, d_model))
    self.pre_norm = torch.nn.LayerNorm(d_model)

    n_pool_enc = self.pad_feat_dim // patch_dim
    n_pool_dec = self.pad_out_dim // patch_dim

    self.encoder = Encoder(e_layers=e_layer, n_agg=n_agg, d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                           dropout=dropout, n_pool=n_pool_enc, factor=factor, actv=actv, hopfield=hopfield)
    
    self.n_pool_records = [n_pool_enc]
    for i in range(1, e_layer + 1):
      self.n_pool_records.append(math.ceil(self.n_pool_records[-1]/n_agg**i))
    self.n_pool_total = sum(self.n_pool_records)

    self.d_layer = d_layer
    if d_layer > 0:
      if d_layer > (e_layer + 1):
        d_layer = e_layer + 1
      self.pos_enc_decoder = torch.nn.Parameter(torch.randn(1, emb_dim, n_pool_dec, d_model))
      self.decoder = Decoder(d_layer=d_layer, patch_dim=patch_dim, n_pool=n_pool_dec, factor=factor, actv=actv, 
                             hopfield=hopfield, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    elif d_layer == 0:    
      self.decoder = torch.nn.Linear(self.n_pool_records[-1]*d_model, self.pad_out_dim)
    else:
      self.decoder = torch.nn.Linear(self.n_pool_total*d_model, self.pad_out_dim)

  def forward(self, x):
    batch = x.shape[0]
    if self.pad_feat_add != 0:
      x = torch.cat((x[:, :1, :].expand(-1, self.pad_feat_add, -1), x), dim=1)
    x = self.PatchEmb(x)
    x += self.pos_enc_encoder
    x = self.pre_norm(x)
    enc_out = self.encoder(x)
    if self.d_layer>0:
      dec_in = repeat(self.pos_enc_decoder, 'b emb_dim n_pool_dec d_model -> (repeat b) emb_dim n_pool_dec d_model', repeat=batch)

      dec_final = self.decoder(dec_in, enc_out)

      return dec_final
    elif self.d_layer==0:
      enc_final = enc_out[-1]
    else:
      enc_final = torch.cat(enc_out, -2) 
    enc_final = rearrange(enc_final, 'b emb_dim n_pools d_model -> b emb_dim (n_pools d_model)')
    dec_final = self.decoder(enc_final)
    dec_final = rearrange(dec_final, 'b emb_dim out_dim -> b out_dim emb_dim')
    return dec_final
    
    