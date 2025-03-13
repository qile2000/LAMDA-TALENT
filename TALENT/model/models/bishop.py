import math
import torch

from einops import rearrange, repeat

from TALENT.model.lib.bishop.embed import CatEmb, NumEmb
from TALENT.model.lib.bishop.encoder import Encoder

from TALENT.model.lib.bishop.module import MLP, BAModel

import torch
import random
import numpy as np

# Source: https://github.com/MAGICS-LAB/BiSHop/blob/main/models/model.py
class BiSHop(torch.nn.Module):
  """
  The main class for BiSHop model.

  Description: TO BE FILLED

  Parameters
  ----------
  n_cat        : Number of categorical features.
  n_num        : Number of numerical features.
  n_out        : Output features.

  n_class      : Number of classes in categorical features.
  share        : Whether to include shared embedding or not.
  share_div    : Division factor for shared embedding; shared embedding dimension = emb_dim / share_div.
  share_add    : Whether to add shared embedding to identical embedding or concatenate them.
  full_dropout : Whether to apply full embedding dropout or not.
  emb_dropout  : Dropout ratio for categorical embedding.

  emb_dim      : Initial embedding dimensions for categorical and numerical features.
  e_layer      : Number of layers in the encoder.
  d_layer      : Number of layers in the decoder; set to encoder+1 if lager than encoder, 
                 Set 0 if using final output of encoder, -1 if using all outputs of encoder.
  out_dim      : Number of decoded representations (after hierarchical multi-cell learning).
  patch_dim    : Length of stride in patch embeddings. (Seg Length)
  factor       : Factor for BAModule.
  flip         : Flip cross-feature and cross embedding.
  n_agg        : Number of aggregations in each level of the encoder.
  dropout      : Dropout ratio in hierarchical multi-cell learning.
  actv         : Activation function in GSH (Generalized Sparse Hopfield module).
  hopfield     : True if using Hopfield attention, False if using classical Transformer attention.
  d_model      : Number of features in GSH (Generalized Sparse Hopfield module) inputs.
  d_ff         : Dimension of the feedforward network in GSH (Generalized Sparse Hopfield module).
  n_heads      : Number of heads in the multi-head attention models in GSH (Generalized Sparse Hopfield module).

  mlp_actv     : Activation function in MLP.
  mlp_bn       : Add batch normalization after each layer or not.
  mlp_bn_final : Add batch normalization in the final layer.
  mlp_dropout  : Dropout ratio in MLP.
  mlp_hidden   : MLP hidden layers (exclude final layer).
  mlp_skip     : Add skip connection in MLP or not.
  mlp_softmax   : Whether to use softmax in the last layer of MLP (for classification) or not.

  device       : Model device.
  """
  def __init__(self, n_cat, n_num, n_out, n_class, emb_dim=32, out_dim=24, patch_dim=8, factor=8, n_agg=4, 
                d_model=512, n_heads=8, e_layer=3, d_layer=3,  emb_dropout=0.1,
                dropout=0.2, share_div=8, mlp_dropout=0.2,d_ff=None,hopfield=True,share_add=False,
                share=True, mlp_bn=True, mlp_bn_final=False, flip=True, full_dropout=False,
                mlp_skip=True, mlp_softmax=False,actv='entmax',
                mlp_hidden=(4,2,1),mlp_actv=torch.nn.ReLU()):
    super(BiSHop, self).__init__()
    
    self.n_cat = n_cat
    self.n_num = n_num
    if n_cat != 0:
      self.CatEmb = CatEmb(n_cat, emb_dim, n_class, share, share_add, share_div, full_dropout, emb_dropout)
      self.Emb = torch.nn.Embedding(num_embeddings=n_class, embedding_dim=emb_dim)
    if n_num != 0:
      self.NumEmb = NumEmb(n_num, emb_dim)
    
    if flip:
      self.emb_dim = n_cat + n_num
      self.feat_dim = emb_dim
    else: 
      self.emb_dim = emb_dim
      self.feat_dim = n_cat + n_num

    self.flip = flip
    self.BAModel = BAModel(feat_dim=self.feat_dim, emb_dim=self.emb_dim, out_dim=out_dim, patch_dim=patch_dim, factor=factor,
                           n_agg=n_agg, actv=actv, hopfield=hopfield, d_model=d_model, d_ff=d_ff, n_heads=n_heads, 
                           e_layer=e_layer, d_layer=d_layer, dropout=dropout)
    # self.BAModel.double()
    mlp_in = self.emb_dim * self.BAModel.pad_out_dim
    mlp_out = n_out
    self.MLP = MLP(mlp_in, mlp_out, actv=mlp_actv,  bn=mlp_bn, bn_final=mlp_bn_final, dropout=mlp_dropout,
                   hidden=mlp_hidden, skip_connect=mlp_skip, softmax=mlp_softmax)
  def get_bins(self, data):
    self.NumEmb.get_bins(data)

  def forward(self, x_num, x_cat):
    if self.n_cat != 0:
      cat_emb = self.CatEmb(x_cat)
      # cat_emb = self.Emb(x_cat)
    if self.n_num != 0:
      num_emb = self.NumEmb(x_num)
    
    if self.n_cat == 0:
      x = num_emb
    if self.n_num == 0:
      x = cat_emb
    if self.n_cat != 0 and self.n_num != 0:
      x = torch.cat([cat_emb, num_emb], dim=1)

    if self.flip:
      x = rearrange(x, 'b n e -> b e n')
    x = self.BAModel(x)
    x = rearrange(x, 'b e o -> b (e o)')
    return self.MLP(x)
  
  def _to(self, device):
    if self.n_cat != 0:
      self.CatEmb = self.CatEmb.to(device)
      self.Emb = self.Emb.to(device)
    if self.n_num != 0:
      self.NumEmb._to(device)
    self.BAModel = self.BAModel.to(device)
    self.MLP = self.MLP.to(device)