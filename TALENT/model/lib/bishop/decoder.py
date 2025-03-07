import torch

from TALENT.model.lib.bishop.attn import GSHLayer, BAModule
from einops import rearrange, repeat
import torch
import random
import numpy as np
# Source: https://github.com/MAGICS-LAB/BiSHop/blob/main/models/decoder.py
class DecoderLayer(torch.nn.Module):
  # patch_dim : seg_len
  # factor : factor
  # n_pool : out_seg_num
  def __init__(self, patch_dim=10, n_pool=10, factor=10, actv='entmax', hopfield=True, 
               d_model=512, n_heads=8, d_ff=1024, dropout=0.2):
    super(DecoderLayer, self).__init__()
    
    self.BAModule = BAModule(n_pool=n_pool, factor=factor, d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                             dropout=dropout, actv=actv, hopfield=hopfield)
    self.GSHLayer = GSHLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, actv=actv, hopfield=hopfield)
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.dropout = torch.nn.Dropout(dropout)
    self.mlp = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), 
                                    torch.nn.GELU(), 
                                    torch.nn.Linear(d_model, d_model))
    self.linear = torch.nn.Linear(d_model, patch_dim)
  
  def forward(self, x, enc_x):
    batch = x.shape[0]
    x = self.BAModule(x)
    x = rearrange(x, 'b emb_dim n_pool d_model -> (b emb_dim) n_pool d_model')

    enc_x = rearrange(enc_x, 'b emb_dim n_pool d_model -> (b emb_dim) n_pool d_model')
    tmp = self.GSHLayer(x, enc_x, enc_x,)

    x = x + self.dropout(tmp)

    y = x = self.norm1(x)

    y = self.mlp(y)

    dec_out = self.norm2(x+y)

    dec_out = rearrange(dec_out, '(b emb_dim) n_out d_model -> b emb_dim n_out d_model', b = batch)
    layer_predict = self.linear(dec_out)
    layer_predict = rearrange(layer_predict, 'b emb_dim n_out seg_len -> b (emb_dim n_out) seg_len')
    return dec_out, layer_predict
  
class Decoder(torch.nn.Module):
  """
  
  Input:

  Output:
  """
  def __init__(self, d_layer=3, patch_dim=10, n_pool=10, factor = 10, actv='entmax', hopfield=True, 
               d_model=512, n_heads=8, d_ff=1024, dropout=0.2):
    super(Decoder, self).__init__()
    self.decode_layers = torch.nn.ModuleList()
    for _ in range(d_layer):
      self.decode_layers.append(DecoderLayer(patch_dim=patch_dim, n_pool=n_pool, factor=factor, actv=actv, hopfield=hopfield, 
                                             d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout))
    
  def forward(self, x, enc):
    final_predict = None
    i = 0
    # print(x.shape)
    emb_dim = x.shape[1]
    for layer in self.decode_layers:
      enc_x = enc[i]
      x, layer_predict = layer(x, enc_x)

      if final_predict is None:
        final_predict = layer_predict
      else:
        final_predict = final_predict + layer_predict
      i += 1
    final_predict = rearrange(final_predict, 'b (emb_dim n_patch) patch_dim -> b (n_patch patch_dim) emb_dim', emb_dim=emb_dim)
    return final_predict


