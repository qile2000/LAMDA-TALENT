# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from TALENT.model.lib.excelformer import batch_dim_shuffle,batch_feat_shuffle,get_activation_fn,mixup_data



def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

# def attenuated_init(tensor: Tensor, activation='none', scale=1.):
#     if activation in ['relu', 'tanh', 'sigmoid']:
#         nn_init.xavier_uniform_(tensor, gain=nn_init.calculate_gain(activation) * scale)
#     elif activation == 'attention':
#         nn_init.xavier_uniform_(tensor, gain=1 / math.sqrt(2) * scale)
#     else:
#         nn_init.xavier_uniform_(tensor, gain=1.0 * scale)


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.weight2 = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        self.bias2 = nn.Parameter(Tensor(d_bias, d_token)) if bias else None

        # v4
        attenuated_kaiming_uniform_(self.weight)
        attenuated_kaiming_uniform_(self.weight2)
        nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn_init.kaiming_uniform_(self.bias2, a=math.sqrt(5))


    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        x_some = x_num
        assert x_some is not None
        x1 = self.weight[None] * x_num[:, :, None] + self.bias[None]
        x2 = self.weight2[None] * x_num[:, :, None] + self.bias2[None]
        return x1 * torch.tanh(x2)


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, init_scale: float = 0.01
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        # assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for i, m in enumerate([self.W_q, self.W_k, self.W_v]):
            # all small
            attenuated_kaiming_uniform_(m.weight, scale=init_scale)
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            attenuated_kaiming_uniform_(self.W_out.weight)
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
    
    def get_attention_mask(self, input_shape, device):
        bs, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        # attention_mask = seq_ids[None, :].repeat(seq_len, 1) <= seq_ids[:, None]
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        return attention_mask

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key) # b f f
        masks = self.get_attention_mask(attention_scores.shape, attention_scores.device)
        attention = F.softmax(attention_scores + masks, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class ExcelFormer(nn.Module):
    """ExcelFormer with All initialized by small value
    
    initial function: v4
    """
    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        prenormalization: bool,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        init_scale: float = 0.1,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        # assert initialization in ['xavier', 'kaiming']
        n_tokens = d_numerical 
        self.tokenizer = Tokenizer(d_numerical, None, d_token, token_bias)
        self.n_categories = 0 
        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    # Attenuated Initialization
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, init_scale=init_scale
                    ),
                    'linear0': nn.Linear(d_token, d_token * 2),
                    'norm1': make_normalization(),
                }
            )
            # Attenuated Initialization
            attenuated_kaiming_uniform_(layer['linear0'].weight, scale=init_scale)
            nn_init.zeros_(layer['linear0'].bias)

            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn('tanglu')
        self.last_activation = nn.PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        # Attenuated Initialization
        self.head = nn.Linear(d_token, d_out)
        attenuated_kaiming_uniform_(self.head.weight)
        # nn_init.zeros_(self.head.bias)
        self.last_fc = nn.Linear(n_tokens, 1) # b f d -> b 1 d
        attenuated_kaiming_uniform_(self.last_fc.weight)
        # nn_init.zeros_(self.last_fc.bias)
        

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def forward(self, x_num:Tensor,x_cat = None,mix_up: bool=False, beta=0.5, mtype='feat_mix') -> Tensor:

        if mtype == 'niave_mix': # naive mixup
            x_num, feat_masks, shuffled_ids = mixup_data(x_num, beta=beta)
        x = self.tokenizer(x_num) # TODO: replace with PWE  b. f -> b,f ,d
        if mix_up and mtype != 'niave_mix':
            mixup_func = {
                'feat_mix': batch_feat_shuffle,
                'hidden_mix': batch_dim_shuffle,
            }[mtype]
            x, feat_masks, shuffled_ids = mixup_func(x, beta=beta)

        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)

            # reglu
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        x = self.last_fc(x.transpose(1,2))[:,:,0] # b f d -> b d
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x) # TODO: before last_fc？
        x = self.head(x)
        x = x.squeeze(-1)

        if mix_up:
            return x, feat_masks, shuffled_ids
        return x