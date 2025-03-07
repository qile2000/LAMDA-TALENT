# %%
import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from TALENT.model.models.ftt import get_nonglu_activation_fn, get_activation_fn,Tokenizer

class MultiheadGEAttention(nn.Module):
    """
    FR-Graph integrated attention
    ---
    Learn relations among features and feature selection strategy in data-driven manner.

    """
    def __init__(
        # Normal Attention Args
        self, d: int, n_heads: int, dropout: float, initialization: str,
        # FR-Graph Args
        n: int, sym_weight: bool = True, sym_topology: bool = False, nsi: bool = True,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        
        """FR-Graph Params: Edge weights"""
        # head and tail transformation
        self.W_head = nn.Linear(d, d)
        if sym_weight:
            self.W_tail = self.W_head # symmetric weights
        else:
            self.W_tail = nn.Linear(d, d) # ASYM
        # relation embedding: learnable diagonal matrix
        self.rel_emb = nn.Parameter(torch.ones(n_heads, d // self.n_heads))

        for m in [self.W_head, self.W_tail, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

        """FR-Graph Params: Graph topology (column = node = feature)"""
        self.n_cols = n + 1 # Num of Nodes: input feature nodes + [Cross-level Readout]
        self.nsi = nsi # no self-interaction

        # column embeddings: semantics for each column
        d_col = math.ceil(2 * math.log2(self.n_cols)) # dim for column header embedding -> d_header += d
        self.col_head = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        if not sym_topology:
            self.col_tail = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        else:
            self.col_tail = self.col_head # share the parameter
        for W in [self.col_head, self.col_tail]:
            if W is not None:
                # correspond to Tokenizer initialization
                nn_init.kaiming_uniform_(W, a=math.sqrt(5))
        
        # Learnable bias and fixed threshold for topology
        self.bias = nn.Parameter(torch.zeros(1))
        self.threshold = 0.5

        """Frozen topology"""
        # for some sensitive datasets set to `True`
        # after training several epoch, which helps
        # stability and better performance
        self.frozen = False


    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
        )
    
    def _no_self_interaction(self, x):
        if x.shape[-2] == 1: # only [Readout Node]
            return x
        assert x.shape[-1] == x.shape[-2] == self.n_cols
        # mask diagonal interaction
        nsi_mask = 1.0 - torch.diag_embed(torch.ones(self.n_cols, device=x.device))
        return x * nsi_mask
    
    def _prune_to_readout(self, x):
        """Prune edges from any features to [Readout Node]"""
        assert x.shape[-1] == self.n_cols
        mask = torch.ones(self.n_cols, device=x.device)
        mask[0] = 0 # zero out interactions from features to [Readout]
        return x * mask
    
    def _get_topology(self, top_score, elewise_func=torch.sigmoid):
        """
        Learning static knowledge topology (adjacency matrix)
        ---
        top_score: N x N tensor, relation topology score
        adj: adjacency matrix A of FR-Graph
        """
        adj_probs = elewise_func(top_score + self.bias) # choose `sigmoid` as element-wise activation (sigma1)
        if self.nsi:
            adj_probs = self._no_self_interaction(adj_probs) # apply `nsi` function
        adj_probs = self._prune_to_readout(adj_probs) # cut edges from features to [Readout]
        
        if not self.frozen:
            # using `Straight-through` tirck for non-differentiable operation
            adj = (adj_probs > 0.5).float() - adj_probs.detach() + adj_probs
        else:
            # frozen graph topology: no gradient
            adj = (adj_probs > 0.5).float()
        return adj

    def forward(
        self,
        x_head: Tensor,
        x_tail: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
        elewise_func = torch.sigmoid,
        comp_func = torch.softmax,
    ) -> Tensor:
        f_head, f_tail, f_v = self.W_head(x_head), self.W_tail(x_tail), self.W_v(x_tail)
        for tensor in [f_head, f_tail, f_v]:
            # check multi-head
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            f_tail = key_compression(f_tail.transpose(1, 2)).transpose(1, 2)
            f_v = value_compression(f_v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(f_head)
        d_head_tail = f_tail.shape[-1] // self.n_heads
        d_value = f_v.shape[-1] // self.n_heads
        n_head_nodes = f_head.shape[1]

        # reshape to multi-head view
        f_head = self._reshape(f_head)
        f_tail = self._reshape(f_tail)

        # edge weight scores (Gw)
        weight_score = f_head @ torch.diag_embed(self.rel_emb) @ f_tail.transpose(-1, -2) / math.sqrt(d_head_tail)
        
        col_emb_head = F.normalize(self.col_head, p=2, dim=-1) # L2 normalized column embeddings
        col_emb_tail = F.normalize(self.col_tail, p=2, dim=-1)
        # topology score (Gt)
        top_score = col_emb_head @ col_emb_tail.transpose(-1, -2)
        # graph topology (A)
        adj = self._get_topology(top_score, elewise_func)
        if n_head_nodes == 1: # only [Cross-level Readout]
            adj = adj[:, :1]
        
        # graph assembling: apply FR-Graph on interaction like attention mask
        adj_mask = (1.0 - adj) * -10000 # analogous to attention mask

        # FR-Graph of this layer
        # Can be used for visualization on Feature Relation and Readout Collection
        fr_graph = comp_func(weight_score + adj_mask, dim=-1) # choose `softmax` as competitive function

        if self.dropout is not None:
            fr_graph = self.dropout(fr_graph)
        x = fr_graph @ self._reshape(f_v)
        x = (
            x.transpose(1, 2)
            .reshape(batch_size, n_head_nodes, self.n_heads * d_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, fr_graph.detach()


class T2GFormer(nn.Module):
    """T2G-Former

    References:
    - FT-Transformer: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py#L151
    """
    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        # graph estimator
        sym_weight: bool = True,
        sym_topology: bool = False,
        nsi: bool = True,
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        n_tokens = d_numerical if categories is None else d_numerical + len(categories)
        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadGEAttention(
                        d_token, n_heads, attention_dropout, initialization,
                        n_tokens, sym_weight=sym_weight, sym_topology=sym_topology, nsi=nsi,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

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

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_fr: bool = False) -> Tensor:
        fr_graphs = [] # FR-Graph of each layer
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual, fr_graph = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            fr_graphs.append(fr_graph)
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x if not return_fr else (x, fr_graphs)
    
    def froze_topology(self):
        """API to froze FR-Graph topology in training"""
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            layer['attention'].frozen = True