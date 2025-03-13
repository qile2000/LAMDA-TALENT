from typing import Iterable, List, Dict, Tuple, Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import TensorInfo, DictDataset
from TALENT.model.lib.realmlp.nn_models.base import FitterFactory, IdentityFitter, Layer, Fitter, Variable
from TALENT.model.lib.realmlp.torch_utils import cat_if_necessary


class SingleEncodingFactory(FitterFactory):
    def __init__(self, create_fitter, min_cat_size=0, max_cat_size=-1):
        super().__init__()
        self.min_cat_size = min_cat_size
        self.max_cat_size = max_cat_size
        self.create_fitter = create_fitter

    def apply_on(self, cat_size: int, n_classes: int):
        # can be overridden
        return cat_size >= self.min_cat_size and (self.max_cat_size < 0 or cat_size <= self.max_cat_size)

    def _create(self, tensor_infos):
        if 'x_cat' not in tensor_infos:
            return IdentityFitter()
        x_cat_sizes = tensor_infos['x_cat'].get_cat_sizes().numpy()
        if len(x_cat_sizes) != 1:
            raise ValueError(
                'SingleEncoderFactory has to be applied to a single category but was applied to category sizes '
                + str(x_cat_sizes))
        cat_size = x_cat_sizes[0]
        n_classes = tensor_infos['y'].get_cat_sizes()[0].item()
        if self.apply_on(cat_size, n_classes):
            return self.create_fitter(tensor_infos)
        return IdentityFitter()


class EncodingLayer(Layer):
    def __init__(self, single_enc_layers: Iterable[Layer], enc_output_name: str, fitter):
        super().__init__(fitter=fitter)
        self.emb_layers = nn.ModuleList(single_enc_layers)
        self.enc_output_name = enc_output_name

    def forward_tensors(self, tensors):
        x_cat = tensors['x_cat']
        prev_output_tensors = [tensors[self.enc_output_name]] if self.enc_output_name in tensors else []

        new_tensors = []
        for i, l in enumerate(self.emb_layers):
            sub_x_cat = x_cat[[slice(None)] * (x_cat.dim() - 1) + [slice(i, i + 1)]]
            sub_tensors = {'x_cat': sub_x_cat}
            if 'y' in tensors:
                sub_tensors['y'] = tensors['y']
            new_tensors.append(l.forward_tensors(sub_tensors))
        output_tensors = prev_output_tensors + [t['x_cont'] for t in new_tensors if 'x_cont' in t]
        if len(output_tensors) == 0:
            # create empty tensor
            new_conts = torch.zeros(*x_cat.shape[:-1], 0, device=x_cat.device, dtype=torch.float32)
        else:
            new_conts = cat_if_necessary(output_tensors, dim=-1)
        cat_tensors = [t['x_cat'] for t in new_tensors if 'x_cat' in t]
        if len(cat_tensors) > 0:
            new_cats = torch.cat(cat_tensors, dim=-1)
            return utils.update_dict(tensors, {self.enc_output_name: new_conts, 'x_cat': new_cats})
        else:
            return utils.update_dict(tensors, {self.enc_output_name: new_conts}, remove_keys='x_cat')

    def _stack(self, layers: List['EncodingLayer']):
        return EncodingLayer([layers[0].emb_layers[i].stack([layers[j].emb_layers[i] for j in range(len(layers))])
                              for i in range(len(layers[0].emb_layers))], layers[0].enc_output_name, layers[0].fitter)


class EncodingFitter(Fitter):
    def __init__(self, single_encoder_fitters: List[Fitter], enc_output_name: str = 'x_cont', **config):
        super().__init__(needs_tensors=any([enc.needs_tensors for enc in single_encoder_fitters]),
                         is_individual=any([enc.is_individual for enc in single_encoder_fitters]))
        self.single_encoder_fitters = single_encoder_fitters
        self.enc_output_name = enc_output_name  # allow to have something other than x_cont
        assert enc_output_name != 'x_cat'

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return sum([f.get_n_params(ti)
                    for f, ti in zip(self.single_encoder_fitters, self._sub_tensor_infos(tensor_infos))])

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        # for splitting categories
        forward_bytes = self._get_n_values(tensor_infos, ['x_cat'])
        forward_bytes += sum([f.get_n_forward(ti)
                              for f, ti in zip(self.single_encoder_fitters, self._sub_tensor_infos(tensor_infos))])
        # for concat
        forward_bytes += self._get_n_values(self.forward_tensor_infos(tensor_infos),
                                            [self.enc_output_name, 'x_cat'])
        return forward_bytes

    def _sub_tensor_infos(self, tensor_infos):
        x_cat_sizes = tensor_infos['x_cat'].get_cat_sizes().numpy()
        if 'y' in tensor_infos:
            return [{'x_cat': TensorInfo(cat_sizes=[cat_sz]), 'y': tensor_infos['y']} for cat_sz in x_cat_sizes]
        return [{'x_cat': TensorInfo(cat_sizes=[cat_sz])} for cat_sz in x_cat_sizes]

    def forward_tensor_infos(self, tensor_infos):
        x_cat_sizes = tensor_infos['x_cat'].get_cat_sizes().numpy()
        n_cont = tensor_infos[self.enc_output_name].get_n_features() \
            if self.enc_output_name in tensor_infos else 0
        out_cat_sizes = []
        for cat_sz, enc in zip(x_cat_sizes, self.single_encoder_fitters):
            ti = {'x_cat': TensorInfo(cat_sizes=[cat_sz])}
            out_ti = enc.forward_tensor_infos(ti)
            if 'x_cont' in out_ti:
                n_cont += out_ti['x_cont'].get_n_features()
            else:
                out_cat_sizes.append(out_ti['x_cat'].get_cat_sizes()[0].item())
        if len(out_cat_sizes) > 0:
            return utils.update_dict(tensor_infos, {self.enc_output_name: TensorInfo(feat_shape=[n_cont]),
                                                   'x_cat': TensorInfo(cat_sizes=out_cat_sizes)})
        else:
            return utils.update_dict(tensor_infos, {self.enc_output_name: TensorInfo(feat_shape=[n_cont])},
                                     remove_keys='x_cat')

    def _fit(self, ds: DictDataset) -> Layer:
        x_cat_sizes = ds.tensor_infos['x_cat'].get_cat_sizes().numpy()
        enc_layers = []
        for i in range(len(x_cat_sizes)):
            enc = self.single_encoder_fitters[i]
            if enc.needs_tensors:
                tensors = {'x_cat': ds.tensors['x_cat'][:, i:i+1]}
                if 'y' in ds.tensors:
                    tensors['y'] = ds.tensors['y']
            else:
                tensors = None
            tensor_infos = {'x_cat': TensorInfo(cat_sizes=[x_cat_sizes[i]])}
            if 'y' in ds.tensor_infos:
                tensor_infos['y'] = ds.tensor_infos['y']
            enc_layers.append(enc.fit(DictDataset(tensors, tensor_infos, ds.device, ds.n_samples)))

        return EncodingLayer(enc_layers, self.enc_output_name, self)

    # def split_off_dynamic(self):
    #     splits = [f.split_off_dynamic() for f in self.single_encoder_fitters]
    #     s0 = [s[0] for s in splits]
    #     s1 = [s[1] for s in splits]
    #     # todo


class EncodingFactory(FitterFactory):
    def __init__(self, single_encoder_factory, enc_output_name: str = 'x_cont'):
        super().__init__()
        self.single_encoder_factory = single_encoder_factory
        self.enc_output_name = enc_output_name

    def _create(self, tensor_infos):
        if 'x_cat' not in tensor_infos or tensor_infos['x_cat'].get_n_features() == 0:
            return IdentityFitter()

        x_cat_sizes = tensor_infos['x_cat'].get_cat_sizes().numpy()
        single_encoder_fitters = [self.single_encoder_factory.create({'x_cat': TensorInfo(cat_sizes=[cat_sz]),
                                                                      'y': tensor_infos['y']})
                                  for cat_sz in x_cat_sizes]
        return EncodingFitter(single_encoder_fitters, enc_output_name=self.enc_output_name)

# ----- One-Hot ------


class SingleOneHotLayer(Layer):
    def __init__(self, fitter: Fitter, onoff, cat_size, use_missing_zero: bool, use_1d_binary_onehot: bool):
        super().__init__(fitter=fitter)
        self.onoff = onoff
        self.cat_size = cat_size
        self.use_missing_zero = use_missing_zero
        self.use_1d_binary_onehot = use_1d_binary_onehot

    def _binary(self, x_cat, values):
        src = torch.as_tensor(values, dtype=torch.float32, device=x_cat.device)
        # add other dimensions to match those of x_cat
        src = src[[None] * (x_cat.dim()-1) + [slice(None)]].expand(*(list(x_cat.shape[:-1]) + [-1]))
        return src.gather(dim=-1, index=x_cat)

    def _multiple(self, x_cat, on_value, off_value):
        cont_shape = (*x_cat.shape[:-1], self.cat_size)
        cont = torch.full(cont_shape, off_value, dtype=torch.float32, device=x_cat.device)
        src = torch.full([1] * x_cat.dim(), on_value,
                         dtype=torch.float32, device=x_cat.device).expand(*x_cat.shape)
        cont.scatter_(dim=-1, index=x_cat, src=src)
        return cont

    def forward_tensors(self, tensors):
        x_cat = tensors['x_cat']
        # default_slices = [slice(None)] * (x_cat_sq.dim() - 1)
        on_value = self.onoff[0]
        off_value = self.onoff[1]

        if self.use_missing_zero:
            if self.cat_size == 2 and self.use_1d_binary_onehot:
                # should not be used with use_missing_zero anyway
                cont = self._binary(x_cat, [-on_value, on_value])
            elif self.cat_size == 3 and self.use_1d_binary_onehot:
                cont = self._binary(x_cat, [off_value, on_value, -on_value])
            else:
                cont = self._multiple(x_cat, on_value=on_value, off_value=off_value)
                # cont = cont[[slice(None)] * (x_cat.dim() - 1) + [slice(1, None)]]
                cont = cont[..., 1:]  # cut off the dimension for the missing value one-hot
        else:
            if self.cat_size == 2 and self.use_1d_binary_onehot:
                cont = self._binary(x_cat, [off_value, on_value])
            else:
                cont = self._multiple(x_cat, on_value=on_value, off_value=off_value)

        return utils.update_dict(tensors, {'x_cont': cont}, remove_keys='x_cat')


class SingleOneHotFitter(Fitter):
    def __init__(self, use_missing_zero: bool, bin_onoff: Tuple[float, float], multi_onoff: Tuple[float, float],
                 use_1d_binary_onehot: bool):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont', 'x_cat'])
        self.use_missing_zero = use_missing_zero
        self.bin_onoff = bin_onoff
        self.multi_onoff = multi_onoff
        self.use_1d_binary_onehot = use_1d_binary_onehot

    def forward_tensor_infos(self, tensor_infos):
        cat_size = tensor_infos['x_cat'].get_cat_sizes()[0].item()
        if self.use_missing_zero:
            cat_size -= 1
        if cat_size == 2 and self.use_1d_binary_onehot:
            cat_size = 1
        return utils.update_dict(tensor_infos, {'x_cont': TensorInfo(feat_shape=[cat_size])}, remove_keys='x_cat')

    def _fit(self, ds: DictDataset) -> Layer:
        cat_size = ds.tensor_infos['x_cat'].get_cat_sizes()[0].item()
        is_binary = cat_size - int(self.use_missing_zero) <= 2
        return SingleOneHotLayer(self, onoff=self.bin_onoff if is_binary else self.multi_onoff, cat_size=cat_size,
                                 use_missing_zero=self.use_missing_zero,
                                 use_1d_binary_onehot=self.use_1d_binary_onehot)


class SingleOneHotFactory(SingleEncodingFactory):
    def __init__(self, use_missing_zero=True, bin_onoff=(1.0, 0.0), multi_onoff=(1.0, 0.0), min_one_hot_cat_size=0,
                 max_one_hot_cat_size=-1, max_one_hot_size_by_n_classes=False, use_1d_binary_onehot: bool = True,
                 **config):
        super().__init__(create_fitter=lambda tensor_infos:
                                            SingleOneHotFitter(use_missing_zero=use_missing_zero,
                                                               bin_onoff=bin_onoff, multi_onoff=multi_onoff,
                                                               use_1d_binary_onehot=use_1d_binary_onehot),
                         min_cat_size=min_one_hot_cat_size, max_cat_size=max_one_hot_cat_size)
        self.max_one_hot_size_b_n_classes = max_one_hot_size_by_n_classes

    def apply_on(self, cat_size: int, n_classes: int):
        if self.max_one_hot_size_b_n_classes:
            return cat_size <= n_classes
        else:
            return super().apply_on(cat_size, n_classes)


# ------ Embedding --------


class SingleEmbeddingLayer(Layer):
    def __init__(self, emb: Variable):
        super().__init__(new_tensor_infos={'x_cont': TensorInfo(feat_shape=[emb.shape[-1]])}, remove_keys='x_cat')
        # emb.shape should be (parallel dims) x cat_size x emb_size
        # print(f'{emb.numel()=}')
        self.emb = emb

    def forward_tensors(self, tensors):
        x_cat = tensors['x_cat']
        # print(f'{x_cat.shape=}')
        x_cat = x_cat.squeeze(-1)  # squeeze feature dimension, we assume that there is only one feature
        parallel_dims = self.emb.dim() - 2  # subtract category and feature dimension

        # idxs = []
        # for dim in range(parallel_dims):
        #     # todo: could cache these and not create them newly every time?
        #     view_shape = [1] * (parallel_dims+1)
        #     view_shape[dim] = self.emb.shape[dim]
        #     idxs.append(torch.arange(self.emb.shape[dim], dtype=torch.long, device=self.emb.device).view(*view_shape))
        # idxs.append(x_cat)
        # x_cont = self.emb[idxs]

        # code using index_select which is faster than fancy indexing
        # put all parallel dimensions into the batch dimension
        cat_size = self.emb.shape[-2]
        n_flattened_idxs = cat_size
        n_batch = x_cat.shape[-1]
        # shape: (n_parallel * cat_size) x n_features
        emb_flat = self.emb.reshape(-1, self.emb.shape[-1])
        while x_cat.dim() > 1:
            # merge batch dimension with all parallel dimensions
            n_parallel = x_cat.shape[-2]
            parallel_idxs = torch.arange(x_cat.shape[-2], dtype=torch.long, device=self.emb.device)
            # add offsets to parallel dimension
            x_cat = x_cat + n_flattened_idxs * parallel_idxs[:, None]
            # merge parallel and batch dimension
            x_cat = x_cat.reshape(*x_cat.shape[:-2], -1)
            # now the indexes span a larger range
            n_flattened_idxs *= n_parallel
        # for dim in range(parallel_dims):
        #     # todo:
        #     pass
        # print(f'{x_cat.shape=}, {emb_flat.shape=}, {n_flattened_idxs=}, {x_cat.max().item()=}')
        x_cont = emb_flat.index_select(0, x_cat)
        x_cont = x_cont.reshape(*self.emb.shape[:-2], n_batch, self.emb.shape[-1])

        # print(f'{torch.norm(x_cont)=}, {torch.norm(x_cont-x_cont_other)=}')


        return utils.update_dict(tensors, {'x_cont': x_cont}, remove_keys='x_cat')

    def _stack(self, layers: List['SingleEmbeddingLayer']):
        return SingleEmbeddingLayer(Variable.stack([layer.emb for layer in layers]))


def fastai_emb_size_fn(n_cat: int):
    return min(600, round(1.6 * n_cat ** 0.56))


class ConstantFunction:
    def __init__(self, value: Any):
        self.value = value

    def __call__(self, *args, **kwargs) -> Any:
        return self.value


def get_embedding_size(fn: Optional[Union[int, str, Callable[[int], int]]]) -> Callable[[int], int]:
    if fn is None:
        fn = 'fastai'

    if isinstance(fn, int):
        return ConstantFunction(value=fn)
    elif isinstance(fn, str):
        if fn == 'howard' or fn == 'fastai':
            # heuristic by Jeremy Howard in fastai
            return fastai_emb_size_fn
        else:
            raise ValueError(f'Unknown embedding_size name "{fn}"')
    else:
        return fn


class SingleEmbeddingFitter(Fitter):
    def __init__(self, embedding_size=None, **config):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont', 'x_cat'])
        # default option is taken from fastai2
        self.size_func = get_embedding_size(embedding_size) if embedding_size is not None \
            else fastai_emb_size_fn
        self.emb_init_mode = config.get('emb_init_mode', 'normal')
        self.emb_init_gain = config.get('emb_init_gain', 1.0)
        self.emb_reduce_norm = config.get('emb_reduce_norm', False)
        self.emb_lr_factor = config.get('emb_lr_factor', 1.0)

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        cat_sz = tensor_infos['x_cat'].get_cat_sizes()[0].item()
        return cat_sz * self.size_func(cat_sz)

    def forward_tensor_infos(self, tensor_infos):
        new_info = TensorInfo(feat_shape=[self.size_func(tensor_infos['x_cat'].get_cat_sizes()[0].item())])
        return utils.update_dict(tensor_infos, {'x_cont': new_info}, remove_keys='x_cat')

    def _fit(self, ds: DictDataset) -> Layer:
        cat_size = ds.tensor_infos['x_cat'].get_cat_sizes()[0].item()
        emb_size = self.size_func(cat_size)
        if self.emb_init_mode == 'normal':
            emb = torch.randn(cat_size, emb_size, device=ds.device)
        elif self.emb_init_mode == 'uniform':
            emb = 2*torch.rand(cat_size, emb_size, device=ds.device) - 1
        elif self.emb_init_mode == 'kaiming-uniform-t':
            # as in the RTDL nets, use 1/sqrt(out_features)
            emb = (1./np.sqrt(emb_size)) * (2 * torch.rand(cat_size, emb_size, device=ds.device) - 1)
            emb[0, :] = 0.0  # set unknown/missing category to 0
        else:
            raise ValueError(f'Unknown emb_init_mode: {self.emb_init_mode}')
        # todo: should emb_reduce_norm be used differently as for NTK param (Adam vs not Adam)?
        emb_factor = self.emb_init_gain * (np.sqrt(1.0/emb_size) if self.emb_reduce_norm else 1.0)
        return SingleEmbeddingLayer(Variable(emb_factor * emb, trainable=True,
                                             hyper_factors={'lr': self.emb_lr_factor}))


class SingleEmbeddingFactory(SingleEncodingFactory):
    def __init__(self, embedding_size=None, min_embedding_cat_size=0, max_embedding_cat_size=-1, **config):
        super().__init__(create_fitter=lambda tensor_infos:
                                            SingleEmbeddingFitter(embedding_size=embedding_size, **config),
                         min_cat_size=min_embedding_cat_size, max_cat_size=max_embedding_cat_size)

# ------- Target Encoding (a kind of fixed embedding) -------


class SingleTargetEncodingFitter(Fitter):
    def __init__(self, n_classes, **config):
        super().__init__(is_individual=False, modified_tensors=['x_cont', 'x_cat'])
        self.n_classes = n_classes

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        n_classes = tensor_infos['y'].get_cat_sizes()[0].item()
        emb_sz = 1 if n_classes <= 2 else n_classes
        cat_sz = tensor_infos['x_cat'].get_cat_sizes()[0].item()
        return emb_sz * cat_sz

    def forward_tensor_infos(self, tensor_infos):
        new_info = TensorInfo(feat_shape=[1 if self.n_classes <= 2 else self.n_classes])
        return utils.update_dict(tensor_infos, {'x_cont': new_info}, remove_keys='x_cat')

    def _fit(self, ds: DictDataset) -> Layer:
        x_cat = ds.tensors['x_cat'].squeeze(-1)
        x_cat_size = ds.tensor_infos['x_cat'].get_cat_sizes()[0].item()
        y = ds.tensors['y']
        y_cat_sizes = ds.tensor_infos['y'].get_cat_sizes().numpy()
        if y_cat_sizes[0] > 2:
            # multi-class classification
            y = F.one_hot(y[:, 0], num_classes=y_cat_sizes[0]).float()
        elif y_cat_sizes[0] == 2:
            # binary classification
            y = y.float()  # convert int to float

        prior = y.mean(dim=-2)  # mean over batch dimension

        sums = torch.zeros(x_cat_size, y.shape[-1], device=y.device)
        # In the following, scatter_add_ executes sums[x_cat[:, i][j], k] += y[j, k]
        # see also https://discuss.pytorch.org/t/pytorch-equivalent-to-tf-unsorted-segment-sum/25275/5
        sums.scatter_add_(0, x_cat[:,None].expand(-1, y.shape[-1]), y)
        frequencies = torch.bincount(x_cat, minlength=x_cat_size)
        # could also give the prior a different weight, this is just an option
        emb = (sums + prior[None, :]) / (frequencies[:, None] + 1)
        return SingleEmbeddingLayer(Variable(emb, trainable=False))


class SingleTargetEncodingFactory(SingleEncodingFactory):
    def __init__(self, min_targetenc_cat_size=0, max_targetenc_cat_size=-1, **config):
        create_fitter = lambda tensor_infos: SingleTargetEncodingFitter(n_classes=tensor_infos['y'].get_cat_sizes()[0].item())
        super().__init__(create_fitter=create_fitter, min_cat_size=min_targetenc_cat_size,
                         max_cat_size=max_targetenc_cat_size)


# ------- Label Encoding -------


class SingleOrdinalEncodingLayer(Layer):
    def __init__(self, fitter, cat_size: int, permute_ordinal_encoding: bool = False):
        super().__init__(fitter=fitter)
        self.cat_size = cat_size
        self.permute_ordinal_encoding = permute_ordinal_encoding
        self.perm = None
        if permute_ordinal_encoding:
            self.perm = Variable(torch.randperm(cat_size, dtype=torch.long), trainable=False)

    def forward_tensors(self, tensors):
        x_cat = tensors['x_cat']
        if self.permute_ordinal_encoding:
            x_cat = self.perm[x_cat]
        return utils.update_dict(tensors, {'x_cont': x_cat.type(torch.float32)}, remove_keys='x_cat')


class SingleOrdinalEncodingFitter(Fitter):
    def __init__(self, permute_ordinal_encoding: bool = False, **config):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont', 'x_cat'])
        self.permute_ordinal_encoding = permute_ordinal_encoding

    def forward_tensor_infos(self, tensor_infos):
        return utils.update_dict(tensor_infos, {'x_cont': tensor_infos['x_cat']}, remove_keys='x_cat')

    def _fit(self, ds: DictDataset) -> Layer:
        return SingleOrdinalEncodingLayer(self, cat_size=ds.tensor_infos['x_cat'].get_cat_sizes()[0].item(),
                                        permute_ordinal_encoding=self.permute_ordinal_encoding)


class SingleOrdinalEncodingFactory(SingleEncodingFactory):
    def __init__(self, min_labelenc_cat_size=0, max_labelenc_cat_size=-1, **config):
        super().__init__(create_fitter=lambda tensor_infos: SingleOrdinalEncodingFitter(**config),
                         min_cat_size=min_labelenc_cat_size, max_cat_size=max_labelenc_cat_size)

