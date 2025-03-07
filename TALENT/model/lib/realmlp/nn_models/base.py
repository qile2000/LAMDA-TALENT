from TALENT.model.lib.realmlp import torch_utils, utils
from TALENT.model.lib.realmlp.data.data import TensorInfo, DictDataset
from TALENT.model.lib.realmlp.training.coord import HyperparamManager
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._C import _disabled_torch_function_impl
import numpy as np
import threading
import re
import copy
from contextlib import contextmanager
from typing import Optional, List, Union, Dict, Tuple

# have a layer that allows to split/merge DictDatasets?
# need something like numerical_preprocess
# could specify a input to output mapping, e.g. {'x_cont': None, 'x_cat': 'x_cont'}, which could also allow to merge
# or just have a ParallelFitter that merges outputs, with a tensor subselection beforehand
# e.g.
# num_pipeline = SequentialFactory([FilterFactory('x_cont'), PreprocessingFactory(), NumericalEmbeddingFactory()])
# cat_pipeline = SequentialFactory([FilterFactory('x_cat', 'y'), OneHotFactory(), PreprocessingFactory(), EmbeddingFactory()]
# pipeline = ConcatFactory([num_pipeline, cat_pipeline])

# theoretically, could allow to split off fitters by max RAM usage / max num features depending on size of dataset
# then, small datasets could be preprocessed in advance even with heavy parallelization
# but this would require a parallelized version of DictDataset...

# fitter.fit() should pass context to Variable - how? After returning!
# But pass scope into fit already, otherwise parent scope will not be known
# fit() should then take scope, hp_manager - how to pass that on to sub-fitters? Or have nn.ModuleList-like system?
# in the latter case, could have a set_context() function - but then would need to ensure that this is called...
# problem: setting later to layer needs to be done still before layer is called - if fit_transform is implemented,
# this will not work
# use context manager instead?
# use self.create_variable()?  (could also easily be forgotten)
# what about if each layer takes a fitter?
# layer could also forget to pass scope to variable
# should a variable have hyper_getter itself instead of having it in the optimizer?

# Fitter constructor should have an attribute scope_name or so

# use context managers at factory creation, then replicate context manager in create() and fit()
# and implement fit_impl() and create_impl() then grab context in Layer() and Variable() constructors?
# in order to let factory set its own context as well (e.g. weight), include constructor parameter?
# need separate context for HyperparamManager around fit()?
# can we have a thread-local context?
# can those contexts also be used to select configs? (like for first_layer_config etc.)

# could use linux-like scope /first_layer/block/weight or /pipeline/1/robust_scale and then filter it using regexes

# run in problems with stack() and register_hypers() twice? or no problem because of new naming convention?
# or should stack() not call register_hypers() again but use a list of getters?
# (that would be good for having different hypers for different parallel layers,
# but bad for dropout implementation and maybe speed)

# could have a simplify() function in Fitter to remove Identity layers and empty SequentialLayers recursively
# then could maybe save the IdentityLayer check in SequentialLayer

# todo: does multiple inheritance from Fitter and FitterFactory work with contexts?


class Scope:
    def __init__(self, names: Optional[List[str]] = None):
        self._names = names or []

    def get_sub_scope(self, name: str) -> 'Scope':
        return Scope(self._names + [name])

    def __str__(self):
        return '/' + '/'.join(self._names)

    def matches(self, regex: Union[str, re.Pattern]) -> bool:
        if isinstance(regex, str):
            regex = re.compile(regex)
        return bool(regex.match(str(self)))


class TrainContext:
    # see https://stackoverflow.com/questions/51849395/how-can-we-associate-a-python-context-manager-to-the-variables-appearing-in-it
    _data = threading.local()

    def __init__(self, scope: Optional[Scope] = None, hp_manager: Optional[HyperparamManager] = None):
        self.scope = scope or Scope()
        self.hp_manager = hp_manager

    def clone(self):
        return TrainContext(copy.deepcopy(self.scope), self.hp_manager)

    @staticmethod
    def get_global_context() -> 'TrainContext':
        if not hasattr(TrainContext._data, 'context'):
            TrainContext._data.context = TrainContext()
        return TrainContext._data.context


@contextmanager
def sub_scope_context(name: str):
    current_context = TrainContext.get_global_context()
    old_scope = current_context.scope
    current_context.scope = old_scope.get_sub_scope(name)
    yield
    current_context.scope = old_scope


@contextmanager
def sub_scopes_context(names: List[str]):
    current_context = TrainContext.get_global_context()
    old_scope = current_context.scope
    new_scope = old_scope
    for name in names:
        new_scope = new_scope.get_sub_scope(name)
    current_context.scope = new_scope
    yield
    current_context.scope = old_scope


@contextmanager
def set_scope_context(scope: Scope):
    current_context = TrainContext.get_global_context()
    old_scope = current_context.scope
    current_context.scope = scope
    yield
    current_context.scope = old_scope


@contextmanager
def set_hp_context(hp_manager: Optional[HyperparamManager]):
    current_context = TrainContext.get_global_context()
    old_hp_manager = current_context.hp_manager
    if hp_manager:
        current_context.hp_manager = hp_manager
    yield
    current_context.hp_manager = old_hp_manager


class ContextAware:
    def __init__(self, scope_names: Optional[List[str]] = None):
        super().__init__()  # needed in case of multiple inheritance from ContextAware and another base class
        self.scope_names = scope_names or []

    def add_scope(self, name: str):
        self.scope_names.append(name)
        return self

    def add_others_scope(self, other: 'ContextAware'):
        self.scope_names.extend(other.scope_names)
        return self

    @contextmanager
    def set_context(self):
        with sub_scopes_context(self.scope_names):
            yield


class ContextRecorder:
    def __init__(self):
        super().__init__()   # needed in case of multiple inheritance from ContextRecorder and another base class
        self.context = TrainContext.get_global_context().clone()

    @contextmanager
    def set_context(self):
        with set_scope_context(self.context.scope):
            with set_hp_context(self.context.hp_manager):
                yield


class StringConvertible:
    def __init__(self):
        super().__init__()  # for multiple inheritance

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + '(' \
               + ', '.join([f'{key} = {value}' for key, value in self.__dict__.items()]) + ')'


class Variable(ContextRecorder, nn.Parameter):
    def __new__(cls, data=None, trainable=True, requires_grad=None, hyper_factors=None):
        if data is None:
            data = torch.Tensor()
        if requires_grad is None:
            requires_grad = trainable
        obj = super().__new__(cls, data, requires_grad)
        obj.hyper_factors = hyper_factors or dict()
        obj.trainable = trainable
        return obj

    def __init__(self, data=None, trainable=True, requires_grad=None, hyper_factors=None):
        super().__init__()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.trainable,
                                self.requires_grad, self.hyper_factors)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return f'Variable(trainable={self.trainable}) containing:\n' + super(Variable, self).__repr__()

    __torch_function__ = _disabled_torch_function_impl

    @staticmethod
    def stack(vars: List['Variable'], dim=0):
        # vars must not be an empty list
        # todo: could make hyper_factors stackable
        with vars[0].set_context():
            with torch.no_grad():
                return Variable(torch.stack(vars, dim=dim), trainable=vars[0].trainable,
                                requires_grad=vars[0].requires_grad, hyper_factors=vars[0].hyper_factors)


# ------- Layers -------


class Layer(ContextRecorder, StringConvertible, nn.Module):
    """
    Extended version of nn.Module, allowing vectorization, processing data sets with multiple tensors,
    using Variable instead of Parameter, ...

    The following methods need to be overridden:
    - forward_tensor_infos (but if the output is constant, we can just set new_tensor_infos in the constructor)
    - forward_tensor or forward_cont (the latter if only x_cont is changed)
    - _stack()
    - optionally __repr__() and __str__()
    """

    def __init__(self, new_tensor_infos: Optional[Dict[str, TensorInfo]] = None,
                 fitter: Optional['Fitter'] = None, remove_keys: Optional[Union[str, List[str]]] = None):
        """
        Constructor. Puts the layer in eval mode, since it might be used inside the fit_transform() of the Fitter.
        The parameters provide different opportunities
        to specify a default implementation for forward_tensor_infos().
        The default implementation is:
        ```
            if self.fitter is not None:
                return self.fitter.forward_tensor_infos(tensor_infos)
            return utils.update_dict(tensor_infos, self.new_tensor_infos, remove_keys=self.remove_keys)
        ```
        """
        super().__init__()
        self.new_tensor_infos = {} if new_tensor_infos is None else new_tensor_infos
        self.remove_keys = remove_keys
        self.fitter = fitter
        self.hp_manager = None
        # don't put in eval mode, so we have realistic behavior during fit_transform()
        self.eval()  # todo: remove

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        """
        Override this method if the information from constructor is not sufficient.
        :param tensor_infos: Tensor infos (shapes etc.)
        :return: Transformed tensor infos.
        """
        if self.fitter is not None:
            return self.fitter.forward_tensor_infos(tensor_infos)
        return utils.update_dict(tensor_infos, self.new_tensor_infos, remove_keys=self.remove_keys)

    def forward(self, data: Union[DictDataset, Dict[str, torch.Tensor]]) -> Union[DictDataset, Dict[str, torch.Tensor]]:
        """
        This is an implementation of the nn.Module forward() function, which is called by __call__().
        Don't override this method.
        :param data: data set or dict of tensors.
        :return: Transformed version of the data set or dict of tensors.
        """
        if isinstance(data, DictDataset):
            return self.forward_ds(data)
        else:
            return self.forward_tensors(data)

    def forward_ds(self, ds: DictDataset) -> DictDataset:
        # default implementation
        return DictDataset(None if ds.tensors is None else self.forward_tensors(ds.tensors),
                           self.forward_tensor_infos(ds.tensor_infos), device=ds.device, n_samples=ds.n_samples)

    def forward_cont(self, x: torch.Tensor) -> torch.Tensor:
        # only needs to be overridden if the default implementation of forward_tensors() is used
        # we check this to avoid infinite recursion if forward_tensors() is not overridden
        if self.__class__.forward_tensors != Layer.forward_tensors:
            return self.forward_tensors({'x_cont': x})['x_cont']
        raise NotImplementedError()

    def forward_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transforms the given tensors.
        :param tensors:
        :return:
        """
        # default implementation just updates x_cont using self.forward_cont()
        # print(f'{self.__class__.__name__}: {tensors.keys()=}')
        return utils.join_dicts(tensors, {'x_cont': self.forward_cont(tensors['x_cont'])})

    def _stack(self, layers: List['Layer']) -> 'Layer':
        """
        Implementation of stack(). Can be overridden.
        Vectorizes the given layers. The given layers should all have the same structure.
        If layers[0] has no parameters (trainable or buffer), then the default implementation simply returns layers[0].
        Override if another implementation is desired.
        :param layers: Layers that should be stacked for vectorization.
        :return: Returns the stacked Layer object
        """
        # this needs to be overridden by some classes
        if len(list(layers[0].state_dict())) == 0:
            # no parameters, can simply vectorize by taking the first layer
            return layers[0]
        else:
            raise NotImplementedError()

    def stack(self, layers: List['Layer']) -> 'Layer':
        """
        Vectorizes the given layers. The given layers should all have the same structure.
        Do not override this method, override _stack() instead.
        :param layers: Layers that should be stacked for vectorization.
        :return: Returns the stacked Layer object
        """
        with self.set_context():
            return self._stack(layers)

    def __setattr__(self, name, value):
        # adapted from nn.Module.__setattr__
        # first checks whether the value is a Variable, otherwise uses nn.Module.__setattr__
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)
        if isinstance(value, Variable):
            if value.trainable:
                if self.__dict__.get('_parameters') is None:
                    raise AttributeError(
                        "cannot assign parameters before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._modules,
                            self._non_persistent_buffers_set)
                self.register_parameter(name, value)
            else:
                if self.__dict__.get('_buffers') is None:
                    raise AttributeError(
                        "cannot assign parameters before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._modules,
                            self._non_persistent_buffers_set)
                self.register_buffer(name, value)
        else:
            super(Layer, self).__setattr__(name, value)


class IdentityLayer(Layer):
    # Attention: do not inherit from IdentityLayer since this might mess with optimizations in SequentialLayer!
    def forward_tensors(self, x):
        return x


class SequentialLayer(Layer):
    def __init__(self, tfms: List[Layer]):
        super().__init__()
        self.tfms = nn.ModuleList([tfm for tfm in tfms if not isinstance(tfm, IdentityLayer)])

    def forward_tensor_infos(self, tensor_infos):
        for tfm in self.tfms:
            tensor_infos = tfm.forward_tensor_infos(tensor_infos)
        return tensor_infos

    def forward_ds(self, ds: DictDataset):
        for tfm in self.tfms:
            ds = tfm.forward_ds(ds)
        return ds

    def forward_tensors(self, tensors):
        for tfm in self.tfms:
            tensors = tfm(tensors)
        return tensors

    def _stack(self, seq_tfms):
        return SequentialLayer([seq_tfms[0].tfms[i].stack([seq_tfm.tfms[i] for seq_tfm in seq_tfms])
                                for i in range(len(seq_tfms[0].tfms))])

    def __repr__(self):
        return str(self)

    def __str__(self):
        sub_strings = ['  ' + line for tfm in self.tfms for line in str(tfm).split('\n')]
        return f'{self.__class__.__name__} [\n' + '\n'.join(sub_strings) + '\n]\n'


class ResidualLayer(Layer):
    def __init__(self, inner_layer: Layer):
        super().__init__()
        self.inner_layer = inner_layer

    def forward_tensor_infos(self, tensor_infos):
        return self.inner_layer.forward_tensor_infos(tensor_infos)

    def forward_tensors(self, tensors: Dict[str, torch.Tensor]):
        new_tensors = self.inner_layer.forward_tensors(tensors)
        new_tensors['x_cont'] = tensors['x_cont'] + new_tensors['x_cont']
        return new_tensors

    def _stack(self, seq_tfms):
        return ResidualLayer(seq_tfms[0].inner_layer.stack([seq_tfm.inner_layer for seq_tfm in seq_tfms]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        sub_strings = ['  ' + line for line in str(self.inner_layer).split('\n')]
        return f'ResidualLayer [\n' + '\n'.join(sub_strings) + '\n]\n'


class ConcatParallelLayer(Layer):
    """
    Executes all layers on the given input
    and combines the resulting output tensors by concatenating along the last dimension (as in DenseNet, for example).
    Not all layers need to output the same tensors, e.g.,
    one can output only 'x_cont' and the other can output 'x_cont' and 'y',
    in which case 'y' will not be concatenated with another tensor.
    """
    def __init__(self, layers: List[Layer], fitter: 'Fitter'):
        super().__init__(fitter=fitter)
        self.layers = nn.ModuleList(layers)

    def forward_tensors(self, tensors):
        out_tensors = [layer.forward_tensors(tensors) for layer in self.layers]
        out_keys = {key for t in out_tensors for key in t.keys()}
        # print(f'{[t["x_cont"].shape for t in out_tensors]=}')
        return {key: torch_utils.cat_if_necessary([t[key] for t in out_tensors if key in t], dim=-1)
                for key in out_keys}

    def _stack(self, tfms: List[Layer]):
        return ConcatParallelLayer([tfms[0].layers[i].stack([tfm.layers[i] for tfm in tfms])
                                    for i in range(len(tfms[0].layers))], fitter=tfms[0].fitter)

    def __repr__(self):
        return str(self)

    def __str__(self):
        sub_strings = ['  ' + line for tfm in self.layers for line in str(tfm).split('\n')]
        return f'{self.__class__.__name__} [\n' + '\n'.join(sub_strings) + '\n]\n'


class FilterTensorsLayer(Layer):
    """
    Only returns those tensors whose name is in a list of names
    """
    def __init__(self, include_keys: Optional[List[str]], exclude_keys: Optional[List[str]], fitter: 'Fitter'):
        """
        :param keys: List of tensor names that is allowed to pass through
        """
        super().__init__(fitter=fitter)
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys

    def forward_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # return {key: value for key, value in tensors.items() if key in self.keys}
        result = {key: (value if (self.include_keys is None or key in self.include_keys) and
                                 (self.exclude_keys is None or key not in self.exclude_keys)
                        else value[..., :0])
                for key, value in tensors.items()}
        # print(result)
        return result


class FunctionLayer(Layer):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward_cont(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class BiasLayer(Layer):
    def __init__(self, bias: Variable, factor: float = 1.0):
        super().__init__()
        self.bias = bias
        self.factor = factor

    def forward_cont(self, x):
        if self.factor != 1.0:
            x = x + self.factor * self.bias
        else:
            x = x + self.bias
        return x

    def _stack(self, tfms):
        return BiasLayer(Variable.stack([tfm.bias for tfm in tfms]), factor=tfms[0].factor)


class ScaleLayer(Layer):
    def __init__(self, scale: Variable):
        super().__init__()
        self.scale = scale

    def forward_cont(self, x):
        # print(f'{x.norm().item()=:g}, {self.scale.norm().item()=:g}')
        return x * self.scale

    def _stack(self, tfms):
        return ScaleLayer(Variable.stack([tfm.scale for tfm in tfms]))


class WeightLayer(Layer):
    def __init__(self, weight: Variable, factor: float = 1.0):
        super().__init__(new_tensor_infos={'x_cont': TensorInfo(feat_shape=[weight.shape[-1]])})
        # weight should be <batch-dims> x in_features x out_features unlike in nn.Linear
        self.weight = weight
        self.factor = factor

    def forward_cont(self, x):
        x = x.matmul(self.weight)
        if self.factor != 1.0:
            x = self.factor * x
        return x

    def _stack(self, tfms):
        return WeightLayer(Variable.stack([tfm.weight for tfm in tfms]), factor=tfms[0].factor)


class RenameTensorLayer(Layer):
    def __init__(self, old_name: str, new_name: str, fitter: 'Fitter'):
        super().__init__(fitter=fitter)
        self.old_name = old_name
        self.new_name = new_name

    def forward_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.old_name not in tensors:
            return tensors
        elif self.new_name not in tensors:
            return utils.update_dict(tensors, {self.new_name: tensors[self.old_name]}, remove_keys=self.old_name)
        else:
            # print(f'{tensors[self.new_name].shape=}, {tensors[self.old_name].shape=}')
            new_tensor = torch.cat([tensors[self.new_name], tensors[self.old_name]], dim=-1)
            return utils.update_dict(tensors, {self.new_name: new_tensor}, remove_keys=self.old_name)

    def _stack(self, layers: List['Layer']) -> 'Layer':
        return layers[0]


# ------ Fitters ------


class Fitter(ContextAware, StringConvertible):
    """
    Fitters produce Layer objects given a data set (of inputs to the fitter at initialization)
    """

    def __init__(self, needs_tensors: bool = True, is_individual: bool = True, scope_names: Optional[List[str]] = None,
                 modified_tensors: Optional[List[str]] = None):
        """
        :param needs_tensors: Set to true if the fitter needs to have the tensors in fit() or fit_transform().
        If false, then in fit(ds) or fit_transform(ds), ds.tensors is allowed to be None.
        :param is_individual: Set to false if fit(ds) deterministically produces a non-trainable layer.
        (In this case, this Fitter only needs to be called once in k-fold CV on the train+val set.)
        :param scope_names: List of names to add to the scope
        (will be present in the names of Variables constructed in this Fitter)
        :param modified_tensors: List of names of tensors that are modified by this Fitter, e.g., ['x_cont'].
        This is used for the default implementation of get_n_forward(),
        which is used to get a RAM estimate for the forward pass.
        The default RAM estimate is simply the size of all modified tensors.
        """
        super().__init__(scope_names=scope_names)
        # needs_data=False specifies that in fit(ds), ds.tensors is allowed to be None
        # is_individual=False specifies that fit(ds) deterministically produces a non-trainable layer
        self.needs_tensors = needs_tensors
        self.is_individual = is_individual
        self.modified_tensors = modified_tensors

    def _get_n_values(self, tensor_infos: Dict[str, TensorInfo], relevant_tensors: Optional[List[str]]):
        """
        Helper function that can be used internally to get the number of elements of a list of tensors.
        Should not be overridden.
        :param tensor_infos: Tensor infos of the data set
        :param relevant_tensors: List of tensor names that should be considered. If None, 0 is returned.
        :return: Returns the number of components of a list of tensors (per batch element).
        """
        if relevant_tensors is None:
            return 0
        return sum([ti.get_n_features() for key, ti in tensor_infos.items() if key in relevant_tensors])

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        """
        Should be overridden if the fitter produces layers with trainable parameters.
        :param tensor_infos: Tensor infos.
        :return: Returns the number of parameters of the fitted layer for the given tensor_infos.
        """
        return 0

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        """
        Should be overridden if the fitter does more than just one operation.
        :param tensor_infos: Ingoing tensor infos.
        :return: Should return the number of bytes used in the forward pass per batch element
        """
        if self.modified_tensors is None:
            return 0
        return self._get_n_values(self.forward_tensor_infos(tensor_infos), self.modified_tensors)

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        """
        Should be overridden if the fitter changes the tensor shapes.
        :param tensor_infos: Tensor infos (for shapes and category sizes).
        :return: Transformed tensor infos.
        """
        return tensor_infos  # should be overridden by subclasses if tensor_infos change

    def fit(self, ds: DictDataset) -> Layer:
        """
        Produces a layer initialized based on a given data set.
        This method should not be overridden, override _fit() instead.
        :param ds: Data set.
        :return: Layer object.
        """
        with self.set_context():
            return self._fit(ds)

    def fit_transform(self, ds: DictDataset, needs_tensors: bool = True) -> Tuple[Layer, DictDataset]:
        """
        Produces a layer initialized based on a given data set.
        This method should not be overridden, override _fit_transform() instead.
        :param ds: Data set.
        :param needs_tensors: Whether the transformed data set should also contain transformed tensors
         (compared to only transformed tensor_infos).
        :return: Layer object and the data set transformed by the Layer.
        """
        with self.set_context():
            return self._fit_transform(ds, needs_tensors)

    def fit_transform_subsample(self, ds: DictDataset, ram_limit_gb: float, needs_tensors: bool = True) \
            -> Tuple[Layer, DictDataset]:
        """
        Similar to fit_transform(), but may subsample the data set in order to stay within a given RAM limit.
        This method should not be overridden, override _fit_transform_subsample() instead.
        :param ds: Data set.
        :param ram_limit_gb: RAM limit in GB.
        :param needs_tensors: Whether the transformed tensors should be output.
        :return: Tuple of the resulting Layer and the transformed DictDataset.
        """
        with self.set_context():
            return self._fit_transform_subsample(ds, ram_limit_gb, needs_tensors)

    def _fit(self, ds: DictDataset) -> Layer:
        """
        Implementation of fit(). At least one of _fit() or _fit_transform() should be overridden by subclasses.
        :param ds: Data set.
        :return: Initialized Layer object.
        """
        if self.__class__._fit_transform != Fitter._fit_transform:
            # avoid infinite recursion if the method is not overridden
            tfm, ds = self._fit_transform(ds, False)
            return tfm
        elif self.__class__._fit_transform_subsample != Fitter._fit_transform_subsample:
            # avoid infinite recursion if the method is not overridden
            tfm, ds = self._fit_transform_subsample(ds, ram_limit_gb=np.Inf, needs_tensors=False)
            return tfm
        if isinstance(self, Layer):
            return self
        raise NotImplementedError()

    def _fit_transform(self, ds: DictDataset, needs_tensors: bool) -> Tuple[Layer, DictDataset]:
        """
        Implementation of fit_transform(). At least one of _fit() or _fit_transform()
        should be overridden by subclasses.
        :param ds: Data set.
        :param needs_tensors: Whether the transformed data set should also contain transformed tensors
         (compared to only transformed tensor_infos).
        :return: Initialized Layer object and transformed data set
        """
        if self.__class__._fit_transform_subsample != Fitter._fit_transform_subsample:
            return self._fit_transform_subsample(ds, ram_limit_gb=np.Inf, needs_tensors=needs_tensors)
        else:
            tfm = self._fit(ds)
            if needs_tensors:
                return tfm, tfm.forward_ds(ds)
            else:
                return tfm, DictDataset(None, tfm.forward_tensor_infos(ds.tensor_infos), ds.device, ds.n_samples)

    def _fit_transform_subsample(self, ds: DictDataset, ram_limit_gb: float, needs_tensors: bool = True) \
            -> Tuple[Layer, DictDataset]:
        n_forward = self.get_n_forward(ds.tensor_infos)

        # check if subsampling is necessary
        if ram_limit_gb < np.Inf and n_forward > 0 and ds.tensors is not None and (self.needs_tensors or needs_tensors):
            # optimistically assume 4 bytes per number, while 8 are needed for categorical values
            max_n_samples = max(1, int(ram_limit_gb * (1024 ** 3) / (4 * n_forward)))
            if max_n_samples < ds.n_samples:
                # subsample the data set
                subsample_idxs = torch.randperm(ds.n_samples, device=ds.device)[:max_n_samples]
                ds = ds.get_sub_dataset(subsample_idxs)

        return self._fit_transform(ds, needs_tensors)

    def split_off_dynamic(self) -> Tuple['Fitter', 'Fitter']:
        """
        Can be overridden by subclasses if a trivial split based on
        self.needs_tensors and self.is_individual is not desired.
        :return: Returns a tuple of a static and a dynamic transform
        such that self is equivalent to SequentialFitter([static, dynamic])
        and such that the static transform does not need data and is not trainable.
        The idea is that in the vectorized setting, the static transform only needs to be applied once to the data set,
        while the dynamic transform needs to be applied separately for each of the vectorized models.
        """
        if self.needs_tensors or self.is_individual:
            return IdentityFitter(), self
        else:
            return self, IdentityFitter()

    def split_off_individual(self):
        """
        Can be overridden by subclasses if a trivial split based on self.is_individual is not desired.
        :return: Returns a tuple of a non-individual and an individual transform
        such that self is equivalent to SequentialFitter([non_individual, individual])
        and such that the non_individual transform deterministically produces a non-trainable layer.
        The idea is that the non-individual transform only needs to be applied once in k-fold cross-validation.
        """
        if self.is_individual:
            return IdentityFitter(), self
        else:
            return self, IdentityFitter()


class IdentityFitter(Fitter):
    def __init__(self, **config):
        super().__init__(needs_tensors=False, is_individual=False)

    def _fit(self, ds: DictDataset) -> Layer:
        return IdentityLayer()


class SequentialFitter(Fitter):
    def __init__(self, fitters: List[Fitter], **config):
        super().__init__(needs_tensors=np.any([f.needs_tensors for f in fitters]),
                         is_individual=np.any([f.is_individual for f in fitters]))
        self.fitters = fitters
        # print(f'Creating SequentialFitter with fitters {fitters} and {self.needs_tensors=}')

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]):
        for f in self.fitters:
            tensor_infos = f.forward_tensor_infos(tensor_infos)
        return tensor_infos

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]):
        n_params = 0
        for f in self.fitters:
            n_params += f.get_n_params(tensor_infos)
            tensor_infos = f.forward_tensor_infos(tensor_infos)
        return n_params

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]):
        forward_bytes = 0
        for f in self.fitters:
            forward_bytes += f.get_n_forward(tensor_infos)
            tensor_infos = f.forward_tensor_infos(tensor_infos)
        return forward_bytes

    def _fit_transform(self, ds: DictDataset, needs_tensors: bool = True):
        needs_tensors_list = [f.needs_tensors for f in self.fitters] + [needs_tensors]
        max_tensors_idx = np.max(np.argwhere(needs_tensors_list)) if np.any(needs_tensors_list) else 0
        tfms = []
        for i, fitter in enumerate(self.fitters):
            tfm, ds = fitter.fit_transform(ds, needs_tensors=(i < max_tensors_idx))
            tfms.append(tfm)
        return SequentialLayer(tfms), ds

    def _fit_transform_subsample(self, ds: DictDataset, ram_limit_gb: float, needs_tensors: bool = True) \
            -> Tuple[Layer, DictDataset]:
        needs_tensors_list = [f.needs_tensors for f in self.fitters] + [needs_tensors]
        max_tensors_idx = np.max(np.argwhere(needs_tensors_list)) if np.any(needs_tensors_list) else 0
        tfms = []
        for i, fitter in enumerate(self.fitters):
            tfm, ds = fitter.fit_transform_subsample(ds, ram_limit_gb=ram_limit_gb, needs_tensors=(i < max_tensors_idx))
            tfms.append(tfm)
        return SequentialLayer(tfms), ds

    def split_off_dynamic(self):
        is_dynamic = [f.needs_tensors or f.is_individual for f in self.fitters]
        if np.any(is_dynamic):
            first_dynamic = np.min(np.argwhere(is_dynamic))
            static, dynamic = self.fitters[first_dynamic].split_off_dynamic()
            return SequentialFitter(self.fitters[:first_dynamic] + [static]).add_others_scope(self), \
                   SequentialFitter([dynamic] + self.fitters[first_dynamic + 1:]).add_others_scope(self)
        else:
            return self, IdentityFitter()

    def split_off_individual(self):
        is_individual = [f.is_individual for f in self.fitters]
        if np.any(is_individual):
            first_indiv = np.min(np.argwhere(is_individual))
            non_indiv, indiv = self.fitters[first_indiv].split_off_individual()
            return SequentialFitter(self.fitters[:first_indiv] + [non_indiv]).add_others_scope(self), \
                   SequentialFitter([indiv] + self.fitters[first_indiv + 1:]).add_others_scope(self)
        else:
            return self, IdentityFitter()

    def __str__(self):
        sub_strings = ['  ' + line for fitter in self.fitters for line in str(fitter).split('\n')]
        return f'{self.__class__.__name__} [\n' + '\n'.join(sub_strings) + '\n]\n'


class ResidualFitter(Fitter):
    def __init__(self, inner_fitter: Fitter, **config):
        super().__init__(needs_tensors=inner_fitter.needs_tensors,
                         is_individual=inner_fitter.is_individual)
        self.inner_fitter = inner_fitter

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]):
        return self.inner_fitter.forward_tensor_infos(tensor_infos)

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]):
        return self.inner_fitter.get_n_params(tensor_infos)

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]):
        return self.inner_fitter.get_n_forward(tensor_infos) + self._get_n_values(tensor_infos, ['x_cont'])

    def _fit_transform(self, ds: DictDataset, needs_tensors=True):
        layer = ResidualLayer(self.inner_fitter.fit(ds))
        if needs_tensors:
            ds = layer.forward_ds(ds)
        return layer, ds

    def split_off_dynamic(self):
        if self.inner_fitter.needs_tensors or self.inner_fitter.is_individual:
            return IdentityFitter(), self
        else:
            return self, IdentityFitter()

    def split_off_individual(self):
        if self.inner_fitter.is_individual:
            return IdentityFitter(), self
        else:
            return self, IdentityFitter()

    def __str__(self):
        sub_strings = ['  ' + line for fitter in [self.inner_fitter] for line in str(fitter).split('\n')]
        return f'{self.__class__.__name__} [\n' + '\n'.join(sub_strings) + '\n]\n'


class FunctionFitter(Fitter):
    def __init__(self, f, **config):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont'])
        self.f = f

    def _fit(self, ds: DictDataset):
        return FunctionLayer(self.f)


class ConcatParallelFitter(Fitter):
    # todo: could implement better _fit_transform_subsample()
    def __init__(self, fitters: List[Fitter]):
        super().__init__(needs_tensors=np.any([f.needs_tensors for f in fitters]),
                         is_individual=np.any([f.is_individual for f in fitters]))
        self.fitters = fitters

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        out_tensor_infos = self.forward_tensor_infos(tensor_infos)
        # pessimistic bound assuming that all tensors need to get concatenated
        concat_space = self._get_n_values(out_tensor_infos, relevant_tensors=list(out_tensor_infos.keys()))
        return sum([f.get_n_forward(tensor_infos) for f in self.fitters]) + concat_space

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return sum([f.get_n_params(tensor_infos) for f in self.fitters])

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        out_tensor_infos_list = [f.forward_tensor_infos(tensor_infos) for f in self.fitters]
        out_keys = {key for ti in out_tensor_infos_list for key in ti.keys()}
        return {key: TensorInfo(cat_sizes=torch_utils.cat_if_necessary([ti[key].get_cat_sizes()
                                                                        for ti in out_tensor_infos_list
                                                                        if key in ti], dim=-1))
                for key in out_keys}

    def _fit(self, ds: DictDataset) -> Layer:
        return ConcatParallelLayer([f.fit(ds) for f in self.fitters], fitter=self)


# ------ Factory -------

class FitterFactory(ContextAware, StringConvertible):
    """
    Class that allows to create Fitter objects depending on
    tensor_infos (the shape and category sizes of the tensors).
    """
    def __init__(self, scope_names: Optional[List[str]] = None):
        super().__init__(scope_names=scope_names)

    def create(self, tensor_infos: Dict[str, TensorInfo]) -> Fitter:
        """
        Creates a Fitter object with the scope given in the constructor.
        Do not override this method, override _create() or _create_transform() instead.
        :param tensor_infos: Tensor infos (shapes etc.)
        :return: Fitter object.
        """
        fitter = self._create(tensor_infos)
        if fitter is self:
            return fitter
        return fitter.add_others_scope(self)

    def create_transform(self, tensor_infos: Dict[str, TensorInfo]) -> Tuple[Fitter, Dict[str, TensorInfo]]:
        """
        Creates a Fitter object with the scope given in the constructor.
        Do not override this method, override _create() or _create_transform() instead.
        :param tensor_infos: Tensor infos (shapes etc.)
        :return: Fitter object and the transformed tensor infos.
        """
        fitter, tensor_infos = self._create_transform(tensor_infos)
        if fitter is self:
            return fitter, tensor_infos
        return fitter.add_others_scope(self), tensor_infos

    def _create(self, tensor_infos: Dict[str, TensorInfo]) -> Fitter:
        """
        If the subclass also inherits from Fitter, this will just return self.
        Otherwise, override at least one of _create() or _create_transform().
        :param tensor_infos: Tensor infos.
        :return: Fitter object.
        """
        if self.__class__._create_transform != FitterFactory._create_transform:
            # don't have to worry about infinite recursion
            return self._create_transform(tensor_infos)[0]
        if isinstance(self, Fitter):
            return self
        raise NotImplementedError()

    def _create_transform(self, tensor_infos: Dict[str, TensorInfo]) -> Tuple[Fitter, Dict[str, TensorInfo]]:
        fitter = self._create(tensor_infos)
        return fitter, fitter.forward_tensor_infos(tensor_infos)


class SequentialFactory(FitterFactory):
    def __init__(self, factories: List[FitterFactory]):
        super().__init__()
        self.factories = factories

    def _create_transform(self, tensor_infos: Dict[str, TensorInfo]):
        fitters = []
        for f in self.factories:
            fitter, tensor_infos = f.create_transform(tensor_infos)
            fitters.append(fitter)
        return SequentialFitter(fitters), tensor_infos

    def __str__(self):
        sub_strings = ['  ' + line for factory in self.factories for line in str(factory).split('\n')]
        return f'{self.__class__.__name__} [\n' + '\n'.join(sub_strings) + '\n]\n'


class IdentityFactory(FitterFactory):
    def _create(self, tensor_infos):
        return IdentityFitter()


class FunctionFactory(FitterFactory):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def _create(self, tensor_infos):
        return FunctionFitter(self.f)


class ConcatParallelFactory(FitterFactory):
    def __init__(self, factories: List[FitterFactory]):
        super().__init__()
        self.factories = factories

    def _create(self, tensor_infos) -> Fitter:
        return ConcatParallelFitter([factory.create(tensor_infos) for factory in self.factories])


class FilterTensorsFactory(Fitter, FitterFactory):
    def __init__(self, include_keys: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None):
        super().__init__(needs_tensors=False, is_individual=False)
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        return {key: (ti if ((self.include_keys is None or key in self.include_keys)
                             and (self.exclude_keys is None or key not in self.exclude_keys))
                      else TensorInfo(feat_shape=0 * ti.get_feat_shape()))
                      for key, ti in tensor_infos.items()}

    def _fit(self, ds: DictDataset) -> Layer:
        return FilterTensorsLayer(include_keys=self.include_keys, exclude_keys=self.exclude_keys, fitter=self)


class RenameTensorFactory(Fitter, FitterFactory):
    def __init__(self, old_name: str, new_name: str, **config):
        super().__init__(needs_tensors=False, is_individual=False)
        self.old_name = old_name
        self.new_name = new_name

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        if self.old_name in tensor_infos and self.new_name in tensor_infos:
            return self._get_n_values(tensor_infos, [self.old_name, self.new_name])
        else:
            return 0

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        if self.old_name not in tensor_infos:
            return tensor_infos
        elif self.new_name not in tensor_infos:
            return utils.update_dict(tensor_infos, {self.new_name: tensor_infos[self.old_name]},
                                     remove_keys=self.old_name)
        else:
            # both names exist in tensor_infos
            new_tensor_info = TensorInfo.concat([tensor_infos[self.new_name], tensor_infos[self.old_name]])
            return utils.update_dict(tensor_infos, {self.new_name: new_tensor_info},
                                     remove_keys=self.old_name)

    def _fit(self, ds: DictDataset) -> Layer:
        return RenameTensorLayer(old_name=self.old_name, new_name=self.new_name, fitter=self)
