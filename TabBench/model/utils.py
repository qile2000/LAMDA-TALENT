import os
import shutil
import time
import errno
import pprint
import torch
import numpy as np
import random
import json



THIS_PATH = os.path.dirname(__file__)

def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
    else:
        os.mkdir(path)


#  --- criteria helper ---
class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

#  ---- import from lib.util -----------
def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    assert 0 <= base_seed < 2 ** 32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)

def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sklearn.metrics as skm
def rmse(y, prediction, y_info):
    rmse = skm.mean_squared_error(y, prediction) ** 0.5  # type: ignore[code]
    if y_info['policy'] == 'mean_std':
        rmse *= y_info['std']
    return rmse
    
def load_config(args, config=None, config_name=None):
    if config is None:
        config_path = os.path.join(os.path.abspath(os.path.join(THIS_PATH, '..')), 
                                   'configs', args.dataset, 
                                   '{}.json'.format(args.model_type if args.config_name is None else args.config_name))
        with open(config_path, 'r') as fp:
            config = json.load(fp)

    # set additional parameters
    args.config = config 
    
    # save the config files
    with open(os.path.join(args.save_path, 
                           '{}.json'.format('config' if config_name is None else config_name)), 'w') as fp:
        args_dict = vars(args)
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return args

# parameter search
def sample_parameters(trial, space, base_config):
    def get_distribution(distribution_name):
        return getattr(trial, f'suggest_{distribution_name}')

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            distribution, *args = subspace

            if distribution.startswith('?'):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip('?'))(label, *args[1:])
                    if trial.suggest_categorical(f'optional_{label}', [False, True])
                    else default_value
                )

            elif distribution == '$mlp_d_layers':
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
                suggest_dim = lambda name: trial.suggest_int(name, d_min, d_max)  # noqa
                d_first = [suggest_dim('d_first')] if n_layers else []
                d_middle = (
                    [suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else []
                )
                d_last = [suggest_dim('d_last')] if n_layers > 1 else []
                result[label] = d_first + d_middle + d_last

            elif distribution == '$d_token':
                assert len(args) == 2
                try:
                    n_heads = base_config['model']['n_heads']
                except KeyError:
                    n_heads = base_config['model']['n_latent_heads']

                for x in args:
                    assert x % n_heads == 0
                result[label] = trial.suggest_int('d_token', *args, n_heads)  # type: ignore[code]

            elif distribution in ['$d_ffn_factor', '$d_hidden_factor']:
                if base_config['model']['activation'].endswith('glu'):
                    args = (args[0] * 2 / 3, args[1] * 2 / 3)
                result[label] = trial.suggest_uniform('d_ffn_factor', *args)

            else:
                result[label] = get_distribution(distribution)(label, *args)
    return result

def merge_sampled_parameters(config, sampled_parameters):
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            # If there are parameters in the default config, the value of the parameter will be overwritten.
            config[k] = v


def modeltype_to_method(model):
    if model == "mlp":
        from model.methods.mlp import MLPMethod
        return MLPMethod
    elif model == 'resnet':
        from model.methods.resnet import ResNetMethod
        return ResNetMethod
    elif model == 'node':
        from model.methods.node import NodeMethod
        return NodeMethod
    elif model == 'ftt':
        from model.methods.ftt import FTTMethod
        return FTTMethod
    elif model == 'tabpfn':
        from model.methods.tabpfn import TabPFNMethod
        return TabPFNMethod
    elif model == 'tabr':
        from model.methods.tabr import TabRMethod
        return TabRMethod
    elif model == 'modernNCA':
        from model.methods.modernNCA import ModernNCAMethod
        return ModernNCAMethod
    elif model == 'tabcaps':
        from model.methods.tabcaps import TabCapsMethod
        return TabCapsMethod
    elif model == 'tabnet':
        from model.methods.tabnet import TabNetMethod
        return TabNetMethod
    elif model == 'saint':
        from model.methods.saint import SaintMethod
        return SaintMethod
    elif model == 'tangos':
        from model.methods.tangos import TangosMethod
        return TangosMethod    
    elif model == 'snn':
        from model.methods.snn import SNNMethod
        return SNNMethod
    elif model == 'ptarl':
        from model.methods.ptarl import PTARLMethod
        return PTARLMethod
    elif model == 'danets':
        from model.methods.danets import DANetsMethod
        return DANetsMethod
    elif model == 'dcn2':
        from model.methods.dcn2 import DCN2Method
        return DCN2Method
    elif model == 'tabtransformer':
        from model.methods.tabtransformer import TabTransformerMethod
        return TabTransformerMethod
    elif model == 'grownet':
        from model.methods.grownet import GrowNetMethod
        return GrowNetMethod
    elif model == 'autoint':
        from model.methods.autoint import AutoIntMethod
        return AutoIntMethod
    elif model == 'dnnr':
        from model.methods.dnnr import DNNRMethod
        return DNNRMethod
    elif model == 'switchtab':
        from model.methods.switchtab import SwitchTabMethod
        return SwitchTabMethod
    elif model == 'xgboost':
        from model.classical_methods.xgboost import XGBoostMethod
        return XGBoostMethod
    elif model == 'LogReg':
        from model.classical_methods.logreg import LogRegMethod
        return LogRegMethod
    elif model == 'NCM':
        from model.classical_methods.ncm import NCMMethod
        return NCMMethod
    elif model == 'lightgbm':
        from model.classical_methods.lightgbm import LightGBMMethod
        return LightGBMMethod
    elif model == 'NaiveBayes':
        from model.classical_methods.naivebayes import NaiveBayesMethod
        return NaiveBayesMethod
    elif model == 'knn':
        from model.classical_methods.knn import KnnMethod
        return KnnMethod
    elif model == 'RandomForest':
        from model.classical_methods.randomforest import RandomForestMethod
        return RandomForestMethod
    elif model == 'catboost':
        from model.classical_methods.catboost import CatBoostMethod
        return CatBoostMethod
    elif model == 'svm':
        from model.classical_methods.svm import SvmMethod
        return SvmMethod
    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")
