import numpy as np
import random

import torch
from torch.utils.checkpoint import checkpoint

import pickle
import io
import os
import pathlib
from pathlib import Path
from functools import partial

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import column_or_1d

from . import transformer
from . import encoders


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num

def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))

def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)

def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)

def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data

def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features)**(1 / 2)
    return x / (num_features_used / num_features)

def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = (data[:, :, col] >= data[:, :, col].unsqueeze(-2))
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x

def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
            # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X

def load_model_only_inference(path, filename, device):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    model_state, optimizer_state, config_sample = torch.load(os.path.join(path, filename), map_location='cpu')

    if (('nan_prob_no_reason' in config_sample and config_sample['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config_sample and config_sample['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config_sample and config_sample['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    n_out = config_sample['max_num_classes']

    device = device if torch.cuda.is_available() else 'cpu:0'
    encoder = encoder(config_sample['num_features'], config_sample['emsize'])

    nhid = config_sample['emsize'] * config_sample['nhid_factor']
    y_encoder_generator = encoders.get_Canonical(config_sample['max_num_classes']) \
        if config_sample.get('canonical_y_encoder', False) else encoders.Linear

    assert config_sample['max_num_classes'] > 2
    loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.ones(int(config_sample['max_num_classes'])))

    model = transformer.TransformerModel(encoder, n_out, config_sample['emsize'], config_sample['nhead'], nhid,
                             config_sample['nlayers'], y_encoder=y_encoder_generator(1, config_sample['emsize']),
                             dropout=config_sample['dropout'],
                             efficient_eval_masking=config_sample['efficient_eval_masking'])

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float('inf'), float('inf'), model), config_sample # no loss measured


def load_model_workflow(i, e, add_name, base_path, device='cpu', eval_addition='', only_inference=True):
    """
    Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param i:
    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """
    def get_file(e):
        """
        Returns the different paths of model_file, model_path and results_file
        """
        model_file = f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt'
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(base_path,
                                    f'models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl')
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            print('We have to download the TabPFN, as there is no checkpoint at ', model_path)
            print('It has about 100MB, so this might take a moment.')
            import requests
            url = 'https://github.com/automl/TabPFN/raw/main/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
            print('hhh')
            r = requests.get(url, allow_redirects=True)
            print('hhh')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, 'wb').write(r.content)
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = model_file_, model_path_, results_file_
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception('No checkpoint found at '+str(model_path))


    #print(f'Loading {model_file}')
    if only_inference:
        # print('Loading model that can be used for inference only')
        model, c = load_model_only_inference(base_path, model_file, device)
    '''
    else:
        #until now also only capable of inference
        model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)
    '''
    #model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file



class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        


import time
def transformer_predict(model, eval_xs, eval_ys, eval_position,
                        device='cpu',
                        max_features=100,
                        style=None,
                        inference_mode=False,
                        num_classes=2,
                        extend_features=True,
                        normalize_with_test=False,
                        normalize_to_ranking=False,
                        softmax_temperature=0.0,
                        multiclass_decoder='permutation',
                        preprocess_transform='mix',
                        categorical_feats=[],
                        feature_shift_decoder=False,
                        N_ensemble_configurations=10,
                        batch_size_inference=16,
                        differentiable_hps_as_style=False,
                        average_logits=True,
                        fp16_inference=False,
                        normalize_with_sqrt=False,
                        seed=0,
                        no_grad=True,
                        return_logits=False,
                        **kwargs):
    """

    :param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        # Initialize results array size S, B, Classes

        # no_grad disables inference_mode, because otherwise the gradients are lost
        inference_mode_call = torch.inference_mode() if inference_mode and no_grad else NOP()
        with inference_mode_call:
            start = time.time()
            output = model(
                    (used_style.repeat(eval_xs.shape[1], 1) if used_style is not None else None, eval_xs, eval_ys.float()),
                    single_eval_pos=eval_position)[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            #else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        #print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))

        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        if eval_xs.shape[2] > max_features:
            eval_xs = eval_xs[:, :, sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))]

        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

        # Removing empty features
        eval_xs = eval_xs[:, 0, :]
        sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            eval_xs = eval_xs.cpu().numpy()
            feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
                range(eval_xs.shape[1])) - set(categorical_feats)
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col:col + 1] = trans
                except:
                    pass
            eval_xs = torch.tensor(eval_xs).float()
        warnings.simplefilter('default')

        eval_xs = eval_xs.unsqueeze(1)

        # TODO: Caution there is information leakage when to_ranking is used, we should not use it
        eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
                if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
        # Rescale X
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                               normalize_with_sqrt=normalize_with_sqrt)

        return eval_xs.to(device)

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    import itertools
    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = softmax_temperature if softmax_temperature.shape else softmax_temperature.unsqueeze(
            0).repeat(num_styles)
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)
    def get_preprocess(i):
        if i == 0:
            return 'power_all'
#            if i == 1:
#                return 'robust_all'
        if i == 1:
            return 'none'

    preprocess_transform_configurations = ['none', 'power_all'] if preprocess_transform == 'mix' else [preprocess_transform]

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    class_shift_configurations = torch.randperm(len(torch.unique(eval_ys))) if multiclass_decoder == 'permutation' else [0]

    ensemble_configurations = list(itertools.product(class_shift_configurations, feature_shift_configurations))
    #default_ensemble_config = ensemble_configurations[0]

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(itertools.product(ensemble_configurations, preprocess_transform_configurations, styles_configurations))
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    #if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration

        style_ = style[styles_configuration:styles_configuration+1, :] if style is not None else style
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            eval_xs_ = preprocess_input(eval_xs_, preprocess_transform=preprocess_transform_configuration)
            if no_grad:
                eval_xs_ = eval_xs_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat([eval_xs_[..., feature_shift_configuration:],eval_xs_[..., :feature_shift_configuration]],dim=-1)

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [eval_xs_,
                 torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)
        inputs += [eval_xs_]
        labels += [eval_ys_]

    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    #print('PREPROCESSING TIME', str(time.time() - start))
    outputs = []
    start = time.time()
    for batch_input, batch_label in zip(inputs, labels):
        #preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="None of the inputs have requires_grad=True. Gradients will be None")
            warnings.filterwarnings("ignore",
                                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
            if device == 'cpu':
                output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
        outputs += [output_batch]
    #print('MODEL INFERENCE TIME ('+str(batch_input.device)+' vs '+device+', '+str(fp16_inference)+')', str(time.time()-start))

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration
        output_ = outputs[:, i:i+1, :]
        output_ = torch.cat([output_[..., class_shift_configuration:],output_[..., :class_shift_configuration]],dim=-1)

        #output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits and not return_logits:
            # transforms every ensemble_configuration into a probability -> equal contribution of every configuration
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits and not return_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output

def get_params_from_config(c):
    return {'max_features': c['num_features']
        , 'rescale_features': c["normalize_by_used_features"]
        , 'normalize_to_ranking': c["normalize_to_ranking"]
        , 'normalize_with_sqrt': c.get("normalize_with_sqrt", False)
            }