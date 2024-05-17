import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import json
import torch
import warnings
from model.utils import (
    pprint, set_gpu, mkdir, set_seeds,
    sample_parameters, merge_sampled_parameters, modeltype_to_method
)
from model.lib.data import (
    dataname_to_numpy
)


import optuna
import optuna.samplers
import optuna.trial

import warnings
warnings.filterwarnings("ignore")

def get_args():    
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--dataset', type=str, default='Bank_Customer_Churn_Dataset')
    parser.add_argument('--model_type', type=str, 
                        default='mlp', 
                        choices=['mlp', 'resnet', 'ftt', 'node', 'autoint',
                                 'tabpfn', 'tangos', 'saint', 'tabcaps', 'tabnet',
                                 'snn','ptarl','danets','dcn2','tabtransformer',
                                 'dnnr', 'switchtab', 'grownet', 'tabr'])
    
    # optimization parameters
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)  
    parser.add_argument('--normalization', type=str, default='standard', choices=['none', 'standard', 'minmax', 'quantile', 'maxabs', 'power', 'robust'])
    parser.add_argument('--num_nan_policy', type=str, default='mean', choices=['mean', 'median'])
    parser.add_argument('--cat_nan_policy', type=str, default='new', choices=['new', 'most_frequent'])
    parser.add_argument('--cat_policy', type=str, default='ohe', choices=['indices', 'ordinal', 'ohe', 'binary', 'hash', 'loo', 'target', 'catboost'])
    parser.add_argument('--cat_min_frequency', type=float, default=0.0)

    # other choices
    parser.add_argument('--n_trials', type=int, default=50)    
    parser.add_argument('--seed_num', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tune', action='store_true', default=False)  
    parser.add_argument('--retune', action='store_true', default=False)  
    parser.add_argument('--evaluate_option', type=str, default='best-val')   
    parser.add_argument('--dataset_path', type=str, default='data_cls_resplit')  
    parser.add_argument('--model_path', type=str, default='results_model')
    args = parser.parse_args()
    
    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type])
    save_path2 = 'Epoch{}BZ{}'.format(args.max_epoch, args.batch_size)
    save_path2 += '-Norm-{}'.format(args.normalization)
    save_path2 += '-Nan-{}-{}'.format(args.num_nan_policy, args.cat_nan_policy)
    save_path2 += '-Cat-{}'.format(args.cat_policy)

    if args.cat_min_frequency > 0.0:
        save_path2 += '-CatFreq-{}'.format(args.cat_min_frequency)
    if args.tune:
        save_path1 += '-Tune'

    save_path = osp.join(save_path1, save_path2)
    args.save_path = osp.join(args.model_path, save_path)
    mkdir(args.save_path)    
    
    # load config parameters
    args.seed = 0
    args.config = default_para[args.model_type]
    set_seeds(args.seed)
    return args   


def objective(trial):
    config = {}
    merge_sampled_parameters(
        config, sample_parameters(trial, opt_space[args.model_type], config)
    )    
    if args.model_type in ['resnet']:
        config['model']['activation'] = 'relu'
        config['model']['normalization'] = 'batchnorm'    

    if args.model_type in ['ftt']:
        config['model'].setdefault('prenormalization', False)
        config['model'].setdefault('initialization', 'xavier')
        config['model'].setdefault('activation', 'reglu')
        config['model'].setdefault('n_heads', 8)
        config['model'].setdefault('d_token', 64)
        config['model'].setdefault('token_bias', True)
        config['model'].setdefault('kv_compression', None)
        config['model'].setdefault('kv_compression_sharing', None)    

    if args.model_type in ["node"]:
        config["model"].setdefault("choice_function", "sparsemax")
        config["model"].setdefault("bin_function", "sparsemoid")

    if args.model_type in ['tabr']:
        config['model']["num_embeddings"].setdefault('type', 'PLREmbeddings')
        config['model']["num_embeddings"].setdefault('lite', True)
        config['model'].setdefault('d_multiplier', 2.0)
        config['model'].setdefault('mixer_normalization', 'auto')
        config['model'].setdefault('dropout1', 0.0)
        config['model'].setdefault('normalization', "LayerNorm")
        config['model'].setdefault('activation', "ReLU")

    if args.model_type in ['ptarl']:
        config['model']['n_clusters'] = 20
        config['model']["regularize"]="True"
        config['general']["diversity"]="True"
        config['general']["ot_weight"]=0.25
        config['general']["diversity_weight"]=0.25
        config['general']["r_weight"]=0.25

    if args.model_type in ['danets']:
        config['general']['k'] = 5
        config['general']['virtual_batch_size'] = 256
    
    if args.model_type in ['dcn2']:
        config['model']['stacked'] = False

    if args.model_type in ['grownet']:
        config["ensemble_model"]["lr"] = 1.0
        config['model']["sparse"] = False
        config["training"]['lr_scaler'] = 3

    if args.model_type in ['autoint']:
        config['model'].setdefault('prenormalization', False)
        config['model'].setdefault('initialization', 'xavier')
        config['model'].setdefault('activation', 'relu')
        config['model'].setdefault('n_heads', 8)
        config['model'].setdefault('d_token', 64)
        config['model'].setdefault('kv_compression', None)
        config['model'].setdefault('kv_compression_sharing', None)

    if config.get('config_type') == 'trv4':
        if config['model']['activation'].endswith('glu'):
            # This adjustment is needed to keep the number of parameters roughly in the
            # same range as for non-glu activations
            config['model']['d_ffn_factor'] *= 2 / 3

    trial_configs.append(config)

    # run with this config
    method.fit(N_trainval, C_trainval, y_trainval, info, train=True, config=config)    

    # return validation score    
    return method.trlog['best_res']


with open('default_para.json', 'r') as file:
    default_para = json.load(file)

with open('opt_space.json', 'r') as file:
    opt_space = json.load(file)


if __name__ == '__main__':
        
    args = get_args()
    pprint(vars(args))
    if torch.cuda.is_available():     
        torch.backends.cudnn.benchmark = True    

    N, C, y, info = dataname_to_numpy(args.dataset, args.dataset_path)

    N_trainval = None if N is None else {key: N[key] for key in ["train", "val"]} if "train" in N and "val" in N else None
    N_test = None if N is None else {key: N[key] for key in ["test"]} if "test" in N else None

    C_trainval = None if C is None else {key: C[key] for key in ["train", "val"]} if "train" in C and "val" in C else None
    C_test = None if C is None else {key: C[key] for key in ["test"]} if "test" in C else None

    y_trainval = {key: y[key] for key in ["train", "val"]}
    y_test = {key: y[key] for key in ["test"]} 


    # tune hyper-parameters
    if args.tune:
        if osp.exists(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type))) and args.retune == False:
            with open(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type)), 'rb') as fp:
                args.config = json.load(fp)
        else:
            # get data property
            if info['task_type'] == 'regression':
                direction = 'minimize'
                for key in opt_space[args.model_type]['model'].keys():
                    if 'dropout' in key and '?' not in opt_space[args.model_type]['model'][key][0]:
                        opt_space[args.model_type]['model'][key][0] = '?'+ opt_space[args.model_type]['model'][key][0]
                        opt_space[args.model_type]['model'][key].insert(1, 0.0)
            else:
                direction = 'maximize'  
            
            method = modeltype_to_method(args.model_type)(args, info['task_type'] == 'regression')      

            trial_configs = []
            study = optuna.create_study(
                    direction=direction,
                    sampler=optuna.samplers.TPESampler(seed=0),
                )        
            study.optimize(
                objective,
                **{'n_trials': args.n_trials},
                show_progress_bar=True,
            ) 
            # get best configs
            best_trial_id = study.best_trial.number
            # update config files        
            print('Best Hyper-Parameters')
            print(trial_configs[best_trial_id])
            args.config = trial_configs[best_trial_id]
            with open(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type)), 'w') as fp:
                json.dump(args.config, fp, sort_keys=True, indent=4)

    ## Training Stage over different random seeds
    loss_list, results_list, time_list = [], [], []
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        method = modeltype_to_method(args.model_type)(args, info['task_type'] == 'regression')
        time_cost = method.fit(N_trainval, C_trainval, y_trainval, info)    
        vl, vres, metric_name, predic_logits = method.predict(N_test, C_test, y_test, info, model_name=args.evaluate_option)

        loss_list.append(vl)
        results_list.append(vres)
        time_list.append(time_cost)
        
    metric_arrays = {name: [] for name in metric_name}  


    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])

    metric_arrays['Time'] = time_list
    metric_name = metric_name + ('Time', )

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    mean_loss = np.mean(np.array(loss_list))

    # Printing results
    print(f'{args.model_type}: {args.seed_num} Trials')
    for name in metric_name:
        if info['task_type'] == 'regression' and name != 'Time':
            formatted_results = ', '.join(['{:.8e}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}')
        else:
            formatted_results = ', '.join(['{:.8f}'.format(e) for e in metric_arrays[name]])
            print(f'{name} Results: {formatted_results}')
            print(f'{name} MEAN = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}')

    print(f'Mean Loss: {mean_loss:.8e}')
    
    print('-' * 20, 'GPU info', '-' * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}")
            print(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
            print(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            print(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        print("CUDA is unavailable.")
    print('-' * 50)