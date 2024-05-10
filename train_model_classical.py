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
                        default='xgboost', 
                        choices=['LogReg', 'NCM', 'RandomForest', 
                                 'xgboost', 'catboost', 'lightgbm',
                                 'svm','knn', 'NaiveBayes'
                                 ])
    
    # optimization parameters 
    parser.add_argument('--normalization', type=str, default='standard')
    parser.add_argument('--num_nan_policy', type=str, default='mean', choices=['mean', 'median'])
    parser.add_argument('--cat_nan_policy', type=str, default='new', choices=['new', 'most_frequent'])
    parser.add_argument('--cat_policy', type=str, default='ohe')
    parser.add_argument('--cat_min_frequency', type=float, default=0.0)

    # other choices
    parser.add_argument('--n_trials', type=int, default=50)    
    parser.add_argument('--seed_num', type=int, default=10)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tune', action='store_true', default=False)  
    parser.add_argument('--retune', action='store_true', default=False)  
    parser.add_argument('--dataset_path', type=str, default='data_cls_resplit')  
    parser.add_argument('--model_path', type=str, default='results_model')
    parser.add_argument('--evaluate_option', type=str, default='best-val') 
    args = parser.parse_args()
    
    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type])
    
    save_path2 = 'Norm-{}'.format(args.normalization)
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
    if args.model_type == 'xgboost' and torch.cuda.is_available():
        config['model']['tree_method'] = 'gpu_hist' 
        config['model']['gpu_id'] = args.gpu
        config['fit']["verbose"] = False
    elif args.model_type == 'catboost' and torch.cuda.is_available():
        config['fit']["logging_level"] = "Silent"
    
    elif args.model_type == 'RandomForest':
        config['model']['max_depth'] = 12
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
    
    if args.model_type == 'xgboost' and torch.cuda.is_available():
        args.config['model']['tree_method'] = 'gpu_hist' 
        args.config['model']['gpu_id'] = args.gpu

    ## Training Stage over different random seeds
    results_list = []
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        method = modeltype_to_method(args.model_type)(args, info['task_type'] == 'regression')
        method.fit(N_trainval, C_trainval, y_trainval, info,train=True)    
        vres, metric_name, predic_logits = method.predict(N_test, C_test, y_test, info, model_name=args.evaluate_option)

        results_list.append(vres)
        
    metric_arrays = {name: [] for name in metric_name}  


    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])


    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    

    # Printing results
    print(f'{args.model_type}: {args.seed_num} Trials')
    for name in metric_name:
        formatted_results = ','.join(['{:.4f}'.format(e) for e in metric_arrays[name]])
        print(f'{name} Results: {formatted_results}')
        print(f'{name} MEAN = {mean_metrics[name]:.4f} ± {std_metrics[name]:.4f}')

    