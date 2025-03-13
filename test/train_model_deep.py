
from tqdm import tqdm
from TALENT.model.utils import (
    get_deep_args,show_results,tune_hyper_parameters,
    get_method,set_seeds
)
from TALENT.model.lib.data import (
    get_dataset
)



if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()
    train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)
    if args.tune:
        args = tune_hyper_parameters(args,opt_space,train_val_data,info)
    ## Training Stage over different random seeds
    
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        time_cost = method.fit(train_val_data, info)    
        vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)

        loss_list.append(vl)
        results_list.append(vres)
        time_list.append(time_cost)

    show_results(args,info, metric_name,loss_list,results_list,time_list)

