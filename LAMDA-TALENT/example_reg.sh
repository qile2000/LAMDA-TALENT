# for regression
datasets_reg=(
    cpu_small
)
models=(
    danets
    dnnr
    mlp
    node
    resnet
    switchtab
    tabnet
    tangos
)
indices_models=(
    autoint
    dcn2
    ftt
    grownet
    ptarl
    saint
    snn
    tabr
    tabtransformer
)

tabr_ohe_models=(
tabr
modernNCA
)
for dataset in "${datasets_reg[@]}"; do
    for model in "${models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0 > "./log/${dataset}-${model}.txt"
    done
    for model in "${indices_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0 --cat_policy indices > "./log/${dataset}-${model}.txt"
    done
    for model in "${tabr_ohe_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0 --cat_policy tabr_ohe  > "./log/${dataset}-${model}.txt"
    done
done