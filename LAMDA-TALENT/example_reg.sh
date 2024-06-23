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
for dataset in "${datasets_reg[@]}"; do
    for model in "${models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --model_type $model --gpu 1 > "./log/${dataset}-${model}.txt"
    done
    for model in "${indices_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --model_type $model --gpu 1 --cat_policy indices > "./log/${dataset}-${model}.txt"
    done
done