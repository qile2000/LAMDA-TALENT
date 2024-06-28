# for classification
datasets_cls=(
    bank
)
models=(
    danets
    mlp
    node
    resnet
    switchtab
    tabcaps
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
    tabtransformer
)
tabr_ohe_models=(
tabr
modernNCA
)
for dataset in "${datasets_cls[@]}";do
    for model in "${models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0  > "./log/${dataset}-${model}.txt"
    done
    for model in "${indices_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0 --cat_policy indices  > "./log/${dataset}-${model}.txt"
    done
    for model in "${tabr_ohe_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --max_epoch 5 --seed_num 1 --model_type $model --gpu 0 --cat_policy tabr_ohe  > "./log/${dataset}-${model}.txt"
    done
    python ./train_model_deep.py --dataset $dataset --dataset_path example_datasets --seed_num 1 --model_type tabpfn --gpu 0 --normalization none --cat_policy indices  > "./log/${dataset}-tabpfn.txt"
done