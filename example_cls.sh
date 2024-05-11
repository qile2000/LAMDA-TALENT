# for classification
datasets_cls=(
    Bank_Customer_Churn_Dataset
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
    tabr
    tabtransformer
)

for dataset in "${datasets_cls[@]}"; do
    for model in "${models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --model_type $model --gpu 0 > "./log/${dataset}-${model}.txt"
    done
    for model in "${indices_models[@]}"; do
        python ./train_model_deep.py --dataset $dataset --model_type $model --gpu 0 --cat_policy indices > "./log/${dataset}-${model}.txt"
    done
    python ./train_model_deep.py --dataset $dataset --model_type tabpfn --gpu 0 --normalization none --cat_policy indices  > "./log/${dataset}-tabpfn.txt"
done