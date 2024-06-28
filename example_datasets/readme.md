
# Benchmark Dataset for *A Closer Look at Deep Learning on Tabular Data*

This repository contains supplemental datasets for the paper ***A Closer Look at Deep Learning on Tabular Data***. which are available at [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link). The datasets are provided in two zip files: `benchmark_dataset.zip` and `training_dynamic_informations.zip`. 

## 1. `benchmark_dataset.zip`

This zip file contains all the tabular datasets used in the paper. Each dataset is stored in a separate subfolder named after the dataset. Each dataset folder consists of:

- **Numeric features**: `N_train.npy`, `N_val.npy`, `N_test.npy` (can be omitted if there are no numeric features)
- **Categorical features**: `C_train.npy`, `C_val.npy`, `C_test.npy` (can be omitted if there are no categorical features)
- **Labels**: `y_train.npy`, `y_val.npy`, `y_test.npy`
- **Info file**: `info.json`, which must include the following three components:

  ```json
  {
    "task_type": "regression", 
    "n_num_features": 10,
    "n_cat_features": 10
  }
  ```
  - `task_type`: The type of task, can be `"regression"`, `"multiclass"`, or `"binclass"`.
  - `n_num_features`: Number of numeric features.
  - `n_cat_features`: Number of categorical features.

## 2. `training_dynamic_informations.zip`

This zip file contains training dynamic information recorded during the training of various deep learning methods on the benchmark tabular datasets. The information is organized into three folders:

### 2.1 `time`

This folder contains files named with the pattern `seed_time_costs`, which record the time costs for different seeds during the training process. Each file provides insights into the computational efficiency and time required for training the models on the benchmark datasets, recorded at each epoch.

### 2.2 `loss`

This folder contains files named with the pattern `seed_val_losses`, which record the validation losses for different seeds during the training process. Each file helps evaluate the performance of the models in terms of validation loss, offering a perspective on model accuracy and generalization, recorded at each epoch.

### 2.3 `val_res`

This folder contains files named with the pattern `seed_results`, which record various results for different seeds during the training process. Each file includes metrics Accuracy/RMSE on the vaildation set, etc., providing a comprehensive view of the models' performance on the benchmark datasets, recorded at each epoch.

## Additional Information

These datasets are essential for understanding the training dynamics and performance characteristics of different deep learning methods applied to tabular data. The information can be used for in-depth analysis and comparison of various models, aiding in the development and refinement of deep learning techniques for tabular data.
