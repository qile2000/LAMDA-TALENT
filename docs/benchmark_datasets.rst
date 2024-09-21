====================================
Benchmark Datasets
====================================

TALENT includes a collection of benchmark datasets designed to cover a wide range of tasks, domains, and dataset sizes. These datasets are essential for evaluating the performance of various machine learning models provided in TALENT.

==========================
Available Datasets
==========================

TALENT supports over **300 datasets** across various task types, including:

- Binary classification
- Multi-class classification
- Regression

These datasets span different domains and sizes, ensuring comprehensive benchmarking of models on tabular data.

==========================
Downloading Datasets
==========================

You can download the benchmark datasets from the following link:

- `Download Datasets from Google Drive <https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link>`_

Once downloaded, place the datasets in the appropriate directory as described below.

==========================
Dataset Structure
==========================

Each dataset consists of the following files:

1. **Numeric features**: `N_train.npy`, `N_val.npy`, `N_test.npy` (can be omitted if there are no numeric features).
2. **Categorical features**: `C_train.npy`, `C_val.npy`, `C_test.npy` (can be omitted if there are no categorical features).
3. **Labels**: `y_train.npy`, `y_val.npy`, `y_test.npy`.
4. **Dataset metadata**: `info.json`, which must include the following fields:
   
   ```json
   {
     "task_type": "regression",  // or "multiclass", "binclass"
     "n_num_features": 10,
     "n_cat_features": 10
   }
   ```

==========================
Placing Datasets
==========================

To correctly set up the datasets for use with TALENT, follow these steps:

1. Place the downloaded datasets in the root directory of your project. The directory structure should follow this format:

   ```bash
   LAMDA-TALENT/
   ├── args.dataset_path/
       ├── dataset1/
       ├── dataset2/
       └── ...
   ```

2. Update the `args.dataset_path` parameter in your configuration files to point to the correct location of the datasets.

For each dataset, `args.dataset_path` should point to the root folder containing the data, and `args.dataset` should specify the exact folder for the dataset being used.

==========================
Using Datasets
==========================

To use a dataset for training or evaluation, follow these steps:

1. Update the dataset path in your experiment configuration file (for example, `configs/default/[MODEL_NAME].json`).

2. Ensure the paths in the configuration file are correct:

   ```json
   {
     "dataset_path": "./args.dataset_path",
     "dataset": "dataset1"
   }
   ```

3. Run your experiment with the appropriate dataset by specifying the correct configuration file.

==========================
Custom Datasets
==========================

You can easily add new datasets to TALENT by following the same structure as the benchmark datasets. Ensure that:

1. The dataset is stored in `args.dataset_path`.
2. The dataset follows the format with numeric and categorical features, as well as labels.
3. The `info.json` file is correctly structured with `task_type`, `n_num_features`, and `n_cat_features`.

Once your dataset is ready, you can update the experiment configuration to include the new dataset, and TALENT will automatically handle it.

==========================
Task Types
==========================

TALENT supports the following task types:

- **Binary Classification**: A task where there are two possible labels for each instance.
- **Multi-class Classification**: A task where there are more than two possible labels for each instance.
- **Regression**: A task where the goal is to predict a continuous value.

Ensure that the `info.json` file for each dataset correctly specifies the task type.

==========================
Conclusion
==========================

The benchmark datasets in TALENT offer a comprehensive set of challenges for evaluating models on tabular data. You can easily add, modify, and use datasets in TALENT by following the structure and instructions provided above. For any additional datasets, follow the same format and place them in the correct directory.