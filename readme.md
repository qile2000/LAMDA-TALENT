<p align="center">
<img src="./resources/TALENT-LOGO.png"  width="1000px">
</p>
<p align="center">
    <a href='https://arxiv.org/abs/2407.04057'><img src='https://img.shields.io/badge/Arxiv-2407.04057-b31b1b.svg?logo=arXiv'></a>
    <a href='https://zhuanlan.zhihu.com/p/708721145'><img src='https://img.shields.io/badge/中文解读-b0.svg?logo=zhihu'></a>
  <a href=""><img src="https://img.shields.io/github/stars/qile2000/LAMDA-TALENT?color=4fb5ee"></a>
  <a href=""><img src="https://img.shields.io/github/last-commit/qile2000/LAMDA-TALENT?color=blue"></a>
   <br>
    <img src="https://img.shields.io/badge/PYTORCH-2.0.1-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version" height="21">
    <img src="https://img.shields.io/badge/PYTHON-3.10-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version" height="21">
    <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</p>
<div align="center">
    <p>
        TALENT: A Tabular Analytics and Learning Toolbox
    <p>
    <p>
        <a href="https://arxiv.org/abs/2407.04057">[Paper]</a> <a href="https://zhuanlan.zhihu.com/p/708721145">[中文解读]</a> 
    <p>
</div>






---

## 🎉 Introduction

Welcome to **TALENT**, a benchmark with a comprehensive machine learning toolbox designed to enhance model performance on tabular data. TALENT integrates advanced deep learning models, classical algorithms, and efficient hyperparameter tuning, offering robust preprocessing capabilities to optimize learning from tabular datasets. The toolbox is user-friendly and adaptable, catering to both novice and expert data scientists.

**TALENT** offers the following advantages:

- **Diverse Methods**: Includes various classical methods, tree-based methods, and the latest popular deep learning methods.
- **Extensive Dataset Collection**: Equipped with 300 datasets, covering a wide range of task types, size distributions, and dataset domains.
- **Customizability**: Easily allows the addition of datasets and methods.
- **Versatile Support**: Supports diverse normalization, encoding, and metrics.

## 📚Citing TALENT

**If you use any content of this repo for your work, please cite the following bib entries:**

```bibtex
@article{ye2024closerlookdeeplearning,
         title={A Closer Look at Deep Learning on Tabular Data}, 
         author={Han-Jia Ye and Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and De-Chuan Zhan},
         journal={arXiv preprint arXiv:2407.00956},
         year={2024}
}

@article{liu2024talenttabularanalyticslearning,
         title={TALENT: A Tabular Analytics and Learning Toolbox}, 
         author={Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and Han-Jia Ye},
         journal={arXiv preprint arXiv:2407.04057},
         year={2024}
}
```

## 📰 What's New

- [2024-09]🌟 Add [Trompt](https://arxiv.org/abs/2305.18446) (ICML 2023).
- [2024-09]🌟 Add [AMFormer](https://arxiv.org/abs/2402.02334) (AAAI 2024).
- [2024-08]🌟 Add [GRANDE](https://arxiv.org/abs/2309.17130) (ICLR 2024).
- [2024-08]🌟 Add [Excelformer](https://arxiv.org/abs/2301.02819) (KDD 2024).
- [2024-08]🌟 Add [MLP_PLR](https://arxiv.org/abs/2203.05556) (NeurIPS 2022).
- [2024-07]🌟 Add [RealMLP](https://arxiv.org/abs/2407.04491).
- [2024-07]🌟 Add [ProtoGate](https://arxiv.org/abs/2306.12330) (ICML 2024).
- [2024-07]🌟 Add [BiSHop](https://arxiv.org/abs/2404.03830) (ICML 2024).
- [2024-06]🌟 Check out our new baseline [ModernNCA](https://arxiv.org/abs/2407.03257), inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!
- [2024-06]🌟 Check out our [benchmark paper](https://arxiv.org/abs/2407.00956) about tabular data, which provides comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner!

## 🌟 Methods

TALENT integrates an extensive array of 20+ deep learning architectures for tabular data, including but not limited to:

1. **MLP**: A multi-layer neural network, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
2. **ResNet**: A DNN that uses skip connections across many layers, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
3. **[SNN](https://arxiv.org/abs/1706.02515)**: An MLP-like architecture utilizing the SELU activation, which facilitates the training of deeper neural networks.
4. **[DANets](https://arxiv.org/abs/2112.02962)**: A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.
5. **[TabCaps](https://openreview.net/pdf?id=OgbtSLESnI)**: A capsule network that encapsulates all feature values of a record into vectorial features.
6. **[DCNv2](https://arxiv.org/abs/2008.13535)**: Consists of an MLP-like module combined with a feature crossing module, which includes both linear layers and multiplications.
7. **[NODE](https://arxiv.org/abs/1909.06312)**: A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.
8. **[GrowNet](https://arxiv.org/abs/2002.07971)**: A gradient boosting framework that uses shallow neural networks as weak learners.
9. **[TabNet](https://arxiv.org/abs/1908.07442)**: A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.
10. **[TabR](https://arxiv.org/abs/2307.14338)**: A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.
11. **[ModernNCA](https://arxiv.org/abs/2407.03257)**: A deep tabular model inspired by traditional Neighbor Component Analysis, which makes predictions based on the relationships with neighbors in a learned embedding space.
12. **[DNNR](https://arxiv.org/abs/2205.08434)**: Enhances KNN by using local gradients and Taylor approximations for more accurate and interpretable predictions.
13. **[AutoInt](https://arxiv.org/abs/1810.11921)**: A token-based method that uses a multi-head self-attentive neural network to automatically learn high-order feature interactions.
14. **[Saint](https://arxiv.org/abs/2106.01342)**: A token-based method that leverages row and column attention mechanisms for tabular data.
15. **[TabTransformer](https://arxiv.org/abs/2012.06678)**: A token-based method that enhances tabular data modeling by transforming categorical features into contextual embeddings.
16. **[FT-Transformer](https://arxiv.org/abs/2106.11959)**: A token-based method which transforms features to embeddings and applies a series of attention-based transformations to the embeddings.
17. **[TANGOS](https://openreview.net/pdf?id=n6H86gW8u0d)**: A regularization-based method for tabular data that uses gradient attributions to encourage neuron specialization and orthogonalization.
18. **[SwitchTab](https://arxiv.org/abs/2401.02013)**: A self-supervised method tailored for tabular data that improves representation learning through an asymmetric encoder-decoder framework. Following the original paper, our toolkit uses a supervised learning form, optimizing both reconstruction and supervised loss in each epoch.
19. **[PTaRL](https://openreview.net/pdf?id=G32oY4Vnm8)**: A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space.
20. **[TabPFN](https://arxiv.org/abs/2207.01848)**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.
21. **[HyperFast](https://arxiv.org/abs/2402.14335)**: A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.
22. **[TabPTM](https://arxiv.org/abs/2311.00055)**: A general method for tabular data that standardizes heterogeneous datasets using meta-representations, allowing a pre-trained model to generalize to unseen datasets without additional training.
23. **[BiSHop](https://arxiv.org/abs/2404.03830)**: An end-to-end framework for deep tabular learning which leverages a sparse Hopfield model with adaptable sparsity, enhanced by column-wise and row-wise modules.
24. **[ProtoGate](https://arxiv.org/abs/2306.12330)**: A prototype-based model for feature selection in HDLSS biomedical data that adapts global and local feature selection to enhance prediction accuracy and interpretability, addressing co-adaptation issues through a non-parametric prototype-based mechanism.
25. **[RealMLP](https://arxiv.org/abs/2407.04491)**: An improved multilayer perceptron (MLP).
26. **[MLP_PLR](https://arxiv.org/abs/2203.05556)**: An improved multilayer perceptron (MLP), which utilizes periodic activations.
27. **[Excelformer](https://arxiv.org/abs/2301.02819)**: A deep learning model for tabular data prediction, featuring a semi-permeable attention module to address rotational invariance, tailored data augmentation, and an attentive feedforward network, making it a reliable solution across diverse datasets.
28. **[GRANDE](https://arxiv.org/abs/2309.17130)**: A tree-mimic method for learning hard, axis-aligned decision tree ensembles using end-to-end gradient descent.
29. **[AMFormer](https://arxiv.org/abs/2402.02334)**: A token-based method which improves the transformer architecture for tabular data by incorporating parallel addition and multiplication attention mechanisms, utilizing prompt tokens to constrain feature interactions.
30. **[Trompt](https://arxiv.org/abs/2305.18446)**: A prompt-based deep neural network for tabular data that separates learning into intrinsic column features and sample-specific feature importance.




## ☄️ How to Use TALENT

### 🕹️ Clone

Clone this GitHub repository:

```bash
git clone https://github.com/qile2000/LAMDA-TALENT
cd LAMDA-TALENT/LAMDA-TALENT
```

### 🔑 Run experiment

1. Edit the `configs/default/[MODEL_NAME].json`  and `config/opt_space/[MODEL_NAME].json` for global settings and hyperparameters.

2. Run:

    ```bash
    python train_model_deep.py --model_type MODEL_NAME
    ```
    for deep methods, or:
    ```bash
    python train_model_classical.py --model_type MODEL_NAME
    ```
    for classical methods.	

### 🛠️How to Add New Methods

For methods like the MLP class that only need to design the model, you only need to:

- Add the model class in `model/models`.
- Inherit from `model/methods/base.py` and override the `construct_model()` method in the new class.
- Add the method name in the `get_method` function in `model/utils.py`.
- Add the parameter settings for the new method in `configs/default/[MODEL_NAME].json` and `configs/opt_space/[MODEL_NAME].json`.

For other methods that require changing the training process, partially override functions based on `model/methods/base.py`. For details, refer to the implementation of other methods in `model/methods/`.

### 📦 Dependencies

1. [torch](https://github.com/pytorch/pytorch)

2. [scikit-learn](https://github.com/scikit-learn/scikit-learn)

3. [pandas](https://github.com/pandas-dev/pandas)

4. [tqdm](https://github.com/tqdm/tqdm)

5. [numpy](https://github.com/numpy/numpy)

6. [scipy](https://github.com/scipy/scipy)

7. If you want to use **TabR**, you have to manually install faiss, which is only available on **conda**:

   ```bash
   conda install faiss-gpu -c pytorch
   ```

## 🗂️ Benchmark Datasets

Datasets are available at [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link).

### 📂How to Place Datasets

Datasets are placed in the project's current directory, corresponding to the file name specified by `args.dataset_path`. For instance, if the project is `LAMDA-TALENT`, the data should be placed in `LAMDA-TALENT/args.dataset_path/args.dataset`.

Each dataset folder `args.dataset` consists of:

- Numeric features: `N_train/val/test.npy` (can be omitted if there are no numeric features)

- Categorical features: `C_train/val/test.npy` (can be omitted if there are no categorical features)

- Labels: `y_train/val/test.npy`

- `info.json`, which must include the following three contents (task_type can be "regression", "multiclass" or "binclass"):
  

  ```json
  {
    "task_type": "regression", 
    "n_num_features": 10,
    "n_cat_features": 10
  }
  ```

## 📝 Experimental Results

We provide comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner in the Figure. Three tabular prediction tasks, namely, binary classification, multi-class classification, and regression, are considered, and each subfigure represents a different task type.

We use `Accuracy` and `RMSE` as the metrics for classification tasks and regression tasks, respectively. To calibrate the metrics, we choose the average performance rank to compare all methods, where a lower rank indicates better performance, following  [Sheskin (2003)](https://www.taylorfrancis.com/books/mono/10.1201/9781420036268/handbook-parametric-nonparametric-statistical-procedures-david-sheskin). Efficiency is calculated by the average training time in seconds, with lower values denoting better time efficiency. The model size is visually indicated by the radius of the circles, offering a quick glance at the trade-off between model complexity and performance.

The classical method `SVM` provided in TALENT is a `LinearSVM` to ensure faster training.  We also consider the `Dummy` baseline, which outputs the label of the major class and the average labels for classification and regression tasks, respectively.

- Binary classification

  <img src="./resources/binclass.png" style="zoom:36%;" />

- Multiclass Classification

  <img src="./resources/multiclass.png" style="zoom:36%;" />

- Regression

  <img src="./resources/regression.png" style="zoom:36%;" />

- All tasks

  <img src="./resources/all_tasks.png" style="zoom:36%;" />

From the comparison, we observe that **CatBoost** achieves the best average rank in most classification and regression tasks. Among all deep tabular methods, **ModernNCA** performs the best in most cases while maintaining an acceptable training cost. These results highlight the effectiveness of CatBoost and ModernNCA in handling various tabular prediction tasks, making them suitable choices for practitioners seeking high performance and efficiency.

These visualizations serve as an effective tool for quickly and fairly assessing the strengths and weaknesses of various tabular methods across different task types, enabling researchers and practitioners to make informed decisions when selecting suitable modeling techniques for their specific needs.

## 👨‍🏫 Acknowledgments

We thank the following repos for providing helpful components/functions in our work:

- [Rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)
- [Rtdl-num-embeddings](https://github.com/yandex-research/rtdl-num-embeddings)
- [Tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr)
- [DANet](https://github.com/WhatAShot/DANet)
- [TabCaps](https://github.com/WhatAShot/TabCaps)
- [DNNR](https://github.com/younader/dnnr)
- [PTaRL](https://github.com/HangtingYe/PTaRL)
- [Saint](https://github.com/somepago/saint)
- [SwitchTab](https://github.com/avivnur/SwitchTab)
- [TabNet](https://github.com/dreamquark-ai/tabnet)
- [TabPFN](https://github.com/automl/TabPFN)
- [Tabtransformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)
- [TANGOS](https://github.com/alanjeffares/TANGOS)
- [GrowNet](https://github.com/sbadirli/GrowNet)
- [HyperFast](https://github.com/AI-sandbox/HyperFast)
- [BiSHop](https://github.com/MAGICS-LAB/BiSHop)
- [ProtoGate](https://github.com/SilenceX12138/ProtoGate)
- [Pytabkit](https://github.com/dholzmueller/pytabkit)
- [Excelformer](https://github.com/WhatAShot/ExcelFormer)
- [GRANDE](https://github.com/s-marton/GRANDE)
- [AMFormer](https://github.com/aigc-apps/AMFormer)

## 🤗 Contact

If there are any questions, please feel free to propose new features by opening an issue or contact the author: **Siyang Liu** ([liusy@lamda.nju.edu.cn](mailto:liusy@lamda.nju.edu.cn)) and **Haorun Cai** ([caihr@smail.nju.edu.cn](mailto:caihr@smail.nju.edu.cn)) and **Qile Zhou** ([zhouql@lamda.nju.edu.cn](mailto:zhouql@lamda.nju.edu.cn)) and **Han-Jia Ye** ([yehj@lamda.nju.edu.cn](mailto:yehj@lamda.nju.edu.cn)). Enjoy the code.

## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qile2000/LAMDA-TALENT&type=Date)](https://star-history.com/#qile2000/LAMDA-TALENT&Date)

> Thanks [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) and [LAMDA-ZhiJian](https://github.com/zhangyikaii/LAMDA-ZhiJian) for the template.
