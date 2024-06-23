
<p align="center">
  <a href=""><img src="https://img.shields.io/badge/TALENT-v0.1-darkcyan"></a>
  <a href=""><img src="https://img.shields.io/github/stars/qile2000/LAMDA-TALENT?color=4fb5ee"></a>
  <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
  <a href=""><img src="https://img.shields.io/github/last-commit/qile2000/LAMDA-TALENT?color=blue"></a>
   <br>
    <img src="https://img.shields.io/badge/PYTORCH-2.0.1-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version" height="21">
    <img src="https://img.shields.io/badge/PYTHON-3.10-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version" height="21">
</p>
<div align="center">
    <p>
        TALENT: A Tabular Analytics and Learning Toolbox
    <p>
    <p>
        <a href="TODO">[Paper]</a> [<b>Code</b>]
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

**If you use any content of this repo for your work, please cite the following bib entry:**

```bibtex
TODO
```

## 🌟 Methods

TALENT integrates an extensive array of 20+ deep learning architectures for tabular data, including but not limited to:

- **MLP**: A multi-layer neural network, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
- **ResNet**: A DNN that uses skip connections across many layers, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
- **[SNN](https://arxiv.org/abs/1706.02515)**: An MLP-like architecture utilizing the SELU activation, which facilitates the training of deeper neural networks.
- **[DANets](https://arxiv.org/abs/2112.02962)**: A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.
- **[TabCaps](https://openreview.net/pdf?id=OgbtSLESnI)**: A capsule network that encapsulates all feature values of a record into vectorial features.
- **[DCNv2](https://arxiv.org/abs/2008.13535)**: Consists of an MLP-like module combined with a feature crossing module, which includes both linear layers and multiplications.
- **[NODE](https://arxiv.org/abs/1909.06312)**: A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.
- **[GrowNet](https://arxiv.org/abs/2002.07971)**: A gradient boosting framework that uses shallow neural networks as weak learners.
- **[TabNet](https://arxiv.org/abs/1908.07442)**: A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.
- **[TabR](https://arxiv.org/abs/2307.14338)**: A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.
- **[ModernNCA](TODO)**: A deep tabular model inspired by traditional Neighbor Component Analysis, which makes predictions based on the relationships with neighbors in a learned embedding space.
- **[DNNR](https://arxiv.org/abs/2205.08434)**: Enhances KNN by using local gradients and Taylor approximations for more accurate and interpretable predictions.
- **[AutoInt](https://arxiv.org/abs/1810.11921)**: A token-based method that uses a multi-head self-attentive neural network to automatically learn high-order feature interactions.
- **[Saint](https://arxiv.org/abs/2106.01342)**: A token-based method that leverages row and column attention mechanisms for tabular data.
- **[TabTransformer](https://arxiv.org/abs/2012.06678)**: A token-based method that enhances tabular data modeling by transforming categorical features into contextual embeddings.
- **[FT-Transformer](https://arxiv.org/abs/2106.11959)**: A token-based method which transforms features to embeddings and applies a series of attention-based transformations to the embeddings.
- **[TANGOS](https://openreview.net/pdf?id=n6H86gW8u0d)**: A regularization-based method for tabular data that uses gradient attributions to encourage neuron specialization and orthogonalization.
- **[SwitchTab](https://arxiv.org/abs/2401.02013)**: A regularization-based method tailored for tabular data that improves representation learning through an asymmetric encoder-decoder framework.
- **[PTaRL](https://openreview.net/pdf?id=G32oY4Vnm8)**: A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space.
- **[TabPFN](https://arxiv.org/abs/2207.01848)**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.
- **[HyperFast](https://arxiv.org/abs/2402.14335)**: A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.
- **[TabPTM](https://arxiv.org/abs/2311.00055)**: A general method for tabular data that standardizes heterogeneous datasets using meta-representations, allowing a pre-trained model to generalize to unseen datasets without additional training.

## ☄️ How to Use TALENT

### 🕹️ Clone

Clone this GitHub repository:

```bash
git clone https://github.com/qile2000/LAMDA-TALENT
cd LAMDA-TALENT/TabBench
```

### 🔑 Run experiment

1. Edit the `[MODEL_NAME].json` file for global settings and hyperparameters.

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

TODO

### 📦 Dependencies

1. [torch](https://github.com/pytorch/pytorch)
2. [scikit-learn](https://github.com/scikit-learn/scikit-learn)
3. [pandas](https://github.com/pandas-dev/pandas)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)

## 🗂️ Benchmark Datasets

Datasets are available at [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link).

### 📂How to Place Datasets

TODO


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

## 📝 Experimental Results

We provide comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner in the Figure. Three tabular prediction tasks, namely, binary classification, multi-class classification, and regression, are considered, and each subfigure represents a different task type.

We use accuracy and RMSE as the metrics for classification and regression, respectively. To calibrate the metrics, we choose the average performance rank to compare all methods, where a lower rank indicates better performance, following  [Sheskin (2003)](https://www.taylorfrancis.com/books/mono/10.1201/9781420036268/handbook-parametric-nonparametric-statistical-procedures-david-sheskin). Efficiency is calculated by the average training time in seconds, with lower values denoting better time efficiency. The model size is visually indicated by the radius of the circles, offering a quick glance at the trade-off between model complexity and performance.

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

## 🤗 Contact

If there are any questions, please feel free to propose new features by opening an issue or contact the author: **Siyang Liu** ([liusiyang@smail.nju.edu.cn](mailto:liusiyang@smail.nju.edu.cn)) and **Haorun Cai** ([caihr@smail.nju.edu.cn](mailto:caihr@smail.nju.edu.cn)) and **Qile Zhou** ([zhouql@lamda.nju.edu.cn](mailto:zhouql@lamda.nju.edu.cn)) and **Han-Jia Ye** ([yehj@lamda.nju.edu.cn](mailto:yehj@lamda.nju.edu.cn)). Enjoy the code.

## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qile2000/LAMDA-TALENT&type=Date)](https://star-history.com/#qile2000/LAMDA-TALENT&Date)
