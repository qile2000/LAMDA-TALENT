.. TALENT documentation master file

====================================
TALENT: A Tabular Analytics and Learning Toolbox
====================================

.. image:: ../resources/TALENT-LOGO.png
   :width: 1000px
   :align: center

Welcome to **TALENT**, a comprehensive machine learning toolbox designed to enhance model performance on tabular data. 

TALENT integrates advanced deep learning models, classical algorithms, and efficient hyperparameter tuning, offering robust preprocessing capabilities to optimize learning from tabular datasets. The toolbox is user-friendly and adaptable, catering to both novice and expert data scientists.

.. important::
   If you use any content of this repo for your work, please make sure to cite the relevant papers as described in the `Citing TALENT` section below.

==========================
Citing TALENT
==========================

If you use **TALENT** in your research, please consider citing the following works:

.. code-block:: bibtex

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

==========================
What's New
==========================

Here are the recent updates to **TALENT**:

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

==========================
Contents
==========================

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods
   
.. toctree::
   :maxdepth: 2
   :caption: Dependencies
   
   dependencies

.. toctree::
   :maxdepth: 2
   :caption: Benchmark_Datasets   

   benchmark_datasets

.. toctree::
   :maxdepth: 2
   :caption: Experimental_Results

   experimental_results

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   api/utils
   api/data
   api/method_base

.. toctree::
   :maxdepth: 2
   :caption: Acknowledgements

   acknowledgements

