# 如何放置数据集

数据集放置于项目目录的上层路径，由args.dataset_path对应文件名。即若项目为TabularBenchmark，数据放置于TabularBenchmark/../args.dataset_path/args.dataset

每个数据集文件夹args.dataset由:

- 数值型特征N_train/val/test.npy(无类别型特征时可略去)

- C_train/val/test.npy(无数值型特征时可略去)

- y_train/val/test.npy

- info.json组成，info.json中必须包含以下三个内容：

  {

    "task_type": 'regression'或'multiclass'或'binclass'

    "n_num_features": 10,

    "n_cat_features": 10

  }

# 如何跑方法

示例见example_cls.sh与example_reg.sh，其余args调整参照train_model_deep.py/train_model_classical.py的get_args()

设有多种:

- **encoding：**见model/lib/data.py的data_enc_process函数。目前实验发现target encoding效果较好，除了标准的onehot encoding外，推荐使用target。
- **normalization：**见model/lib/data.py的data_norm_process函数。目前实验发现normalization影响较小，推荐使用standard。
- **metric：**见model/methods/base.py的metric函数，跑任何方法与数据集将会计算所有metric。

# 如何加新方法

如MLP类仅需设计模型的方法，仅需：

- 继承model/methods/base.py，并在新类中重写construct_model()
- 在model/models中添加模型类
- 在model/utils.py的函数modeltype_to_method中添加方法名
- 在default_para.json, opt_space.json中加入新方法的参数设定

其余需要改变训练流程的方法，基于model/methods/base.py重写部分函数即可。具体可参考model/methods/中其他方法的实现。

