## 项目介绍
LeNet5模型与MNIST手写数字数据集的组合，被广泛誉为深度学习领域的“Hello World”。这一经典组合不仅是初学者理解卷积神经网络(CNN)核心原理的入门载体，也为基于深度学习的图像分类任务奠定了基础范式。然而在实际应用中，从头训练一个深度模型不仅耗时且需要大量计算资源，模型还容易陷入局部极小值和过拟合。因此，大部分任务都会选择预训练模型，在其上做微调(Fine‑Tune)，以较小成本实现高精度分类。

垃圾分类是环境保护与智慧城市建设中的重要环节，亟需高效的智能识别技术支撑。本实验以MindSpore框架为载体，探索如何将轻量级网络应用于垃圾分类这一细分图像分类任务，为深度学习技术在实际场景的落地应用积累实践经验。

## 环境说明
使用python版本为3.11

使用uv管理python环境

## 项目结构
最主要的模块就是 src/ ，存放了核心代码。

## 代码快速复现
垃圾分类数据集：https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/MindStudio-pc/data_en.zip

mnist数据集：https://www.kaggle.com/datasets/hojjatk/mnist-dataset

mnist预训练文件：https://download-mindspore.osinfra.cn/model_zoo/r1.1/lenet_ascend_v111_offical_cv_mnist_bs32_acc98/lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt

首先下载对应的数据集和权重放置在./data/
- mnist数据集：.\data\mnist_data
- 垃圾分类数据集：.\data\rubbish
- 权重：.\data\lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt

接下来运行`uv sync`下载对应的python环境

进入具体代码的文件夹：`cd ./src`

1. 复现预训练模型在MNIST上的效果：`uv run test_mnist.py`
2. 进行微调模型的测试：`uv run finetune_rubbish.py`
3. 改进模型后的效果测试：`uv run model_improve.py`

## 效果说明
原始的预训练模型在MNIST数据集上准确率可以达到98.5%，但是我的代码微调和改进模型之后在垃圾分类数据集上准确率只有70%左右。所以只可以供参考学习。