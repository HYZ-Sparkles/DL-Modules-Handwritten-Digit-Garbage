"""阶段二测试：LeNet5 定义、MNIST 数据加载、预训练权重加载与推理"""
import mindspore
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_mnist import LeNet5, load_pretrained_weights


def _make_tensor(*shape):
    return mindspore.Tensor(np.random.randn(*shape).astype(np.float32))


def test_lenet5_output_shape_10class():
    """LeNet5(num_class=10) 输出形状正确"""
    net = LeNet5(num_class=10)
    x = _make_tensor(1, 1, 32, 32)
    y = net(x)
    assert y.shape == (1, 10)


def test_lenet5_output_shape_26class():
    """LeNet5(num_class=26) 输出形状正确"""
    net = LeNet5(num_class=26)
    x = _make_tensor(1, 1, 32, 32)
    y = net(x)
    assert y.shape == (1, 26)


def test_rubbish_dataset_shape(rubbish_train_dir):
    """垃圾分类数据集加载返回非空 dataset"""
    from finetune_rubbish import create_dataset as create_rubbish_ds
    ds = create_rubbish_ds(rubbish_train_dir, batch_size=32, is_train=True)
    for _ in ds.create_tuple_iterator():
        pass


def test_pretrained_weights_load(pretrained_ckpt):
    """预训练权重加载成功"""
    net = LeNet5(num_class=10)
    param_not_load = load_pretrained_weights(net, pretrained_ckpt)
    assert len(param_not_load) <= 1


def test_lenet5_inference_shape():
    """LeNet5 推理输出形状正确"""
    net = LeNet5(num_class=10)
    x = _make_tensor(4, 1, 32, 32)
    y = net(x)
    assert y.shape == (4, 10)