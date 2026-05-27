import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


@pytest.fixture
def data_dir():
    """返回 data/ 目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '..', 'data')


@pytest.fixture
def mnist_data_dir(data_dir):
    """MNIST 数据目录"""
    return os.path.join(data_dir, 'mnist_data')


@pytest.fixture
def rubbish_train_dir(data_dir):
    """垃圾分类训练集目录"""
    return os.path.join(data_dir, 'rubbish', 'train')


@pytest.fixture
def rubbish_test_dir(data_dir):
    """垃圾分类测试集目录"""
    return os.path.join(data_dir, 'rubbish', 'test')


@pytest.fixture
def pretrained_ckpt(data_dir):
    """预训练权重路径"""
    return os.path.join(data_dir, 'lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt')


@pytest.fixture
def src_dir():
    """src/ 目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '..', 'src')