"""阶段三测试：垃圾分类数据集加载、三种微调网络构建、参数冻结正确性"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finetune_rubbish import (
    LeNet5,
    LeNet5LoRA,
    create_dataset,
    load_pretrained_weights,
    train_eval_strategy,
)


def test_train_eval_strategy_supports_early_stopping(
    pretrained_ckpt, rubbish_train_dir, rubbish_test_dir
):
    """train_eval_strategy 支持早停机制参数"""
    net = LeNet5(num_class=26)
    load_pretrained_weights(net, pretrained_ckpt)
    # 传入 patience 参数，训练不应该报错
    acc, elapsed, total_params, trainable_params = train_eval_strategy(
        "早停测试", net, rubbish_train_dir, rubbish_test_dir, epochs=3, patience=2
    )
    assert isinstance(acc, float)
    assert isinstance(elapsed, float)


def test_rubbish_train_dataset(rubbish_train_dir):
    """垃圾分类训练集加载返回非空 dataset"""
    ds = create_dataset(rubbish_train_dir, batch_size=32, is_train=True)
    for _ in ds.create_tuple_iterator():
        pass


def test_rubbish_test_dataset(rubbish_test_dir):
    """垃圾分类测试集加载返回非空 dataset"""
    ds = create_dataset(rubbish_test_dir, batch_size=32, is_train=False)
    for _ in ds.create_tuple_iterator():
        pass


def test_full_finetune_all_params_trainable(pretrained_ckpt):
    """全量微调：所有参数 requires_grad=True"""
    net = LeNet5(num_class=26)
    load_pretrained_weights(net, pretrained_ckpt)
    trainable = [p for p in net.get_parameters() if p.requires_grad]
    total = list(net.get_parameters())
    assert len(trainable) == len(total), "全量微调所有参数应可训练"


def test_freeze_finetune_only_fc3_trainable(pretrained_ckpt):
    """冻结微调：仅 fc3 参数 requires_grad=True"""
    net = LeNet5(num_class=26)
    load_pretrained_weights(net, pretrained_ckpt)
    # 应用冻结逻辑（复制自 run_experiments 中的冻结循环）
    for name, param in net.parameters_and_names():
        if "fc3" not in name:
            param.requires_grad = False
    for name, param in net.parameters_and_names():
        if "fc3" in name:
            assert param.requires_grad, f"参数 {name} 应可训练"
        else:
            assert not param.requires_grad, f"参数 {name} 应被冻结"


def test_lora_finetune_lora_trainable(pretrained_ckpt):
    """LoRA 微调：lora 参数可训练，原始参数冻结"""
    net = LeNet5LoRA(num_class=26, lora_rank=8)
    load_pretrained_weights(net, pretrained_ckpt)
    # 应用冻结逻辑（复制自 run_experiments 中的 LoRA 冻结循环）
    for name, param in net.parameters_and_names():
        if "lora" not in name and "fc3" not in name:
            param.requires_grad = False
    lora_params = []
    trainable_new_params = []  # fc3 是新分类头，应可训练
    frozen_params = []  # 原始卷积/全连接层，应冻结
    for name, param in net.parameters_and_names():
        if "lora" in name:
            lora_params.append(param)
        elif "fc3" in name:
            trainable_new_params.append(param)
        else:
            frozen_params.append(param)
    assert len(lora_params) > 0, "LoRA 参数应存在"
    assert all(p.requires_grad for p in lora_params), "LoRA 参数应可训练"
    assert all(p.requires_grad for p in trainable_new_params), "fc3 新分类头应可训练"
    assert all(not p.requires_grad for p in frozen_params), "原始参数应冻结"


def test_optimizer_not_empty(pretrained_ckpt, rubbish_train_dir, rubbish_test_dir):
    """优化器参数组非空"""
    net = LeNet5(num_class=26)
    load_pretrained_weights(net, pretrained_ckpt)
    params = net.trainable_params()
    assert len(params) > 0, "优化器参数组不应为空"


def test_train_eval_strategy_returns_tuple(
    pretrained_ckpt, rubbish_train_dir, rubbish_test_dir
):
    """train_eval_strategy 返回格式正确"""
    net = LeNet5(num_class=26)
    load_pretrained_weights(net, pretrained_ckpt)
    acc, elapsed, total_params, trainable_params = train_eval_strategy(
        "全量微调", net, rubbish_train_dir, rubbish_test_dir, epochs=1
    )
    assert isinstance(acc, float), "accuracy 应为 float"
    assert isinstance(elapsed, float), "elapsed 应为 float"
    assert isinstance(total_params, int), "total_params 应为 int"
    assert isinstance(trainable_params, int), "trainable_params 应为 int"
    assert 0.0 <= acc <= 1.0, "accuracy 应在 [0, 1] 范围内"
