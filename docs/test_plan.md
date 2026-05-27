# 测试计划

## 目标
按照 AGENT.md 规范，为三个源代码文件编写集成测试，覆盖率 ≥ 80%（排除训练主循环和可视化代码）。

## 测试文件结构
```
tests/
├── conftest.py         # 共享 fixtures（Dummy 网络、临时数据集路径）
├── test_mnist.py       # 阶段二：test_mnist.py
├── test_finetune.py    # 阶段三：finetune_rubbish.py
└── test_metrics.py     # 阶段四对比实验
```

## 各文件测试覆盖点

### test_mnist.py
- `LeNet5` 网络定义：输出形状正确（10类）
- `LeNet5` 网络定义：输出形状正确（26类，修改后的分类头）
- MNIST 数据集加载：`create_dataset` 返回正确的 batch 形状
- 预训练权重加载：`load_checkpoint` 成功加载官方 ckpt
- 推理：`model.eval` 返回 accuracy 指标

### test_finetune.py
- 垃圾分类数据集加载：train/test 目录均返回非空 dataset
- 全量微调网络：`LeNet5(26)` 参数全部可训练
- 冻结微调网络：`requires_grad=False` 应用于除 fc3 外的所有参数
- LoRA 微调网络：`LeNet5LoRA` 的 lora 参数可训练，原始卷积/全连接参数冻结
- 优化器非空：`train_eval_strategy` 内优化器参数组不为空
- 三种策略训练准确率返回格式：`(accuracy, time, total_params, trainable_params)`

### test_metrics.py
- `train_eval` 返回值类型检查
- 对比表格输出格式正确（字符串格式化）

## 覆盖率排除规则
- `if __name__ == "__main__"` 块
- `run_experiments()` 函数整体（已在 test_finetune.py 分解测试）
- 可视化/打印语句

## 执行命令
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## 数据依赖
- MNIST：自动下载到 `data/mnist_data/`
- Rubbish：`data/rubbish/train/`, `data/rubbish/test/`
- 预训练权重：`data/lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt`