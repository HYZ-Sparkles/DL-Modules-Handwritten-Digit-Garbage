# 微调任务优化计划

## 一、目标
1. ~~优化冻结微调和 LoRA 微调的效果~~（已完成）
2. 解决 25 epochs 时网络崩溃的问题（进行中）

---

## 二、已完成：冻结/LoRA 微调优化

### 冻结微调
- **原问题**：只解冻 fc3（2210 参数），准确率 10.38%
- **解决方案**：只冻结 conv1，解冻 conv2, fc1, fc2, fc3
- **结果**：61.54%

### LoRA 微调
- **原问题**：位置错误 + rank=8 太小
- **解决方案**：位置改到 conv2 输出（400 维），rank=16
- **结果**：53.08%

---

## 三、待解决：25 epochs 崩溃问题

### 25 epochs 实验结果

| 策略 | 崩溃时间 | 最终准确率 |
|------|---------|-----------|
| 全量微调 | Epoch 20 | 崩溃（3.85%） |
| 冻结微调 | Epoch 19 | 崩溃（3.85%） |
| **LoRA 微调** | **无崩溃** | **73.08%** |

### 失败方案
1. **CosineDecayLR**：第 2 个 epoch 就降到 1e-5，训练不足
2. **学习率 0.0001**：训练太慢，10 epochs 仅 18%

### 成功发现
- LoRA 微调在 25 epochs 时表现最好（73.08%）
- 说明 LoRA 的低秩结构提供了天然的 regularization

---

## 四、解决方案：早停机制 + 梯度裁剪

### 方案 A：早停机制（推荐）
```python
best_acc = 0.0
patience = 5  # 连续 5 个 epoch 没有改善就停止
no_improve_count = 0

for epoch in range(epochs):
    model.train(1, train_ds, dataset_sink_mode=False)
    acc = model.eval(test_ds, dataset_sink_mode=False)['accuracy']
    print(f"Epoch {epoch+1}/{epochs}  验证准确率: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        no_improve_count = 0
        # 保存最佳模型
        save_checkpoint(net, f"output/best_model_{strategy_name}.ckpt")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"早停：连续 {patience} 个 epoch 没有改善")
            break
```

### 方案 B：梯度裁剪
```python
from mindspore.nn import TrainOneStepCell, ClipGradients

# 创建带梯度裁剪的训练步骤
grad_clip = ClipGradients(clip_type=2, clip_value=1.0)
train_cell = TrainOneStepCell(loss_net, optimizer, grad_clip)
```

---

## 五、实施计划

### Phase 1：添加早停机制
1. [ ] 修改 `train_eval_strategy` 函数，添加 `patience` 参数
2. [ ] 实现早停逻辑：连续 N 个 epoch 准确率不提升则停止
3. [ ] 保存最佳模型到 `output/` 目录
4. [ ] 测试 25 epochs 是否稳定

### Phase 2：可选 - 梯度裁剪
5. [ ] 如早停不足以解决问题，添加梯度裁剪

---

## 六、验证标准

| 指标 | 目标 |
|------|------|
| 25 epochs 稳定性 | 不崩溃，准确率保持 >30% |
| 最终准确率 | 三种策略均 >60%（除 LoRA 外） |

---

*计划创建时间：2026-05-25*
*最后更新：2026-05-25（添加早停机制方案）*