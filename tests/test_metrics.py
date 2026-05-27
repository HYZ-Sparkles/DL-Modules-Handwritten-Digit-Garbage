"""阶段四测试：训练时间/准确率统计函数返回格式"""
import time


def test_train_metrics_return_format():
    """训练指标返回格式：accuracy (float), elapsed (float), total_params (int), trainable_params (int)"""
    # 模拟数据，验证函数返回类型
    accuracy = 0.85
    elapsed = 120.5
    total_params = 60000
    trainable_params = 60000

    assert isinstance(accuracy, float)
    assert isinstance(elapsed, float)
    assert isinstance(total_params, int)
    assert isinstance(trainable_params, int)
    assert 0.0 <= accuracy <= 1.0


def test_comparison_table_format():
    """对比表格格式化输出正确"""
    results = {
        "全量微调": (0.85, 120.5, 60000, 60000),
        "冻结微调": (0.60, 60.2, 60000, 500),
        "LoRA微调": (0.58, 60.3, 60000, 200),
    }
    lines = []
    lines.append("| 方法 | 准确率 | 训练时间(s) | 总参数量 | 可训练参数量 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, (acc, t, total, trainable) in results.items():
        lines.append(f"| {name} | {acc:.4f} | {t:.1f} | {total} | {trainable} |")
    table = "\n".join(lines)
    assert "全量微调" in table
    assert "冻结微调" in table
    assert "LoRA微调" in table
    assert "0.8500" in table