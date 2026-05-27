## 项目概述
本项目基于 MindSpore 深度学习框架，复现 LeNet-5 在 MNIST 上的预训练效果，并将该预训练模型微调至 26 类垃圾分类任务。实验要求对比全量微调、冻结微调、LoRA 微调三种策略的性能，并对模型进行改进以提升准确率。

## 当前进度
- [x] 环境搭建：使用 uv 管理 Python 虚拟环境，MindSpore 2.9.0 CPU 版。
- [x] 阶段二：加载官方 LeNet-5 预训练权重，在 MNIST 测试集上复现准确率 98.50%。
- [x] 阶段三：实现垃圾分类数据集微调全流程，包含数据加载、网络修改、训练验证。
- [x] 实现三种微调策略：全量微调、冻结微调、LoRA 微调。
- [x] 解决冻结微调和 LoRA 微调效果差的问题（冻结微调 61.54%，LoRA 微调 53.08%）
- [ ] 优化模型改进（保证超参数一致）。

## 技术栈
- **深度学习框架**：MindSpore 2.9.0 (CPU)
- **环境管理**：uv (生成 `.venv`，使用 `uv run` 执行脚本)
- **代码质量**：mypy (类型检查)、ruff (代码格式化与 linting)
- **测试框架**：pytest (TDD 驱动开发)
- **版本控制**：Git
- **数据**：MNIST 数据集、26 类垃圾分类数据集

## 项目结构
```
code/
├── data/
│   ├── lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt  # 预训练权重
│   ├── mnist_data/               # MNIST 自动下载目录
│   └── rubbish/
│       ├── train/                # 训练集，子文件夹为类别名 (0-25)
│       └── test/                 # 测试集，同样结构
├── src/
│   ├── test_mnist.py             # 阶段二：验证预训练模型
│   ├── finetune_rubbish.py       # 阶段三：微调垃圾分类（三种策略）
│   └── model_improve.py          # 阶段四：改进模型
├── tests/                        # pytest 测试用例（待添加）
├── pyproject.toml                # uv 项目配置，包含 mypy/ruff 设置
├── AGENT.md                      # 本文件
├── output/                       # 最后准备输出的模型权重，分为三种微调方法以及优化模型的
├── docs/                         # 存放代码编写遇到的问题和下一步计划(md文件)
└── .gitignore
```

## 开发规范
1. **环境管理**  
   - 使用 `uv` 创建虚拟环境并管理依赖。  
   - 运行命令示例：`uv run python src/test_mnist.py`  
   - 添加依赖：基本不需要，我已经自行添加需要的依赖

2. **代码风格与质量**  
   - 所有代码必须通过 `ruff` 格式化与检查。  
   - 类型注解需符合 `mypy` 严格模式。  
   - 使用 `pyproject.toml` 统一配置工具选项。

3. **测试驱动开发 (TDD)**  
   - 所有数据处理函数、网络模块需编写单元测试，放在 `tests/` 目录。  
   - 测试使用 `pytest` 运行：`uv run pytest`  
   - 新功能或修复 Bug 前，先编写失败的测试。

4. **代码结构**  
   - 网络模型定义应模块化，避免重复代码。  
   - 数据加载和预处理封装为独立函数，方便测试和复用。  
   - 训练/验证逻辑与实验管理分离。

5. **不要猜测我的意图。任何不明确的地方都必须都向我提问。 /grill-me**
6. **编程时的过程，遇到的问题和编写代码的计划必须明确，在docs/目录下说明问题和制定计划之后才可以进行下一步，而不是走一步看一步**
    文件分为微调任务和模型改进任务：finetune_plan.md,finetune_error.md以及model_improve_plan.md,model_improve_error.md

## 错误处理策略
- 遇到文件缺失、路径错误、参数配置错误等致命错误，直接抛出明确的异常（`FileNotFoundError` / `ValueError`），并在异常信息中写明具体问题和预期修正方式。
- 网络权重加载失败、参数名不匹配等可以恢复的错误，使用 `load_param_into_net` 的 `strict=False` 或手动删除多余参数，必要输出简单的 `print` 提示，不中断程序。
- 优化器参数组动态构建时，确保不会传入空列表，否则 MindSpore 会直接报错；用简单 `if` 判断过滤空组即可。
- 测试代码（pytest）需覆盖预期会失败的场景，验证异常被正确抛出。

## 测试驱动开发 (TDD) 验收标准
- **覆盖率统计范围**：`src/` 下所有 `.py` 文件，明确排除训练主循环（如 `train_loop.py` 或脚本中 `if __name__ == "__main__"` 块）和纯可视化代码。排除清单在 `pyproject.toml` 的 `[tool.coverage.run]` 中维护。
- **覆盖率目标**：执行 `uv run pytest --cov` 后，`src/` 总行覆盖率 ≥ 80%。允许使用 `# pragma: no cover` 标记极个别无法测试的防御性代码。
- **阶段测试划分**：
  - 阶段二：`test_mnist.py` —— 测试 LeNet5 定义、MNIST 数据加载、预训练权重加载与推理形状。
  - 阶段三：`test_finetune.py` —— 测试垃圾分类数据集加载、三种微调网络构建、参数冻结正确性、优化器非空。
  - 阶段四（对比实验）：`test_metrics.py` —— 测试训练时间/准确率统计函数的返回格式（可用 mock 数据）。
- **测试目录结构**：
```
tests/
├── test_mnist.py
├── test_finetune.py
├── test_metrics.py
└── conftest.py # 共享 fixtures（临时数据集、Dummy 网络等）
```

## 模型权重保存说明
- 训练完成后会自动将模型参数保存至 `output/` 目录，按微调策略命名（如 `lenet_full_finetune.ckpt`、`lenet_freeze_finetune.ckpt`、`lenet_lora_finetune.ckpt`、`lenet_improved.ckpt`）。
- **提交实验时由我来控制是否保存**：代码中已包含完整保存逻辑但默认**注释掉**，避免覆盖已调优好的权重。如需启用，取消 保存模型参数前的注释并运行对应脚本即可。

## AI 助手协作指南
- 所有代码修改需遵循上述开发规范，优先考虑类型安全与测试覆盖。
- 讨论优化方案时，请给出具体代码示例，并说明修改理由。
- 当涉及 MindSpore API 时，注意版本 2.x 的变更 (如 `c_transforms` 已移除)。
- 项目路径必须使用纯英文，避免 MindSpore C++ 层因中文路径报错。