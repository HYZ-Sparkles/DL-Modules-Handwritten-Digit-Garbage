import mindspore
from mindspore import nn, load_checkpoint, load_param_into_net, Model
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal, Constant
import os
import time


# ---------- 1. 基础 LeNet5 ----------
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# ---------- 2. LoRA 适配器 ----------
class LoRAAdapter(nn.Cell):
    """纯低秩适配模块，没有原始权重"""

    def __init__(self, in_dim, r=8):
        super().__init__()
        self.lora_A = nn.Dense(in_dim, r, has_bias=False, weight_init=Normal(0.01))
        self.lora_B = nn.Dense(r, in_dim, has_bias=False, weight_init=Constant(0.0))

    def construct(self, x):
        return self.lora_B(self.lora_A(x))


class LeNet5LoRA(LeNet5):
    """LoRA 插入在 conv2 输出（flatten 后，400 维），在 fc1 之前"""

    def __init__(self, num_class=26, lora_rank=16):
        super().__init__(num_class)
        # conv2 输出 flatten 后是 400 维
        self.lora = LoRAAdapter(16 * 5 * 5, lora_rank)  # 400 -> rank -> 400
        self.fc3 = nn.Dense(84, num_class)  # 新的分类头（随机初始化）

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)  # 400 维特征
        # LoRA 调制：在进入 fc1 之前对特征进行低秩扰动
        x = x + self.lora(x)  # 残差连接
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# ---------- 3. 数据加载 ----------
def create_dataset(data_path, batch_size=32, is_train=True):
    dataset = ImageFolderDataset(
        data_path, num_parallel_workers=4, shuffle=is_train, decode=True
    )
    transform_img = [
        vision.ToPIL(),
        vision.Grayscale(),
        vision.Resize((32, 32), interpolation=Inter.LINEAR),
        vision.ToTensor(),
    ]
    dataset = dataset.map(operations=transform_img, input_columns="image")
    dataset = dataset.map(
        operations=transforms.TypeCast(mindspore.int32), input_columns="label"
    )
    dataset = dataset.batch(batch_size)
    return dataset


# ---------- 4. 权重加载 ----------
def load_pretrained_weights(net, ckpt_path):
    param_dict = load_checkpoint(ckpt_path)
    # 删除 fc3 相关参数（无论前缀）
    keys_to_del = [k for k in param_dict if "fc3" in k]
    for k in keys_to_del:
        del param_dict[k]
    param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
    return param_not_load


# ---------- 5. 训练与评估 ----------
def train_eval_strategy(
    strategy_name,
    net,
    train_path,
    test_path,
    epochs=10,
    lr=0.001,
    patience=5,
):
    """训练与评估，支持早停机制。

    Args:
        strategy_name: 实验名称
        net: 网络模型
        train_path: 训练集路径
        test_path: 测试集路径
        epochs: 最大训练轮数
        lr: 学习率
        patience: 早停耐心值，连续 N 个 epoch 无改善则停止
    """
    train_ds = create_dataset(train_path, 32, is_train=True)
    test_ds = create_dataset(test_path, 32, is_train=False)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})

    print(f"\n{'=' * 50}")
    print(f"开始训练：【{strategy_name}】")
    best_acc = 0.0
    no_improve_count = 0
    best_params = None
    start_time = time.time()
    for epoch in range(epochs):
        model.train(1, train_ds, dataset_sink_mode=False)
        acc = model.eval(test_ds, dataset_sink_mode=False)["accuracy"]
        print(f"Epoch {epoch + 1}/{epochs}  验证准确率: {acc:.4f}")
        # 崩溃检测：准确率低于随机猜测基线（26类≈3.85%）且非首个epoch
        if epoch > 0 and acc < 0.04:
            print(f"崩溃检测：准确率 {acc:.4f} 低于随机基线，停止训练")
            break
        if acc > best_acc:
            best_acc = acc
            no_improve_count = 0
            best_params = [p.clone() for p in net.get_parameters()]
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"早停：连续 {patience} 个 epoch 没有改善，停止训练")
                break

    elapsed = time.time() - start_time
    total_params = sum(p.size for p in net.get_parameters())
    trainable_params = sum(p.size for p in net.trainable_params())
    print(f"训练完成，耗时: {elapsed:.1f} 秒")
    print(f"总参数量: {total_params}，可训练参数量: {trainable_params}")
    # 恢复最佳模型参数
    if best_params is not None:
        for p, best_p in zip(net.get_parameters(), best_params):
            p.set_data(best_p)
    return best_acc, elapsed, total_params, trainable_params


# ---------- 6. 主实验 ----------
def run_experiments():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(
        current_dir, "..", "data", "lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt"
    )
    train_path = os.path.join(current_dir, "..", "data", "rubbish", "train")
    test_path = os.path.join(current_dir, "..", "data", "rubbish", "test")

    num_epochs = 25
    results = {}

    # 策略1：全量微调
    print("\n>>> 策略1：全量微调")
    net_full = LeNet5(26)
    load_pretrained_weights(net_full, ckpt_path)
    # 层信息打印...
    acc_full, time_full, total_full, train_full = train_eval_strategy(
        "全量微调", net_full, train_path, test_path, num_epochs, lr=0.0005
    )
    results["全量微调"] = (acc_full, time_full, total_full, train_full)

    # 策略2：冻结微调（策略 A-4：只冻结 conv1，解冻 conv2, fc1, fc2, fc3）
    print("\n>>> 策略2：冻结微调")
    net_freeze = LeNet5(26)
    load_pretrained_weights(net_freeze, ckpt_path)
    for name, param in net_freeze.parameters_and_names():
        if "conv1" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("冻结层: conv1 | 解冻层: conv2, fc1, fc2, fc3")
    acc_freeze, time_freeze, total_freeze, train_freeze = train_eval_strategy(
        "冻结微调", net_freeze, train_path, test_path, num_epochs, lr=0.0005
    )
    results["冻结微调"] = (acc_freeze, time_freeze, total_freeze, train_freeze)

    # 策略3：LoRA 微调（解冻 conv2, fc1, fc2, lora, fc3；冻结 conv1）
    print("\n>>> 策略3：LoRA 微调")
    net_lora = LeNet5LoRA(num_class=26, lora_rank=16)
    load_pretrained_weights(net_lora, ckpt_path)
    for name, param in net_lora.parameters_and_names():
        if "conv1" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("冻结层: conv1 | 解冻层: conv2, fc1, fc2, lora, fc3")
    acc_lora, time_lora, total_lora, train_lora = train_eval_strategy(
        "LoRA微调", net_lora, train_path, test_path, num_epochs
    )
    results["LoRA微调"] = (acc_lora, time_lora, total_lora, train_lora)


if __name__ == "__main__":
    run_experiments()
