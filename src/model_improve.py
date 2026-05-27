import mindspore
from mindspore import nn, load_checkpoint, load_param_into_net, Model
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal, Constant
import os
import time


# ---------- 1. 标准 LeNet5 ----------
class StandardLeNet5(nn.Cell):
    """Standard LeNet5 with ReLU"""
    def __init__(self, num_class=26):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
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


# ---------- 2. LeNet5 + LoRA + ELU（最佳模型） ----------
class LoRAAdapter(nn.Cell):
    """低秩适配模块"""
    def __init__(self, in_dim, r=8):
        super().__init__()
        self.lora_A = nn.Dense(in_dim, r, has_bias=False, weight_init=Normal(0.01))
        self.lora_B = nn.Dense(r, in_dim, has_bias=False, weight_init=Constant(0.0))

    def construct(self, x):
        return self.lora_B(self.lora_A(x))


class LeNet5LoRAELU(nn.Cell):
    """LeNet5 with LoRA adapter and ELU activation.

    结构：conv1 → ELU → MaxPool → conv2 → ELU → MaxPool → flatten → LoRA调制 → fc1 → ELU → fc2 → ELU → fc3
    预训练权重加载：conv1, conv2, fc1, fc2 正常加载，fc3 和 LoRA 参数随机初始化
    """
    def __init__(self, num_class=26, lora_rank=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.elu = nn.ELU(alpha=1.0)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # LoRA: 400 -> rank -> 400, 残差连接
        self.lora = LoRAAdapter(16 * 5 * 5, lora_rank)

    def construct(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)  # 400维特征
        x = x + self.lora(x)  # LoRA残差调制
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return x


# ---------- 3. 数据预处理 ----------
def create_dataset(data_path, batch_size=32, is_train=True):
    dataset = ImageFolderDataset(data_path, num_parallel_workers=4,
                                 shuffle=is_train, decode=True)
    transform_img = [
        vision.ToPIL(),
        vision.Grayscale(),
        vision.Resize((32, 32), interpolation=Inter.LINEAR),
        vision.ToTensor(),
    ]
    dataset = dataset.map(operations=transform_img, input_columns="image")
    dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label")
    dataset = dataset.batch(batch_size)
    return dataset


# ---------- 4. 权重加载 ----------
def load_pretrained_weights(net, ckpt_path):
    param_dict = load_checkpoint(ckpt_path)
    keys_to_del = [k for k in list(param_dict.keys()) if 'fc3' in k]
    for k in keys_to_del:
        del param_dict[k]
    param_not_load, _ = load_param_into_net(net, param_dict)
    print(f"未加载的参数（将随机初始化）: {param_not_load}")


# ---------- 5. 训练与评估 ----------
def train_eval(net, train_ds, test_ds, epochs=25, lr=0.0005, patience=5):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})

    best_acc = 0.0
    no_improve_count = 0
    best_params = None
    start_time = time.time()
    for epoch in range(epochs):
        model.train(1, train_ds, dataset_sink_mode=False)
        acc = model.eval(test_ds, dataset_sink_mode=False)['accuracy']
        print(f"Epoch {epoch+1}/{epochs}  验证准确率: {acc:.4f}")
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
    if best_params is not None:
        for p, best_p in zip(net.get_parameters(), best_params):
            p.set_data(best_p)
    print(f"训练耗时: {elapsed:.1f} 秒")
    return best_acc, elapsed


# ---------- 6. 主实验 ----------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, '..', 'data', 'lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt')
    train_path = os.path.join(current_dir, '..', 'data', 'rubbish', 'train')
    test_path = os.path.join(current_dir, '..', 'data', 'rubbish', 'test')

    baseline_acc = 0.7115
    print(f"基准验证准确率: {baseline_acc:.4f}")
    print(f"超参数: lr=0.0005, epochs=25, batch_size=32, patience=5")

    # 测试标准 LeNet5
    print("\n===== 标准 LeNet5（ReLU）基准 =====")
    net_std = StandardLeNet5(num_class=26)
    load_pretrained_weights(net_std, ckpt_path)
    train_ds = create_dataset(train_path, batch_size=32, is_train=True)
    test_ds = create_dataset(test_path, batch_size=32, is_train=False)
    std_acc, std_time = train_eval(net_std, train_ds, test_ds, epochs=25, lr=0.0005, patience=5)
    print(f"标准LeNet5结果: {std_acc:.4f}")

    # 测试最佳模型：LeNet5 + LoRA(r=8) + ELU
    print("\n===== LeNet5 + LoRA(r=8) + ELU（最佳模型） =====")
    net_best = LeNet5LoRAELU(num_class=26, lora_rank=8)
    load_pretrained_weights(net_best, ckpt_path)
    train_ds = create_dataset(train_path, batch_size=32, is_train=True)
    test_ds = create_dataset(test_path, batch_size=32, is_train=False)
    best_acc, best_time = train_eval(net_best, train_ds, test_ds, epochs=25, lr=0.0005, patience=5)
    print(f"LeNet5+LoRA+ELU结果: {best_acc:.4f}")

    # 对比结果
    print("\n===== 对比结果 =====")
    print(f"基准准确率:      {baseline_acc:.4f}")
    print(f"标准LeNet5:      {std_acc:.4f}  ({std_acc - baseline_acc:+.4f})")
    print(f"LeNet5+LoRA+ELU: {best_acc:.4f}  ({best_acc - baseline_acc:+.4f})")

    if best_acc > baseline_acc:
        print(f"\n成功超越基准！提升: {best_acc - baseline_acc:+.4f}")
    else:
        print(f"\n未能超越基准。差距: {best_acc - baseline_acc:+.4f}")