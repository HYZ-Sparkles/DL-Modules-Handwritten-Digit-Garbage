import mindspore
from mindspore import nn, load_checkpoint, load_param_into_net
from mindspore.dataset import MnistDataset
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
import os

# ---------- 1. 定义 LeNet5 网络结构 ----------
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
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
        x = self.conv1(x)         # [B,6,28,28]
        x = self.relu(x)
        x = self.max_pool2d(x)    # [B,6,14,14]
        x = self.conv2(x)         # [B,16,10,10]
        x = self.relu(x)
        x = self.max_pool2d(x)    # [B,16,5,5]
        x = self.flatten(x)       # [B,400]
        x = self.fc1(x)           # [B,120]
        x = self.relu(x)
        x = self.fc2(x)           # [B,84]
        x = self.relu(x)
        x = self.fc3(x)           # [B,10]
        return x


def load_pretrained_weights(net: LeNet5, ckpt_path: str) -> list[str]:
    """加载预训练权重到网络，返回未加载的参数名列表"""
    param_dict = load_checkpoint(ckpt_path)
    param_not_load, _ = load_param_into_net(net, param_dict)
    return param_not_load


# ---------- 3. 准备 MNIST 测试集 ----------
def create_dataset(data_path: str, batch_size: int = 32):
    os.makedirs(data_path, exist_ok=True)
    mnist_ds = MnistDataset(data_path, usage='test')

    # 数据处理流水线：resize -> 归一化 -> 通道转换 -> 类型转换
    transform_ops = [
        vision.Resize((32, 32), interpolation=Inter.LINEAR),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW(),
    ]
    mnist_ds = mnist_ds.map(operations=transform_ops, input_columns="image")
    # 标签转为 float32（与模型输出对齐）
    mnist_ds = mnist_ds.map(operations=transforms.TypeCast(mindspore.float32), input_columns="label")
    mnist_ds = mnist_ds.batch(batch_size)
    return mnist_ds


# ---------- 4. 评估模型 ----------
def eval_model(net: LeNet5, test_dataset) -> float:
    net.set_train(False)
    total = 0
    correct = 0
    for data in test_dataset.create_dict_iterator():
        images = data["image"]
        labels = data["label"]
        output = net(images)
        pred = output.argmax(axis=1)
        correct += (pred == labels).sum().asnumpy()
        total += len(labels)
    return correct / total


# ---------- 主入口 ----------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, '..', 'data', 'lenet_ascend_v111_offical_cv_mnist_bs32_acc98.ckpt')

    net = LeNet5(num_class=10)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    rel_path = os.path.relpath(ckpt_path, os.path.join(current_dir, "..", ".."))
    print(f"预训练权重加载成功: {rel_path}")

    data_dir = os.path.join(current_dir, '..', 'data', 'mnist_data')
    test_dataset = create_dataset(data_dir)
    print("MNIST 测试集加载完成")

    accuracy = eval_model(net, test_dataset)
    print(f"MNIST 测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")