"""
教学版：用 PyTorch 拟合 y = sin(x)。

这个文件保留了和 first_nn.py 一样的核心流程，但把代码拆成函数，
并为每个函数写清楚用途、参数、返回值，便于学习和复用。
"""

import torch
import torch.nn as nn
import torch.optim as optim


def get_device() -> torch.device:
    """
    作用：
    - 自动选择训练设备（GPU 或 CPU）。

    参数：
    - 无

    返回值：
    - torch.device 对象，值通常是 "cuda" 或 "cpu"。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(num_samples: int = 100, noise_std: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    作用：
    - 构造训练数据：输入 X 和标签 y。
    - 目标函数是 y = sin(x)，并叠加高斯噪声。

    参数：
    - num_samples: 样本数量，默认 100。
    - noise_std: 噪声标准差，默认 0.1。

    返回值：
    - X: 形状 [num_samples, 1]，范围在 [-pi, pi]。
    - y: 形状 [num_samples, 1]，约等于 sin(X) + noise。
    """
    X = torch.rand(num_samples, 1) * 2 * torch.pi - torch.pi
    y = torch.sin(X) + torch.randn(num_samples, 1) * noise_std
    return X, y


class SimpleNet(nn.Module):
    """
    两层全连接网络：
    - layer1: Linear(1 -> 10)
    - activation: ReLU
    - layer2: Linear(10 -> 1)

    用法：
    - model = SimpleNet()
    - pred = model(x)
    """

    def __init__(self) -> None:
        """
        作用：
        - 定义网络里的可学习参数（权重和偏置）。

        参数：
        - 只有 self（实例本身）。

        返回值：
        - 无显式返回值（构造器默认返回 None）。
        """
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用：
        - 定义前向传播：输入 x，输出预测值。

        参数：
        - x: torch.Tensor，形状通常是 [batch_size, 1]。

        返回值：
        - torch.Tensor，形状 [batch_size, 1]。
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01,
) -> tuple[nn.Module, list[float]]:
    """
    作用：
    - 执行标准训练循环（前向、算损失、清梯度、反向、更新参数）。

    参数：
    - model: 待训练模型。
    - X: 输入张量，形状 [N, 1]。
    - y: 标签张量，形状 [N, 1]。
    - epochs: 训练轮数。
    - lr: 学习率。

    返回值：
    - model: 训练后的模型。
    - losses: 每一轮的损失列表（Python float）。
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []

    print("\n开始训练...")
    for epoch in range(epochs):
        # 1) 前向传播：得到预测值
        predictions = model(X)

        # 2) 计算损失：衡量 predictions 与 y 的差距
        loss = criterion(predictions, y)

        # 3) 梯度清零：避免梯度在不同轮次累加
        optimizer.zero_grad()

        # 4) 反向传播：计算各参数梯度
        loss.backward()

        # 5) 参数更新：按梯度方向更新权重
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("训练完成！")
    return model, losses


def main() -> None:
    """
    程序入口。

    参数：
    - 无

    返回值：
    - 无
    """
    device = get_device()
    print(f"当前使用设备: {device}")

    X, y = make_dataset(num_samples=100, noise_std=0.1)
    X = X.to(device)
    y = y.to(device)

    model = SimpleNet().to(device)
    _, losses = train(model, X, y, epochs=1000, lr=0.01)
    print(f"最终 Loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
