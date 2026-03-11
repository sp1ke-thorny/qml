import math

import torch
import torch.nn as nn
import torch.optim as optim

# PennyLane 可选；没安装时程序会自动退回纯经典网络。
try:
    import pennylane as qml
except ImportError:
    qml = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(num_samples: int = 100, noise_std: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    # 在 [-pi, pi] 上采样，并用带噪声的 sin(x) 作为回归目标。
    x = torch.rand(num_samples, 1) * 2 * torch.pi - torch.pi
    y = torch.sin(x) + torch.randn(num_samples, 1) * noise_std
    return x, y


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 经典基线模型：一层隐藏层的 MLP。
        self.layer1 = nn.Linear(1, 10)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

    def inspect_flow(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.layer1(x)
        activated = self.activation(hidden)
        output = self.layer2(activated)
        return {
            "input": x,
            "hidden_linear": hidden,
            "hidden_activation": activated,
            "output": output,
        }


if qml is not None:
    class QuantumLayer(nn.Module):
        def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
            super().__init__()
            # 使用 PennyLane 的默认量子模拟器，量子比特数决定量子层输出维度。
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch")
            def circuit(inputs: torch.Tensor, weights: torch.Tensor):
                # 先把经典输入编码到量子线路，再执行可训练的纠缠层。
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                # 每个量子比特测量一个期望值，拼成给后续经典层使用的特征向量。
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)


    class HybridQuantumNet(nn.Module):
        def __init__(self, n_qubits: int = 4, n_layers: int = 2, hidden_dim: int = 8) -> None:
            super().__init__()
            # 经典预处理层先把 1 维输入映射成更适合量子编码的特征。
            self.classical_in = nn.Linear(1, hidden_dim)
            self.activation = nn.Tanh()
            self.to_quantum = nn.Linear(hidden_dim, n_qubits)
            self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
            # 量子层输出再交给经典线性层完成最终回归。
            self.classical_out = nn.Linear(n_qubits, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.classical_in(x)
            x = self.activation(x)
            x = self.to_quantum(x)
            # 量子角编码通常用有限范围输入，这里压到 [-pi, pi]。
            x = torch.tanh(x) * math.pi
            x = self.quantum(x)
            x = self.classical_out(x)
            return x

        def inspect_flow(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            classical_features = self.classical_in(x)
            activated_features = self.activation(classical_features)
            quantum_inputs = self.to_quantum(activated_features)
            quantum_angles = torch.tanh(quantum_inputs) * math.pi
            quantum_outputs = self.quantum(quantum_angles)
            final_output = self.classical_out(quantum_outputs)
            return {
                "input": x,
                "classical_features": classical_features,
                "activated_features": activated_features,
                "quantum_inputs": quantum_inputs,
                "quantum_angles": quantum_angles,
                "quantum_outputs": quantum_outputs,
                "output": final_output,
            }
else:
    class HybridQuantumNet(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            raise RuntimeError(
                "HybridQuantumNet requires PennyLane. Install it in the qml environment with "
                "'conda activate qml' and 'pip install pennylane'."
            )


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01,
) -> tuple[nn.Module, list[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    print("\nStart training...")
    for epoch in range(epochs):
        # 前向传播：根据当前参数得到预测值。
        predictions = model(x)
        loss = criterion(predictions, y)

        # 标准 PyTorch 训练三步：清梯度 -> 反向传播 -> 参数更新。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    print("Training finished.")
    return model, losses


def build_model(use_quantum_layer: bool) -> nn.Module:
    # 根据环境里是否有 PennyLane，选择混合模型或经典模型。
    if use_quantum_layer:
        return HybridQuantumNet()
    return SimpleNet()


def print_model_flow(model: nn.Module, x: torch.Tensor, max_samples: int = 3) -> None:
    sample = x[:max_samples]
    print("\nModel flow inspection:")
    print(f"sample input shape: {tuple(sample.shape)}")

    if hasattr(model, "inspect_flow"):
        with torch.no_grad():
            flow = model.inspect_flow(sample)
        for name, value in flow.items():
            detached = value.detach().cpu()
            print(f"{name} shape: {tuple(detached.shape)}")
            print(detached)
    else:
        print("This model does not expose intermediate tensors.")


def main() -> None:
    device = get_device()
    # 只有 PennyLane 可用时，才真正启用量子层示例。
    use_quantum_layer = qml is not None

    print(f"Current device: {device}")
    if use_quantum_layer:
        print("Model mode: hybrid quantum-classical network")
    else:
        print("Model mode: classical network")
        print("PennyLane not found, so the code falls back to SimpleNet.")
        print("To enable the quantum layer, install PennyLane in the qml environment.")

    x, y = make_dataset(num_samples=100, noise_std=0.1)
    x = x.to(device)
    y = y.to(device)

    model = build_model(use_quantum_layer=use_quantum_layer).to(device)
    print_model_flow(model, x)
    _, losses = train(model, x, y, epochs=1000, lr=0.01)
    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
