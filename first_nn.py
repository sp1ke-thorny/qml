import math

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import pennylane as qml
except ImportError:
    qml = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(num_samples: int = 100, noise_std: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(num_samples, 1) * 2 * torch.pi - torch.pi
    y = torch.sin(x) + torch.randn(num_samples, 1) * noise_std
    return x, y


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


if qml is not None:
    class QuantumLayer(nn.Module):
        def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
            super().__init__()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch")
            def circuit(inputs: torch.Tensor, weights: torch.Tensor):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)


    class HybridQuantumNet(nn.Module):
        def __init__(self, n_qubits: int = 4, n_layers: int = 2, hidden_dim: int = 8) -> None:
            super().__init__()
            self.classical_in = nn.Linear(1, hidden_dim)
            self.activation = nn.Tanh()
            self.to_quantum = nn.Linear(hidden_dim, n_qubits)
            self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
            self.classical_out = nn.Linear(n_qubits, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.classical_in(x)
            x = self.activation(x)
            x = self.to_quantum(x)
            x = torch.tanh(x) * math.pi
            x = self.quantum(x)
            x = self.classical_out(x)
            return x
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
        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    print("Training finished.")
    return model, losses


def build_model(use_quantum_layer: bool) -> nn.Module:
    if use_quantum_layer:
        return HybridQuantumNet()
    return SimpleNet()


def main() -> None:
    device = get_device()
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
    _, losses = train(model, x, y, epochs=1000, lr=0.01)
    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
