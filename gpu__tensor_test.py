import torch

# 1. 优雅地定义计算设备 (这是所有 PyTorch 项目的标准起手式)
# 如果有显卡就用 cuda，没有就退回 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前被选中的计算设备是: {device}")

# 2. 在 CPU 上创建一个张量 (数据)
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("\n--- 搬运前 ---")
print("张量所在位置:", tensor_cpu.device)

# 3. 将张量搬运到 RTX 4060 上！
tensor_gpu = tensor_cpu.to(device)
print("\n--- 搬运后 ---")
print("张量所在位置:", tensor_gpu.device)

# 4. 在 GPU 上执行矩阵乘法运算 (这将会非常快)
result = tensor_gpu @ tensor_gpu  # @ 符号代表矩阵乘法
print("\nGPU 上的计算结果:\n", result)