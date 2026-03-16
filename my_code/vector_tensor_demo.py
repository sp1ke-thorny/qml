"""
PyTorch 向量与基础矩阵运算示例

这个脚本演示：
1. 如何创建向量
2. 如何查看张量属性
3. 如何进行基础向量计算
4. 如何进行矩阵乘法
5. 如何使用广播
6. 如何改变张量形状
"""

import torch


def print_section(title: str) -> None:
    """打印分段标题，方便阅读输出。"""
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)


def show_vector_creation() -> None:
    """演示向量的创建与基础属性查看。"""
    print_section("1. 创建向量")

    vector_a = torch.tensor([1.0, 2.0, 3.0])
    vector_zeros = torch.zeros(3)
    vector_ones = torch.ones(3)
    vector_range = torch.arange(1, 6)

    print("使用 torch.tensor 创建向量:")
    print(vector_a)

    print("\n使用 torch.zeros 创建全 0 向量:")
    print(vector_zeros)

    print("\n使用 torch.ones 创建全 1 向量:")
    print(vector_ones)

    print("\n使用 torch.arange 创建等差向量:")
    print(vector_range)

    print_section("2. 查看张量属性")
    print(f"vector_a 的形状 shape: {vector_a.shape}")
    print(f"vector_a 的数据类型 dtype: {vector_a.dtype}")
    print(f"vector_a 所在设备 device: {vector_a.device}")


def show_basic_vector_ops() -> None:
    """演示向量的基础运算。"""
    print_section("3. 向量基础运算")

    vector_a = torch.tensor([1.0, 2.0, 3.0])
    vector_b = torch.tensor([4.0, 5.0, 6.0])
    scalar = 2.0

    print(f"向量 a: {vector_a}")
    print(f"向量 b: {vector_b}")

    print("\n向量加法 a + b:")
    print(vector_a + vector_b)

    print("\n向量减法 a - b:")
    print(vector_a - vector_b)

    print("\n逐元素乘法 a * b:")
    print(vector_a * vector_b)

    print("\n标量乘法 a * 2:")
    print(vector_a * scalar)

    print_section("4. 向量点积")
    dot_result = torch.dot(vector_a, vector_b)
    print("torch.dot(a, b) 的结果:")
    print(dot_result)


def show_matrix_ops() -> None:
    """演示二维张量与矩阵乘法。"""
    print_section("5. 矩阵创建与矩阵乘法")

    matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print("矩阵 A:")
    print(matrix_a)

    print("\n矩阵 B:")
    print(matrix_b)

    print("\n使用 torch.matmul(A, B) 进行矩阵乘法:")
    print(torch.matmul(matrix_a, matrix_b))


def show_broadcast_and_reshape() -> None:
    """演示广播与形状变化。"""
    print_section("6. 广播示例")

    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(f"原始向量: {vector}")
    print("\n向量与标量相加 vector + 10:")
    print(vector + 10)

    print("\n原始矩阵:")
    print(matrix)
    print("\n矩阵与向量相加 matrix + vector:")
    print(matrix + vector)

    print_section("7. 改变张量形状")
    values = torch.arange(1, 7)
    reshaped = values.reshape(2, 3)

    print(f"原始一维张量: {values}")
    print(f"原始形状: {values.shape}")

    print("\n使用 reshape(2, 3) 后:")
    print(reshaped)
    print(f"新形状: {reshaped.shape}")


def main() -> None:
    """按顺序运行所有教学示例。"""
    print("PyTorch 向量与基础矩阵运算入门示例")
    show_vector_creation()
    show_basic_vector_ops()
    show_matrix_ops()
    show_broadcast_and_reshape()


if __name__ == "__main__":
    main()
