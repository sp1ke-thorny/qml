def generate_fibonacci(n):
    """
    生成并返回前 n 个斐波那契数字的列表。
    """
    # 如果请求的数量小于等于 0，返回空列表
    if n <= 0:
        return []
    # 如果只需要 1 个数字，直接返回 [0]
    elif n == 1:
        return [0]
    
    # 初始化前两个数字
    sequence = [0, 1]
    
    # 使用循环计算后续的数字
    for i in range(2, n):
        next_number = sequence[-1] + sequence[-2]
        sequence.append(next_number)
        
    return sequence

# === 运行测试 ===
# 我们设定想要获取前 10 个数字
terms = 10
result = generate_fibonacci(terms)

print(f"前 {terms} 个斐波那契数字是: {result}")