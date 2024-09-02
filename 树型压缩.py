import torch

def compress_byte(byte):
    """
    压缩单个字节。
    
    :param byte: 输入的字节（0-255之间的整数）
    :return: 字节的8位二进制表示
    """
    return [int(b) for b in f'{byte:08b}']

def compress_group(group):
    """
    压缩8个字节组成的组，构建树形结构。
    
    :param group: 包含8个字节的列表
    :return: 压缩后的比特列表，表示树形结构
    """
    if all(byte == 0 for byte in group):
        return [0]  # 根节点为0，表示整个组都是0
    
    result = [1]  # 根节点为1，表示组内至少有一个非0字节
    
    # 添加8个叶子节点，表示每个字节是否为0
    for byte in group:
        result.append(1 if byte != 0 else 0)
    
    # 对于非0的字节，添加其8位二进制表示
    for byte in group:
        if byte != 0:
            result.extend(compress_byte(byte))
    
    return result

def compress_tensor(tensor):
    """
    压缩整个张量，使用树形结构。
    
    :param tensor: 输入的PyTorch张量（dtype=torch.uint8）
    :return: 压缩后的张量和填充的字节数
    """
    if tensor.dim() != 1:
        tensor = tensor.flatten()
    
    data = tensor.tolist()
    padding = (8 - len(data) % 8) % 8
    data.extend([0] * padding)
    
    compressed = []
    for i in range(0, len(data), 8):
        group = data[i:i+8]
        compressed.extend(compress_group(group))
    
    # 确保压缩后的比特数是8的倍数
    if len(compressed) % 8 != 0:
        compressed.extend([0] * (8 - len(compressed) % 8))
    
    byte_list = []
    for i in range(0, len(compressed), 8):
        byte = int(''.join(map(str, compressed[i:i+8])), 2)
        byte_list.append(byte)
    
    compressed_tensor = torch.tensor(byte_list, dtype=torch.uint8)
    
    return compressed_tensor, padding


import torch
import random

def generate_sparse_tensor(size, sparsity):
    """
    生成具有指定稀疏度的张量
    
    :param size: 张量大小
    :param sparsity: 稀疏度（0值的比例）
    :return: 生成的张量
    """
    tensor = torch.randint(1, 256, (size,), dtype=torch.uint8)
    num_zeros = int(size * sparsity)
    zero_indices = random.sample(range(size), num_zeros)
    tensor[zero_indices] = 0
    return tensor

def run_experiment(size, sparsity):
    """
    运行压缩实验
    
    :param size: 原始张量大小
    :param sparsity: 稀疏度
    """
    original_tensor = generate_sparse_tensor(size, sparsity)
    compressed_tensor, padding = compress_tensor(original_tensor)
    
    original_size = len(original_tensor)
    compressed_size = len(compressed_tensor)
    compression_ratio = original_size / compressed_size
    
    print(f"稀疏度: {sparsity*100}%")
    print(f"原始张量大小: {original_size} 字节")
    print(f"压缩后张量大小: {compressed_size} 字节")
    print(f"压缩比: {compression_ratio:.2f}")
    print(f"填充的字节数: {padding}")
    print("------------------------")

# 运行实验
tensor_size = 10000  # 使用更大的张量以获得更稳定的结果

for sparsity in [0.05, 0.10, 0.15]:
    run_experiment(tensor_size, sparsity)
