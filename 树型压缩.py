import torch
import random
import math

def generate_sparse_tensor(size, bits_per_unit):
    """
    生成具有指定分布的张量
    
    :param size: 张量大小（字节数）
    :param bits_per_unit: 每个判断单位的位数
    :return: 生成的张量
    """
    total_bits = size * 8
    total_units = total_bits // bits_per_unit

    # 根据 bits_per_unit 动态生成分布
    max_value = 2**bits_per_unit
    distribution = [0] * max_value
    distribution[0] = int(0.69 * 10**9)  # 约69%的概率为0
    remaining = 10**9 - distribution[0]
    for i in range(1, max_value):
        distribution[i] = remaining // (max_value - 1)

    probabilities = [d / sum(distribution) for d in distribution]

    units = random.choices(range(max_value), weights=probabilities, k=total_units)
    
    bytes_list = []
    for i in range(0, len(units), 8 // bits_per_unit):
        byte = 0
        for j in range(8 // bits_per_unit):
            if i + j < len(units):
                byte |= units[i+j] << (8 - (j+1)*bits_per_unit)
        bytes_list.append(byte)

    return torch.tensor(bytes_list, dtype=torch.uint8)

def byte_to_bits(byte):
    """将一个字节转换为8位的二进制字符串表示"""
    return format(byte, '08b')

def bits_to_string(bits):
    """将比特列表转换为字符串"""
    return ''.join(map(str, bits))

def print_binary(original, compressed, bytes_per_group):
    """打印原始和压缩后的二进制表示"""
    print(f"原始二进制表示（{bytes_per_group * 2}字节）:")
    for i in range(bytes_per_group * 2):
        print(byte_to_bits(original[i]), end=' ')
        if i == bytes_per_group - 1:
            print()
    print("\n压缩后的二进制表示（两棵树）:")
    
    # 找到第一棵树的结束位置
    first_tree_end = len(compressed) // 2
    
    # 打印第一棵树
    first_tree = bits_to_string(compressed[:first_tree_end])
    print(' '.join(first_tree[i:i+8] for i in range(0, len(first_tree), 8)))
    
    # 打印第二棵树
    second_tree = bits_to_string(compressed[first_tree_end:])
    print(' '.join(second_tree[i:i+8] for i in range(0, len(second_tree), 8)))

def compress_unit(unit, bits_per_unit):
    """
    压缩一个单位。
    
    :param unit: 输入的单位（0到2^bits_per_unit-1之间的整数）
    :param bits_per_unit: 每个判断单位的位数
    :return: 压缩后的比特列表
    """
    if unit == 0:
        return [0]
    return [1] + [int(b) for b in format(unit, f'0{bits_per_unit}b')]

def compress_group(group, bits_per_unit):
    """
    压缩一组字节，构建树形结构。
    
    :param group: 包含bytes_per_group个字节的列表
    :param bits_per_unit: 每个判断单位的位数
    :return: 压缩后的比特列表，表示树形结构
    """
    result = []
    for byte in group:
        for i in range(8 // bits_per_unit - 1, -1, -1):
            unit = (byte >> (i*bits_per_unit)) & ((1 << bits_per_unit) - 1)
            result.extend(compress_unit(unit, bits_per_unit))
    return result

def compress_tensor(tensor, bytes_per_group, bits_per_unit):
    """
    压缩整个张量，使用树形结构。
    
    :param tensor: 输入的PyTorch张量（dtype=torch.uint8）
    :param bytes_per_group: 每组的字节数
    :param bits_per_unit: 每个判断单位的位数
    :return: 压缩后的张量和填充的字节数
    """
    if tensor.dim() != 1:
        tensor = tensor.flatten()
    
    data = tensor.tolist()
    padding = (bytes_per_group - len(data) % bytes_per_group) % bytes_per_group
    data.extend([0] * padding)
    
    compressed = []
    for i in range(0, len(data), bytes_per_group):
        group = data[i:i+bytes_per_group]
        compressed.extend(compress_group(group, bits_per_unit))
    
    # 确保压缩后的比特数是8的倍数
    if len(compressed) % 8 != 0:
        compressed.extend([0] * (8 - len(compressed) % 8))
    
    byte_list = []
    for i in range(0, len(compressed), 8):
        byte = int(''.join(map(str, compressed[i:i+8])), 2)
        byte_list.append(byte)
    
    compressed_tensor = torch.tensor(byte_list, dtype=torch.uint8)
    
    return compressed_tensor, padding, compressed

def run_experiment(original_tensor, bytes_per_group, bits_per_unit):
    """
    运行压缩实验
    
    :param original_tensor: 原始张量
    :param bytes_per_group: 每组的字节数
    :param bits_per_unit: 每个判断单位的位数
    """
    compressed_tensor, padding, compressed_bits = compress_tensor(original_tensor, bytes_per_group, bits_per_unit)
    
    original_size = len(original_tensor)
    compressed_size = len(compressed_tensor)
    compression_ratio = original_size / compressed_size
    
    print(f"参数设置：每组 {bytes_per_group} 字节，每单位 {bits_per_unit} 位")
    print(f"原始张量大小: {original_size} 字节")
    print(f"压缩后张量大小: {compressed_size} 字节")
    print(f"压缩比: {compression_ratio:.2f}")
    print(f"填充的字节数: {padding}")
    
    # 打印两组的原始和压缩后的二进制表示
    print_binary(original_tensor[:bytes_per_group*2], 
                 compressed_bits[:sum(len(compress_group(original_tensor[i:i+bytes_per_group], bits_per_unit)) 
                                      for i in range(0, bytes_per_group*2, bytes_per_group))],
                 bytes_per_group)
    
    print("------------------------")

def main():
    tensor_size = 10000  # 使用更大的张量以获得更稳定的结果
    bits_per_unit_for_generation = 2  # 用于生成张量的 bits_per_unit

    # 生成原始张量
    original_tensor = generate_sparse_tensor(tensor_size, bits_per_unit_for_generation)

    # 测试不同的参数组合
    parameter_sets = [
        (8, 2),  # 8字节分组，2位单位
        (4, 1),  # 4字节分组，1位单位
        (16, 4), # 16字节分组，4位单位
        (8, 1),  # 8字节分组，1位单位
        (8, 4),  # 8字节分组，4位单位
        (8, 8),  # 8字节分组，8位单位
    ]

    for bytes_per_group, bits_per_unit in parameter_sets:
        run_experiment(original_tensor, bytes_per_group, bits_per_unit)

if __name__ == "__main__":
    main()
