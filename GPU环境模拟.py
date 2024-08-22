import torch
import torch.nn as nn
import random

"""
### GPU

- 单比特错误:影响单个DRAM比特位 (73.98%)
- 多比特错误:影响多个DRAM比特位,包括:
  - 字节对齐的多比特错误:影响一个字节内的多个比特 (22.56%)
  - 非字节对齐的多比特错误:可能影响多达64位数据 (0.90%)
- 行错误:影响DRAM的一整行或部分行 (包含在多比特错误中)
- 列错误:影响DRAM的一列或多列 (包含在多比特错误中)
- 单bank错误:影响DRAM的一个bank内的多行 (包含在多比特错误中)
- 多bank错误:影响DRAM的多个bank (包含在多比特错误中)
- 全芯片错误:影响整个DRAM芯片 (2.23%)
- 引脚错误:影响连接DRAM的单个引脚 (0.19%)
- 通道错误:影响整个DRAM通道 (未明确提及比例，忽略)

1. 大小:
   - 论文中提到使用的是32GB HBM2内存。
   - HBM2每个堆叠通常有8个512MB的通道，总计4GB。

2. 行/列大小:
   - 一个行激活命令会带来2KB的数据到行缓冲区。
   - 每次读取访问32B (256位)。
   - 假设每个bank有16384行、128列，那么:
     - 一行 = 2KB = 16384位
     - 一列 = 32B = 256位

3. 常见大小:
   - HBM2常见容量有4GB、8GB、16GB、32GB等。
"""


class GPUErrorModel(nn.Module):
    def __init__(self, err_rate):
        """
        初始化GPU错误模型
        :param err_rate: 每个比特发生错误的概率
        """
        super(GPUErrorModel, self).__init__()
        self.err_rate = err_rate
        # 定义不同类型错误的概率分布
        self.error_distribution = {
            'single_bit': 0.7398,              # 单比特错误
            'multi_bit_byte_aligned': 0.2256,  # 字节对齐的多比特错误
            'multi_bit_non_byte_aligned': 0.0090,  # 非字节对齐的多比特错误
            'whole_chip': 0.0223,              # 全芯片错误
            'pin': 0.0019                      # 引脚错误
        }

    def inject_single_bit_error(self, tensor):
        """
        注入单比特错误
        :param tensor: 输入张量
        :return: 注入错误后的张量
        """
        flat = tensor.view(-1)  # 将张量展平为一维
        idx = random.randint(0, flat.numel() - 1)  # 随机选择一个元素索引
        bit = random.randint(0, 7)  # 随机选择要翻转的比特位（0-7）
        flat[idx] = flat[idx] ^ (1 << bit)  # 使用异或操作翻转选定的比特
        return tensor

    def inject_multi_bit_byte_aligned(self, tensor):
        """
        注入字节对齐的多比特错误
        :param tensor: 输入张量
        :return: 注入错误后的张量
        """
        flat = tensor.view(-1)  # 将张量展平为一维
        idx = random.randint(0, flat.numel() - 1)  # 随机选择一个字节
        num_bits = random.randint(2, 8)  # 随机选择要翻转的比特数（2-8）
        mask = sum(1 << i for i in random.sample(range(8), num_bits))  # 创建随机比特掩码
        flat[idx] = flat[idx] ^ mask  # 使用异或操作应用掩码
        return tensor

    def inject_multi_bit_non_byte_aligned(self, tensor):
        """
        注入非字节对齐的多比特错误
        :param tensor: 输入张量
        :return: 注入错误后的张量
        """
        flat = tensor.view(-1)  # 将张量展平为一维
        start_idx = random.randint(0, flat.numel() - 8)  # 随机选择起始字节
        num_bits = random.randint(2, 64)  # 随机选择要翻转的比特数（2-64）
        for _ in range(num_bits):
            bit_idx = random.randint(0, 63)  # 在64位范围内随机选择一个比特
            byte_idx = start_idx + bit_idx // 8  # 计算目标字节索引
            bit_in_byte = bit_idx % 8  # 计算比特在字节内的位置
            flat[byte_idx] = flat[byte_idx] ^ (1 << bit_in_byte)  # 翻转选定的比特
        return tensor

    def inject_whole_chip_error(self, tensor):
        """
        注入全芯片错误，将整个张量替换为随机值
        :param tensor: 输入张量
        :return: 完全随机化的张量
        """
        return torch.randint_like(tensor, 0, 256)  # 生成与输入相同形状的随机张量

    def inject_pin_error(self, tensor):
        """
        注入引脚错误，翻转所有元素的同一个比特位
        :param tensor: 输入张量
        :return: 注入错误后的张量
        """
        bit = random.randint(0, 7)  # 随机选择一个比特位
        mask = 1 << bit  # 创建掩码
        return tensor ^ mask  # 对整个张量应用异或操作

    def forward(self, x):
        """
        模型的前向传播，对输入张量注入错误
        :param x: 输入张量
        :return: 注入错误后的张量
        """
        total_bits = x.numel() * 8  # 计算总比特数（每个元素8位）
        error_bits = int(total_bits * self.err_rate)  # 计算需要注入错误的比特数

        for _ in range(error_bits):
            # 根据错误分布随机选择错误类型
            error_type = random.choices(
                list(self.error_distribution.keys()),
                weights=list(self.error_distribution.values())
            )[0]
            
            # 根据选择的错误类型调用相应的错误注入函数
            if error_type == 'single_bit':
                x = self.inject_single_bit_error(x)
            elif error_type == 'multi_bit_byte_aligned':
                x = self.inject_multi_bit_byte_aligned(x)
            elif error_type == 'multi_bit_non_byte_aligned':
                x = self.inject_multi_bit_non_byte_aligned(x)
            elif error_type == 'whole_chip':
                x = self.inject_whole_chip_error(x)
            elif error_type == 'pin':
                x = self.inject_pin_error(x)

        return x

# 使用示例
err_rate = 1e-6  # 设置错误率为每百万比特1个错误
model = GPUErrorModel(err_rate)

# 创建一个大小为10000x10000的随机uint8张量
input_tensor = torch.randint(0, 256, (10000, 10000), dtype=torch.uint8)

# 对输入张量进行错误注入
output_tensor = model(input_tensor)

# 打印结果统计
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
print(f"Number of different elements: {(input_tensor != output_tensor).sum().item()}")
