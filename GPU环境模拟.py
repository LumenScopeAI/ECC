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
import torch
import torch.nn as nn
import random
import math
import multiprocessing
from functools import partial

class HBM2Stack:
    def __init__(self, channels=8, channel_size_mb=512, banks_per_channel=16, rows_per_bank=16384, cols_per_row=128):
        """
        初始化 HBM2 内存堆栈
        :param channels: HBM2 堆栈中的通道数
        :param channel_size_mb: 每个通道的大小（MB）
        :param banks_per_channel: 每个通道中的 bank 数量
        :param rows_per_bank: 每个 bank 中的行数
        :param cols_per_row: 每行中的列数
        """
        self.channels = channels
        self.channel_size_mb = channel_size_mb
        self.banks_per_channel = banks_per_channel
        self.rows_per_bank = rows_per_bank
        self.cols_per_row = cols_per_row
        
        # 初始化内存结构：5维列表 [通道][bank][行][列][位]
        self.memory = [[[[[0 for _ in range(256)] for _ in range(cols_per_row)]
                         for _ in range(rows_per_bank)]
                        for _ in range(banks_per_channel)]
                       for _ in range(channels)]
        
        # 计算总比特数和已使用的比特数
        self.total_bits = channels * banks_per_channel * rows_per_bank * cols_per_row * 256
        self.used_bits = 0

    def store_tensor(self, tensor, start_bit=0):
        """
        将张量存储到 HBM2 内存中
        :param tensor: 要存储的张量
        :param start_bit: 开始存储的比特位置
        :return: 已使用的比特数
        """
        flat = tensor.view(-1)
        idx = 0
        for c in range(self.channels):
            for b in range(self.banks_per_channel):
                for r in range(self.rows_per_bank):
                    for col in range(self.cols_per_row):
                        for bit in range(256):
                            absolute_bit = (c * self.banks_per_channel * self.rows_per_bank * self.cols_per_row * 256 +
                                            b * self.rows_per_bank * self.cols_per_row * 256 +
                                            r * self.cols_per_row * 256 +
                                            col * 256 + bit)
                            if absolute_bit < start_bit:
                                continue
                            if idx < flat.numel():
                                self.memory[c][b][r][col][bit] = (flat[idx] >> (bit % 8)) & 1
                                if bit % 8 == 7:
                                    idx += 1
                                self.used_bits += 1
                            else:
                                return self.used_bits
        return self.used_bits

    def retrieve_tensor(self, shape, start_bit=0, end_bit=None):
        """
        从 HBM2 内存中检索张量
        :param shape: 要检索的张量形状
        :param start_bit: 开始检索的比特位置
        :param end_bit: 结束检索的比特位置（如果为None，则检索到已使用的比特）
        :return: 检索出的张量
        """
        if end_bit is None:
            end_bit = self.used_bits
        flat = torch.zeros(shape).view(-1)
        idx = 0
        for c in range(self.channels):
            for b in range(self.banks_per_channel):
                for r in range(self.rows_per_bank):
                    for col in range(self.cols_per_row):
                        byte_val = 0
                        for bit in range(256):
                            absolute_bit = (c * self.banks_per_channel * self.rows_per_bank * self.cols_per_row * 256 +
                                            b * self.rows_per_bank * self.cols_per_row * 256 +
                                            r * self.cols_per_row * 256 +
                                            col * 256 + bit)
                            if absolute_bit < start_bit:
                                continue
                            if absolute_bit >= end_bit:
                                return flat.view(shape)
                            if idx < flat.numel():
                                byte_val |= self.memory[c][b][r][col][bit] << (bit % 8)
                                if bit % 8 == 7:
                                    flat[idx] = byte_val
                                    idx += 1
                                    byte_val = 0
                            else:
                                return flat.view(shape)
        return flat.view(shape)

class MultiHBM2Stack:
    def __init__(self):
        """
        初始化多个 HBM2 堆栈的管理器
        """
        self.stacks = [HBM2Stack()]
        self.current_stack = 0
        self.current_bit = 0

    def store_tensor(self, tensor):
        """
        将张量存储到多个 HBM2 堆栈中
        :param tensor: 要存储的张量
        """
        flat = tensor.view(-1)
        remaining_bits = flat.numel() * 8

        while remaining_bits > 0:
            if self.current_stack >= len(self.stacks):
                self.stacks.append(HBM2Stack())

            bits_stored = self.stacks[self.current_stack].store_tensor(flat, self.current_bit)
            bits_stored_this_round = bits_stored - self.current_bit
            remaining_bits -= bits_stored_this_round

            if bits_stored == self.stacks[self.current_stack].total_bits:
                self.current_stack += 1
                self.current_bit = 0
            else:
                self.current_bit = bits_stored

            if remaining_bits > 0:
                flat = flat[bits_stored_this_round // 8:]

    def retrieve_tensor(self, shape):
        """
        从多个 HBM2 堆栈中检索张量
        :param shape: 要检索的张量形状
        :return: 检索出的张量
        """
        total_bits = math.prod(shape) * 8
        result = torch.zeros(shape).view(-1)
        retrieved_bits = 0

        for stack_idx, stack in enumerate(self.stacks):
            if stack_idx == self.current_stack:
                end_bit = self.current_bit
            else:
                end_bit = stack.used_bits

            start_bit = 0 if stack_idx == 0 else 0
            tensor_part = stack.retrieve_tensor(shape, start_bit, end_bit)
            part_bits = tensor_part.numel() * 8

            if retrieved_bits + part_bits > total_bits:
                part_bits = total_bits - retrieved_bits

            result[retrieved_bits // 8:(retrieved_bits + part_bits) // 8] = tensor_part.view(-1)[:part_bits // 8]
            retrieved_bits += part_bits

            if retrieved_bits >= total_bits:
                break

        return result.view(shape)

class GPUErrorModel(nn.Module):
    def __init__(self, err_rate):
        """
        初始化 GPU 错误模型
        :param err_rate: 每个比特发生错误的概率
        """
        super(GPUErrorModel, self).__init__()
        self.err_rate = err_rate
        self.error_distribution = {
            'single_bit': 0.7398,
            'multi_bit_byte_aligned': 0.2256,
            'multi_bit_non_byte_aligned': 0.0090,
            'whole_chip': 0.0223,
            'pin': 0.0019,
            'row': 0.0007,
            'column': 0.0005,
            'bank': 0.0002
        }
        self.multi_hbm2 = MultiHBM2Stack()

    def inject_errors_parallel(self, stack):
        """
        对单个 HBM2Stack 并行注入错误
        :param stack: HBM2Stack 实例
        """
        total_bits = stack.used_bits
        error_bits = int(total_bits * self.err_rate)

        for _ in range(error_bits):
            error_type = random.choices(list(self.error_distribution.keys()),
                                        weights=list(self.error_distribution.values()))[0]
            
            if error_type == 'single_bit':
                self.inject_single_bit_error(stack)
            elif error_type == 'multi_bit_byte_aligned':
                self.inject_multi_bit_byte_aligned(stack)
            elif error_type == 'multi_bit_non_byte_aligned':
                self.inject_multi_bit_non_byte_aligned(stack)
            elif error_type == 'whole_chip':
                self.inject_whole_chip_error(stack)
            elif error_type == 'pin':
                self.inject_pin_error(stack)
            elif error_type == 'row':
                self.inject_row_error(stack)
            elif error_type == 'column':
                self.inject_column_error(stack)
            elif error_type == 'bank':
                self.inject_bank_error(stack)

    def inject_errors(self):
        """
        并行注入错误到所有 HBM2Stack
        """
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(self.inject_errors_parallel, self.multi_hbm2.stacks)

    def inject_single_bit_error(self, stack):
        """在随机位置注入单比特错误"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        r = random.randint(0, stack.rows_per_bank - 1)
        col = random.randint(0, stack.cols_per_row - 1)
        bit = random.randint(0, 255)
        stack.memory[c][b][r][col][bit] ^= 1

    def inject_multi_bit_byte_aligned(self, stack):
        """在随机字节内注入多比特错误"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        r = random.randint(0, stack.rows_per_bank - 1)
        col = random.randint(0, stack.cols_per_row - 1)
        byte = random.randint(0, 31)  # 32 bytes per column
        num_bits = random.randint(2, 8)
        for _ in range(num_bits):
            bit = random.randint(byte * 8, (byte + 1) * 8 - 1)
            stack.memory[c][b][r][col][bit] ^= 1

    def inject_multi_bit_non_byte_aligned(self, stack):
        """注入跨字节的多比特错误"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        r = random.randint(0, stack.rows_per_bank - 1)
        col_start = random.randint(0, stack.cols_per_row - 1)
        num_bits = random.randint(2, 64)
        for _ in range(num_bits):
            col = (col_start + random.randint(0, 7)) % stack.cols_per_row
            bit = random.randint(0, 255)
            stack.memory[c][b][r][col][bit] ^= 1

    def inject_whole_chip_error(self, stack):
        """注入整个芯片的随机错误"""
        c = random.randint(0, stack.channels - 1)
        for b in range(stack.banks_per_channel):
            for r in range(stack.rows_per_bank):
                for col in range(stack.cols_per_row):
                    for bit in range(256):
                        stack.memory[c][b][r][col][bit] = random.randint(0, 1)

    def inject_pin_error(self, stack):
        """注入引脚错误，影响所有数据的特定位"""
        bit = random.randint(0, 255)
        for c in range(stack.channels):
            for b in range(stack.banks_per_channel):
                for r in range(stack.rows_per_bank):
                    for col in range(stack.cols_per_row):
                        stack.memory[c][b][r][col][bit] ^= 1

    def inject_row_error(self, stack):
        """注入行错误，影响随机一行的所有数据"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        r = random.randint(0, stack.rows_per_bank - 1)
        for col in range(stack.cols_per_row):
            for bit in range(256):
                stack.memory[c][b][r][col][bit] = random.randint(0, 1)

    def inject_column_error(self, stack):
        """注入列错误，影响随机一列的所有数据"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        col = random.randint(0, stack.cols_per_row - 1)
        for r in range(stack.rows_per_bank):
            for bit in range(256):
                stack.memory[c][b][r][col][bit] = random.randint(0, 1)

    def inject_bank_error(self, stack):
        """注入bank错误，影响随机一个bank的所有数据"""
        c = random.randint(0, stack.channels - 1)
        b = random.randint(0, stack.banks_per_channel - 1)
        for r in range(stack.rows_per_bank):
            for col in range(stack.cols_per_row):
                for bit in range(256):
                    stack.memory[c][b][r][col][bit] = random.randint(0, 1)

    def forward(self, x):
        """
        模型的前向传播，存储张量、注入错误并检索结果
        :param x: 输入张量
        :return: 注入错误后的张量
        """
        self.multi_hbm2.store_tensor(x)
        self.inject_errors()
        return self.multi_hbm2.retrieve_tensor(x.shape)

def run_experiment(model, input_tensors):
    """
    运行实验，对输入张量进行错误注入并分析结果
    :param model: GPUErrorModel 实例
    :param input_tensors: 输入张量列表
    :return: 注入错误后的张量列表
    """
    total_different_elements = 0
    total_elements = 0
    output_tensors = []

    for tensor in input_tensors:
        output_tensor = model(tensor)
        output_tensors.append(output_tensor)
        different_elements = (tensor != output_tensor).sum().item()
        total_different_elements += different_elements
        total_elements += tensor.numel()

        print(f"Input shape: {tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        print(f"Number of different elements: {different_elements}")
        print(f"Percentage of different elements: {different_elements / tensor.numel() * 100:.6f}%")
        print("---")

    overall_error_rate = total_different_elements / total_elements
    print(f"Overall error rate: {overall_error_rate * 100:.6f}%")
    print(f"Number of HBM2 stacks used: {len(model.multi_hbm2.stacks)}")

    return output_tensors

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    err_rate = 1e-6  # 每个比特发生错误的概率
    model = GPUErrorModel(err_rate)

    # 创建多个不同大小的随机uint8张量
    input_tensors = [
        torch.randint(0, 256, (1000, 1000), dtype=torch.uint8),
        torch.randint(0, 256, (2000, 2000), dtype=torch.uint8),
        torch.randint(0, 256, (3000, 3000), dtype=torch.uint8),
        torch.randint(0, 256, (4000, 4000), dtype=torch.uint8),
    ]

    output_tensors = run_experiment(model, input_tensors)

    # 对比输入和输出张量
    for i, (input_tensor, output_tensor) in enumerate(zip(input_tensors, output_tensors)):
        diff = (input_tensor != output_tensor).float().mean().item()
        print(f"Tensor {i+1} difference ratio: {diff:.6f}")

        # 找出错误位置
        error_positions = (input_tensor != output_tensor).nonzero()
        print(f"First 10 error positions for tensor {i+1}: {error_positions[:10]}")

        # 分析错误模式
        error_values = output_tensor[input_tensor != output_tensor]
        original_values = input_tensor[input_tensor != output_tensor]
        print(f"First 10 error values for tensor {i+1}:")
        for orig, err in zip(original_values[:10], error_values[:10]):
            print(f"Original: {orig.item()}, Error: {err.item()}")

        print("---")
