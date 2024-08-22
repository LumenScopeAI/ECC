"""
根据论文内容,我总结如下:

1. 错误类型及概率:

- 单比特错误(SBSE): 73.98%
- 单引脚错误(1 Pin): 0.19% 
- 单字节错误(1 Byte): 22.56%
- 双比特错误(2 Bits): 0.11%
- 三比特错误(3 Bits): 0.03%
- 单拍错误(1 Beat): 0.90%
- 单条目错误(1 Entry): 2.23%

2. 各种错误对存储bit的影响:

- 单比特错误:翻转1个随机比特
例如:
```
原始数据:  1010 1101 0011 0001
错误后:    1010 1101 0011 0101  (第13位翻转)
```

- 单引脚错误:影响2-4个连续比特
例如:
```  
原始数据:  1010 1101 0011 0001
错误后:    1010 1101 0011 1111  (后4位受影响)
```

- 单字节错误:影响8个连续比特
例如:
```
原始数据:  1010 1101 0011 0001 1100 1010
错误后:    1010 1101 1100 0110 1100 1010  (中间8位受影响)
```

- 单拍错误:影响4-64个比特
例如:
```
原始数据:  1010 1101 0011 0001 1100 1010 0101 1111
错误后:    0101 0010 1100 1110 0011 0101 1010 0000  (所有64位都可能受影响)
```

- 单条目错误:影响4-256个比特
例如可能影响一整个32字节的存储条目。

3. GPU存储和读取模式:

- 最小访问粒度为32字节
- 每次读取获取一个32字节的存储条目
- 每个32字节条目分为4个64位的代码字
- 每个代码字包含56位数据和8位ECC校验位
- 读取时,4个代码字并行解码
- 写入时,4个代码字并行编码
- 采用交错布局,将字节错误分散到4个代码字中
"""
# gpu_memory_error_simulator.py

import torch
import random

class GPUMemoryErrorSimulator:
    def __init__(self, device='cuda'):
        self.device = device
        self.error_rates = {
            'SBSE': 0.7398,
            'Pin': 0.0019,
            'Byte': 0.2256,
            'TwoBits': 0.0011,
            'ThreeBits': 0.0003,
            'Beat': 0.0090,
            'Entry': 0.0223
        }

    def hamming_encode(self, data):
        device = data.device
        encoded = torch.zeros((*data.shape[:-1], 8), dtype=torch.uint8, device=device)
        
        # 计算校验位
        encoded = (
            (data[..., 0] ^ data[..., 1] ^ data[..., 3] ^ data[..., 4] ^ data[..., 6]) |
            ((data[..., 2] ^ data[..., 3] ^ data[..., 5] ^ data[..., 6]) << 1) |
            ((data[..., 4] ^ data[..., 5] ^ data[..., 6] ^ data[..., 7]) << 2)
        )
        return encoded

    def hamming_decode_and_correct(self, data, ecc):
        device = data.device
        syndrome = (
            (data[..., 0] ^ data[..., 1] ^ data[..., 3] ^ data[..., 4] ^ data[..., 6] ^ ((ecc & 1) != 0)) |
            ((data[..., 2] ^ data[..., 3] ^ data[..., 5] ^ data[..., 6] ^ ((ecc & 2) != 0)) << 1) |
            ((data[..., 4] ^ data[..., 5] ^ data[..., 6] ^ data[..., 7] ^ ((ecc & 4) != 0)) << 2)
        )
        error_pos = syndrome.unsqueeze(-1) == torch.arange(8, device=device)
        corrected = data.clone()
        corrected[error_pos] ^= 1
        return corrected, error_pos.sum().item()

    def organize_data(self, data):
        """将数据组织成符合GPU存储模式的格式"""
        # 确保数据是32字节的倍数
        pad_size = (32 - (data.numel() % 32)) % 32
        padded_data = torch.cat([data, torch.zeros(pad_size, dtype=torch.uint8, device=self.device)])
        
        # 重塑数据为(N, 4, 8)的形状，其中N是32字节条目的数量
        organized_data = padded_data.view(-1, 4, 8)
        
        # 对每个8字节块进行Hamming编码
        ecc_data = self.hamming_encode(organized_data)
        
        # 将数据和ECC组合
        combined_data = torch.zeros((organized_data.shape[0], 4, 9), dtype=torch.uint8, device=self.device)
        combined_data[..., :8] = organized_data
        combined_data[..., 8] = ecc_data
        
        return combined_data

    def inject_errors(self, data):
        """在数据中注入错误"""
        data = data.clone()
        
        for entry in range(data.shape[0]):
            if random.random() < sum(self.error_rates.values()):
                error_type = random.choices(list(self.error_rates.keys()), 
                                            weights=list(self.error_rates.values()))[0]
                self._inject_error(data[entry], error_type)
        
        return data

    def _inject_error(self, entry, error_type):
        """根据错误类型注入特定错误"""
        if error_type == 'SBSE':
            self._flip_single_bit(entry)
        elif error_type == 'Pin':
            self._flip_pin(entry)
        elif error_type == 'Byte':
            self._flip_byte(entry)
        elif error_type == 'TwoBits':
            self._flip_two_bits(entry)
        elif error_type == 'ThreeBits':
            self._flip_three_bits(entry)
        elif error_type == 'Beat':
            self._flip_beat(entry)
        elif error_type == 'Entry':
            self._flip_entry(entry)

    def _flip_single_bit(self, entry):
        codeword = random.randint(0, 3)
        bit = random.randint(0, 8)
        entry[codeword, bit] ^= 1

    def _flip_pin(self, entry):
        codeword = random.randint(0, 3)
        start_bit = random.randint(0, 5)
        num_bits = random.randint(2, 4)
        for bit in range(start_bit, min(start_bit + num_bits, 9)):
            entry[codeword, bit] ^= 1

    def _flip_byte(self, entry):
        codeword = random.randint(0, 3)
        entry[codeword, :8] ^= 0xFF

    def _flip_two_bits(self, entry):
        codeword = random.randint(0, 3)
        bits = random.sample(range(9), 2)
        for bit in bits:
            entry[codeword, bit] ^= 1

    def _flip_three_bits(self, entry):
        codeword = random.randint(0, 3)
        bits = random.sample(range(9), 3)
        for bit in bits:
            entry[codeword, bit] ^= 1

    def _flip_beat(self, entry):
        codeword = random.randint(0, 3)
        for bit in range(9):
            if random.random() < 0.5:
                entry[codeword, bit] ^= 1

    def _flip_entry(self, entry):
        for codeword in range(4):
            for bit in range(9):
                if random.random() < 0.5:
                    entry[codeword, bit] ^= 1

    def simulate_read(self, data):
        """模拟读取过程"""
        corrected_data = torch.zeros((data.shape[0], 4, 8), dtype=torch.uint8, device=self.device)
        total_errors = 0
        for i in range(4):
            corrected, errors = self.hamming_decode_and_correct(data[:, i, :8], data[:, i, 8])
            corrected_data[:, i, :] = corrected
            total_errors += errors
        return corrected_data.reshape(-1), total_errors

    def simulate_write(self, data):
        """模拟写入过程"""
        N = data.numel() // 32
        reshaped_data = data.reshape(N, 4, 8)
        ecc_data = self.hamming_encode(reshaped_data)
        combined_data = torch.zeros((N, 4, 9), dtype=torch.uint8, device=self.device)
        combined_data[..., :8] = reshaped_data
        combined_data[..., 8] = ecc_data
        return combined_data
