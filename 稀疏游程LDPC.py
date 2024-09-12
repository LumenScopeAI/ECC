import torch
import numpy as np
from sklearn.decomposition import DictionaryLearning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class C_LDPC:
    def __init__(self, k, n, m, compression_ratio, error_correction_capability):
        self.k = k
        self.n = n
        self.m = m
        self.compression_ratio = compression_ratio
        self.error_correction_capability = error_correction_capability
        self.D = None

    def train_dictionary(self, training_data):
        """
        使用GPU训练字典D。
        
        输入:
        training_data: torch.Tensor, shape [num_samples, n, 8], dtype=torch.uint8, on GPU
        """
        float_data = training_data.float() / 255.0  # 归一化到 [0, 1]
        reshaped_data = float_data.reshape(-1, self.n * 8)
        
        cpu_data = reshaped_data.cpu().numpy()
        dict_learner = DictionaryLearning(n_components=self.m, fit_algorithm='lars', transform_algorithm='lasso_lars')
        dict_learner.fit(cpu_data)
        
        self.D = torch.from_numpy(dict_learner.components_.T).float().to(device)
        print(f"字典D已训练完成，形状为: {self.D.shape}")

    def sparse_encode(self, x):
        """
        在GPU上进行稀疏编码。
        
        输入:
        x: torch.Tensor, shape [batch_size, n, 8], dtype=torch.uint8, on GPU
        
        输出:
        s: torch.Tensor, shape [batch_size, m], on GPU
        """
        if self.D is None:
            raise ValueError("字典D尚未训练，请先调用train_dictionary方法。")
        
        x_float = x.float().reshape(-1, self.n * 8) / 255.0  # 归一化到 [0, 1]
        s = torch.mm(x_float, self.D)
        
        threshold = torch.mean(torch.abs(s)) + torch.std(torch.abs(s))
        s[torch.abs(s) < threshold] = 0
        
        return s

def unpack_bytes_to_bits(byte_tensor):
    """
    在GPU上将uint8张量解包为bit张量，保持uint8类型。
    
    输入:
    byte_tensor: torch.Tensor, shape [A, B], dtype=torch.uint8, on GPU
    
    输出:
    bit_tensor: torch.Tensor, shape [A, B, 8], dtype=torch.uint8, on GPU
    """
    mask = 2**torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    return (byte_tensor.unsqueeze(-1) & mask).div(mask).to(torch.uint8)

def generate_sparse_bytes(size, sparsity):
    """
    在GPU上生成稀疏的字节数据。
    
    输入:
    size: tuple, 生成张量的形状
    sparsity: float, 非零字节的比例 (0到1之间)
    
    输出:
    sparse_bytes: torch.Tensor, dtype=torch.uint8, on GPU
    """
    sparse_bytes = torch.zeros(size, dtype=torch.uint8, device=device)
    num_elements = np.prod(size)
    num_nonzero = int(sparsity * num_elements)
    indices = torch.randperm(num_elements, device=device)[:num_nonzero]
    sparse_bytes.view(-1)[indices] = torch.randint(1, 256, (num_nonzero,), dtype=torch.uint8, device=device)
    return sparse_bytes

# 在主测试代码部分进行以下修改
if __name__ == "__main__":
    k = 1000
    n = 128
    m = 256
    compression_ratio = 0.9
    error_correction_capability = 1

    codec = C_LDPC(k, n, m, compression_ratio, error_correction_capability)
    
    sparsity = 0.1
    training_data_bytes = generate_sparse_bytes((1000, n), sparsity)
    training_data_bits = unpack_bytes_to_bits(training_data_bytes)
    
    with torch.cuda.device(0):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        codec.train_dictionary(training_data_bits)
        
        test_input_bytes = generate_sparse_bytes((10, n), sparsity)
        test_input_bits = unpack_bytes_to_bits(test_input_bytes)
        sparse_rep = codec.sparse_encode(test_input_bits)
        
        end_time.record()
        torch.cuda.synchronize()
        print(f"GPU执行时间: {start_time.elapsed_time(end_time):.2f} ms")

    # 计算编码前后的比特数
    original_bits = test_input_bits.numel()
    encoded_bits = sparse_rep.numel() * 32  # 假设每个浮点数用32位表示

    print(f"原始输入形状: {test_input_bits.shape}")
    print(f"原始输入比特数: {original_bits}")
    print(f"原始输入非零字节比例: {torch.sum(test_input_bytes != 0).item() / test_input_bytes.numel():.2%}")
    print(f"原始输入非零比特比例: {torch.sum(test_input_bits != 0).item() / test_input_bits.numel():.2%}")
    print(f"稀疏表示形状: {sparse_rep.shape}")
    print(f"编码后比特数: {encoded_bits}")
    print(f"压缩比率: {encoded_bits / original_bits:.2f}")
    print(f"稀疏表示非零元素比例: {torch.sum(sparse_rep != 0).item() / sparse_rep.numel():.2%}")

    # 计算实际压缩后的比特数（只考虑非零元素）
    non_zero_elements = torch.sum(sparse_rep != 0).item()
    compressed_bits = non_zero_elements * (32 + 8)  # 32位用于值，8位用于索引（假设）
    print(f"考虑稀疏性后的压缩比特数: {compressed_bits}")
    print(f"考虑稀疏性后的压缩比率: {compressed_bits / original_bits:.2f}")
