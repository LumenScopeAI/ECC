import torch
import time

def to_bits(data):
    dtype = data.dtype
    if dtype != torch.uint8:
        data = data.to(torch.uint8)
    bits = torch.zeros(data.shape + (8,), dtype=torch.uint8, device=data.device)
    for i in range(8):
        bits[..., i] = (data >> i) & 1
    return bits.flip(-1)

def from_bits(bits):
    bits = bits.flip(-1)
    result = torch.zeros(bits.shape[:-1], dtype=torch.uint8, device=bits.device)
    for i in range(8):
        result |= bits[..., i].to(torch.uint8) << i
    return result

def compress(input_data, A, B):
    device = input_data.device
    Q, W = input_data.shape
    N = (A * 8) // B
    
    bit_tensor = to_bits(input_data).view(-1)
    
    total_bits = Q * W * 8
    padding = (A * 8 - total_bits % (A * 8)) % (A * 8)
    bit_tensor = torch.nn.functional.pad(bit_tensor, (0, padding))
    
    groups = bit_tensor.view(-1, A * 8)
    
    compressed_data = []
    for group in groups:
        units = group.view(-1, B)
        flags = (units.sum(dim=1) != 0).to(torch.uint8)
        non_zero_units = units[flags.bool()].view(-1)
        
        compressed_group = torch.cat([flags, non_zero_units])
        compressed_data.append(compressed_group)
        compressed_data.append(torch.ones(6, dtype=torch.uint8, device=device))  # Separator
    
    compressed_tensor = torch.cat(compressed_data)
    
    if len(compressed_tensor) % 8 != 0:
        padding = 8 - (len(compressed_tensor) % 8)
        compressed_tensor = torch.nn.functional.pad(compressed_tensor, (0, padding))
    
    return from_bits(compressed_tensor.view(-1, 8)), (Q, W)

def error_recovery(compressed_data):
    print('error_recovery: ', compressed_data)
    compressed_data = insert_err_with_ECC(compressed_data)
    # Implement error recovery logic here
    # This is a placeholder and needs to be implemented
    return compressed_data

@torch.no_grad()
def insert_err_with_ECC(org_bit, ERR_TYPE=1, ERR_RATE=0.001):
    print('org_bit: ', org_bit.shape)
    device = org_bit.device
    N = org_bit.shape[0]
    original_shape = org_bit.shape  # 保存原始形状
    
    # 计算填充
    padding = (64 - N % 64) % 64
    
    # 如果需要，进行填充
    if padding > 0:
        padded_input = torch.cat([org_bit, torch.zeros(padding, dtype=torch.uint8, device=device)])
    else:
        padded_input = org_bit
    
    # 重塑为 (-1, 64)
    reshaped = padded_input.reshape(-1, 64)
    
    # 编码
    encoded = hamming_encode_bytes(reshaped)
    
    # 注入错误
    flip_mask = (torch.rand(encoded.shape, device=device) < ERR_RATE).bool()
    encoded_with_errors = encoded ^ flip_mask.byte()
    
    # 解码并纠错
    corrected, errors_corrected = hamming_decode_and_correct_bytes(encoded_with_errors)
    
    # 重塑回原始形状
    result = corrected.reshape(-1)[:N].reshape(original_shape)
    
    # 计算总错误数
    total_errors = flip_mask.sum().item()
    
    print(f"总错误数: {total_errors}")
    print(f"纠正的错误数: {errors_corrected}")
    print('结果: ', result.shape)
    # 简单比较原始输入和返回结果
    is_identical = torch.all(org_bit == result)
    print(f"ECC处理前后数据是否100%一致: {is_identical}")
    return result

def hamming_encode_bytes(data):
    device = data.device
    encoded = torch.zeros((*data.shape[:-1], 72), dtype=torch.uint8, device=device)
    encoded[..., :64] = data
    
    # 计算校验位
    for i in range(8):
        encoded[..., 64+i] = encoded[..., i::8].sum(dim=-1) % 2
    
    return encoded

def hamming_decode_and_correct_bytes(encoded):
    device = encoded.device
    syndrome = torch.zeros((*encoded.shape[:-1], 8), dtype=torch.uint8, device=device)
    
    for i in range(8):
        syndrome[..., i] = encoded[..., i::8].sum(dim=-1) % 2
    
    error_pos = (syndrome * torch.arange(1, 9, device=device)).sum(dim=-1) - 1
    corrected = encoded[..., :64].clone()
    
    valid_errors = (error_pos >= 0) & (error_pos < 64)
    corrected[valid_errors, error_pos[valid_errors]] ^= 1
    
    errors_corrected = valid_errors.sum().item()
    
    return corrected, errors_corrected


def test_compression(input_data, A, B):
    device = input_data.device
    
    start_time = time.time()
    compressed, original_shape = compress(input_data, A, B)
    compress_time = time.time() - start_time
    
    # 在这里调用 error_recovery
    compressed = error_recovery(compressed)
    
    start_time = time.time()
    decompressed = decompress(compressed, A, B, original_shape)
    decompress_time = time.time() - start_time
    
    compression_ratio = input_data.numel() / compressed.numel()
    is_lossless = torch.all(input_data == decompressed)
    
    print(f"Original shape: {input_data.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Decompressed shape: {decompressed.shape}")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Lossless: {is_lossless}")
    print(f"Compression Time: {compress_time:.4f} seconds")
    print(f"Decompression Time: {decompress_time:.4f} seconds")
    
    # 添加详细的比较
    if not is_lossless:
        mismatch = (input_data != decompressed).sum().item()
        mismatch_percentage = (mismatch / input_data.numel()) * 100
        print(f"Mismatched elements: {mismatch} ({mismatch_percentage:.2f}%)")
        
        # 打印一些不匹配的元素
        mismatch_indices = torch.where(input_data != decompressed)
        num_samples = min(10, len(mismatch_indices[0]))
        for i in range(num_samples):
            idx = tuple(index[i] for index in mismatch_indices)
            print(f"Mismatch at {idx}: Original {input_data[idx].item()}, Decompressed {decompressed[idx].item()}")
    
    return compressed, decompressed

def generate_sparse_data(Q, W, sparsity=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 生成 (Q, W, 8) 的随机二进制数据
    data = torch.randint(0, 2, (Q, W, 8), dtype=torch.uint8, device=device)
    # 应用稀疏性掩码
    mask = torch.rand(Q, W, 8, device=device) < sparsity
    data[mask] = 0
    # 将 8 位压缩成一个 uint8
    compressed_data = from_bits(data)
    return compressed_data

def decompress(compressed_data, A, B, original_shape):
    device = compressed_data.device
    Q, W = original_shape
    N = (A * 8) // B
    
    bit_tensor = to_bits(compressed_data).view(-1)
    
    def normal_decompress():
        decompressed_data = []
        i = 0
        while i < len(bit_tensor):
            if i + 6 <= len(bit_tensor) and torch.all(bit_tensor[i:i+6] == 1):
                i += 6  # Skip separator
                continue
            
            if i + N > len(bit_tensor):
                break  # End of data reached
            
            flags = bit_tensor[i:i+N].to(torch.bool)
            i += N
            
            group = torch.zeros(A * 8, dtype=torch.uint8, device=device)
            group_units = group.view(-1, B)
            
            non_zero_count = flags.sum().item()
            if i + non_zero_count * B > len(bit_tensor):
                break  # Incomplete group, stop decompression
            
            non_zero_data = bit_tensor[i:i + non_zero_count * B].view(-1, B)
            i += non_zero_count * B
            
            group_units[flags] = non_zero_data
            decompressed_data.append(group)
        
        if not decompressed_data:
            return torch.zeros(original_shape, device=device, dtype=torch.uint8)
        
        decompressed_tensor = torch.cat(decompressed_data)
        decompressed_tensor = decompressed_tensor[:Q*W*8]  # Remove padding
        return from_bits(decompressed_tensor.view(-1, 8)).reshape(Q, W)

    def error_recovery_decompress():
        decompressed_data = []
        i = 0
        while i < len(bit_tensor):
            # 错误恢复：检查分隔符
            if i + 5 <= len(bit_tensor) and torch.sum(bit_tensor[i:i+5]) == 5:
                if i + 6 > len(bit_tensor) or bit_tensor[i+5] == 0:
                    if len(decompressed_data) > 0:
                        if len(decompressed_data[-1]) % 2 == 1:
                            decompressed_data[-1][-1] = 1
                        else:
                            bit_tensor[i+5] = 1
                i += 6
                continue
            elif i + 6 <= len(bit_tensor) and torch.sum(bit_tensor[i:i+6]) == 5:
                zero_index = torch.where(bit_tensor[i:i+6] == 0)[0][0]
                bit_tensor[i+zero_index] = 1
                i += 6
                continue
            
            if i + N > len(bit_tensor):
                break  # 数据结束
            
            flags = bit_tensor[i:i+N].to(torch.long)
            i += N
            
            # 错误恢复：检查flags
            k = (A * 8 - N) // B
            t = flags.sum().item()
            if t > k:
                flags[:t-k] = 0
            elif t < k:
                flags[t:k] = 1
            
            group = torch.zeros(A * 8, dtype=torch.uint8, device=device)
            non_zero_count = flags.sum().item()
            
            if i + non_zero_count * B > len(bit_tensor):
                remaining_bits = len(bit_tensor) - i
                non_zero_data = torch.nn.functional.pad(bit_tensor[i:], (0, non_zero_count * B - remaining_bits))
                i = len(bit_tensor)
            else:
                non_zero_data = bit_tensor[i:i + non_zero_count * B]
                i += non_zero_count * B
            
            group_units = group.view(-1, B)
            group_units[flags.bool()] = non_zero_data.view(-1, B)
            
            decompressed_data.append(group)
        
        if not decompressed_data:
            return torch.zeros(original_shape, device=device, dtype=torch.uint8)
        
        decompressed_tensor = torch.cat(decompressed_data)
        if decompressed_tensor.size(0) < Q*W*8:
            padding_size = Q*W*8 - decompressed_tensor.size(0)
            decompressed_tensor = torch.cat([decompressed_tensor, torch.zeros(padding_size, dtype=torch.uint8, device=device)])
        elif decompressed_tensor.size(0) > Q*W*8:
            decompressed_tensor = decompressed_tensor[:Q*W*8]
        
        return from_bits(decompressed_tensor.view(-1, 8)).reshape(Q, W)

    # 首先尝试正常解压
    try:
        result = normal_decompress()
        if result.shape == original_shape:
            return result
    except Exception as e:
        print(f"Normal decompression failed: {e}")

    # 如果正常解压失败或结果形状不匹配，尝试错误恢复解压
    print("Attempting error recovery decompression")
    return error_recovery_decompress()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for Q, W in [(1000, 1000), (10, 10), (5, 5), (2, 2)]:
        print(f"\nTesting with Q={Q}, W={W}")
        A, B = 16, 8  # A in bytes, B in bits
        
        test_data = generate_sparse_data(Q, W)
        print("Original test data:")
        print(test_data)
        
        compressed, decompressed = test_compression(test_data, A, B)
        
        print("Decompressed data:")
        print(decompressed)
        
        # 直接比较原始数据和解压缩后的数据
        is_identical = torch.all(test_data == decompressed)
        print(f"Original and decompressed data are identical: {is_identical}")
        
        if not is_identical:
            mismatch = (test_data != decompressed).sum().item()
            mismatch_percentage = (mismatch / test_data.numel()) * 100
            print(f"Total mismatched elements: {mismatch} ({mismatch_percentage:.2f}%)")
        
        print("------------------------")
