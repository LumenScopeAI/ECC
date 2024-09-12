from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from pyldpc import make_ldpc, encode, get_message

# ===========================全局变量===========================
compresser_type = 'Sparse'
bytes_per_group = 16
bits_per_unit = 8
ERR_TYPE = 1
ERR_RATE = 1e-4
code_rate = 0.5
# 全局变量来记录压缩比
compression_ratio = 0

model_path = "/root/autodl-tmp/ECC_LLM/models/Qwen2-0.5B-Instruct"


# ===========================创新压缩===========================

 
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

def Sparse_compress(input_data, A, B):
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

def error_recovery(compressed_data, ERR_TYPE, ERR_RATE):
    print('error_recovery: ', compressed_data)
    compressed_data = insert_err_with_ECC(compressed_data, ERR_TYPE, ERR_RATE)
    return compressed_data

def Sparse_decompress(compressed_data, A, B, original_shape):
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
    
@torch.no_grad()
def Sparse(org_tensor, bytes_per_group, bits_per_unit):
    global compression_ratio, ERR_TYPE, ERR_TYPE

    # 压缩
    compressed, original_shape = Sparse_compress(org_tensor, bytes_per_group, bits_per_unit)
    
    # ECC处理
    compressed_with_ecc = error_recovery(compressed, ERR_TYPE, ERR_RATE)
    
    # 解压缩
    decompressed = Sparse_decompress(compressed_with_ecc, bytes_per_group, bits_per_unit, original_shape)
    
    # 计算压缩比并更新全局变量
    compression_ratio = org_tensor.numel() / compressed.numel()
    
    # 检查是否无损
    is_lossless = torch.all(org_tensor == decompressed)
    
    return compressed, decompressed, compression_ratio, is_lossless

# ===========================错误注入===========================

@torch.no_grad()
def insert_err_without_ECC(org_bit, ERR_TYPE, ERR_RATE):
    print(org_bit.dtype)
    print(org_bit.shape)
    
    device = org_bit.device
    input_shape = org_bit.shape
    
    # 生成与输入形状相同的随机概率矩阵，但在最后一个维度上增加8位
    rand_probs = torch.rand(*input_shape, 8, device=device)
    
    # 生成错误掩码
    error_mask = (rand_probs < ERR_RATE).to(torch.uint8)
    
    # 将 org_bit 展开为位级别的张量
    org_bit_expanded = org_bit.unsqueeze(-1).bitwise_and(torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device))
    org_bit_expanded = (org_bit_expanded != 0).to(torch.uint8)
    
    # 应用错误掩码
    flipped_bits = org_bit_expanded ^ error_mask
    
    # 将结果压缩回 uint8
    result = torch.sum(flipped_bits * torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device), dim=-1).to(torch.uint8)
    
    # 计算总错误数
    total_errors = error_mask.sum().item()
    
    print(f"Total errors: {total_errors}")
    print(f"Corrected errors: 0")  # 这个函数不进行纠错，所以纠正的错误数始终为0
    
    return result

@torch.no_grad()
def insert_err_with_ECC(org_bit, ERR_TYPE, ERR_RATE):
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


# ===========================树型压缩===========================
# 
@torch.no_grad()
def Tree(org_tensor, bytes_per_group, bits_per_unit):
    global compression_ratio
    def tree_compress(org_tensor, bytes_per_group, bits_per_unit):
        device = org_tensor.device
        
        # Step 1: Flatten the tensor
        flattened = org_tensor.reshape(-1)
        
        # Pad the flattened tensor to make its length divisible by bytes_per_group
        pad_size = (bytes_per_group - flattened.size(0) % bytes_per_group) % bytes_per_group
        if pad_size > 0:
            flattened = torch.cat([flattened, torch.zeros(pad_size, dtype=torch.uint8, device=device)])
        
        # Step 2: Compress using tree algorithm
        compressed = []
        for i in range(0, flattened.size(0), bytes_per_group):
            group = flattened[i:i+bytes_per_group]
            if torch.all(group == 0):
                compressed.append(0)
            else:
                compressed.append(1)
                for byte in group:
                    byte_bits = [int(b) for b in f'{byte.item():08b}']
                    for j in range(0, 8, bits_per_unit):
                        bits = byte_bits[j:j+bits_per_unit]
                        if all(bit == 0 for bit in bits):
                            compressed.append(0)
                        else:
                            compressed.append(1)
                            compressed.extend(bits)
        
        # Convert compressed to tensor
        compressed_tensor = torch.tensor(compressed, dtype=torch.uint8, device=device)
        return compressed_tensor, pad_size

    def tree_decompress(compressed_tensor, original_shape, bytes_per_group, bits_per_unit, pad_size=0):
        device = compressed_tensor.device
        compressed = compressed_tensor.tolist()
        
        # Step 4: Decompress
        decompressed = []
        i = 0
        while i < len(compressed):
            if compressed[i] == 0:
                decompressed.extend([0] * bytes_per_group)
                i += 1
            else:
                group = [0] * bytes_per_group
                i += 1
                for byte_idx in range(bytes_per_group):
                    byte = 0
                    for j in range(0, 8, bits_per_unit):
                        if compressed[i] == 0:
                            i += 1
                        else:
                            i += 1
                            for k in range(bits_per_unit):
                                if i + k < len(compressed):
                                    byte |= compressed[i+k] << (7 - j - k)
                            i += bits_per_unit
                    group[byte_idx] = byte
                decompressed.extend(group)
        
        # Remove padding
        decompressed = decompressed[:-pad_size] if pad_size > 0 else decompressed
        
        # Reshape decompressed to match original shape
        decompressed_tensor = torch.tensor(decompressed, dtype=torch.uint8, device=device).reshape(original_shape)
        
        return decompressed_tensor

    compressed_tensor, pad_size = tree_compress(org_tensor, bytes_per_group=bytes_per_group, bits_per_unit=bits_per_unit)
    global ERR_TYPE, ERR_RATE
    # 注入错误（前2bit压缩后）
    # print('compressed_tensor.shape: ',compressed_tensor.shape)
    compressed_tensor = insert_err_with_ECC(compressed_tensor,ERR_TYPE, ERR_RATE)
    
    decompressed_tensor = tree_decompress(compressed_tensor, org_tensor.shape, bytes_per_group=bytes_per_group, bits_per_unit=bits_per_unit, pad_size=pad_size)
    
    # Calculate compression ratio
    original_size = org_tensor.numel() * org_tensor.element_size()
    compressed_size = compressed_tensor.numel() * compressed_tensor.element_size()
    compression_ratio = compressed_size / original_size
    # Verify decompression using assert
    assert torch.all(org_tensor == decompressed_tensor), "Decompression failed: mismatch with original data"

    return decompressed_tensor


# ===========================特殊编码===========================

@torch.no_grad()
def group_and_pad(quantized_tensor):
    # 将输入张量展平
    flat_tensor = quantized_tensor.view(-1)
    
    # 计算需要填充的数量
    pad_size = (64 - (flat_tensor.size(0) % 64)) % 64
    
    # 填充张量
    padded_tensor = torch.nn.functional.pad(flat_tensor, (0, pad_size), mode='constant', value=0)
    
    # 重塑张量为 [..., 64]
    grouped_tensor = padded_tensor.view(-1, 64)
    
    return grouped_tensor, pad_size

@torch.no_grad()
def process_tensor(input_tensor):
    assert input_tensor.shape[1] == 64, "Input tensor must have 64 bytes in the last dimension"
    
    # 提取前2位和后6位
    two_bit = (input_tensor >> 6) & 0b11
    six_bit = input_tensor & 0b00111111
    
    # 重塑张量以便进行位操作
    two_bit = two_bit.view(input_tensor.shape[0], 16, 4)
    six_bit = six_bit.view(input_tensor.shape[0], 16, 4)
    
    # 计算result_2bit
    result_2bit = (two_bit[:, :, 0] << 6) | (two_bit[:, :, 1] << 4) | (two_bit[:, :, 2] << 2) | two_bit[:, :, 3]
    
    # 计算result_6bit
    result_6bit = torch.zeros((input_tensor.shape[0], 48), dtype=torch.uint8, device=input_tensor.device)
    result_6bit[:, 0::3] = (six_bit[:, :, 0] << 2) | (six_bit[:, :, 1] >> 4)
    result_6bit[:, 1::3] = ((six_bit[:, :, 1] & 0b1111) << 4) | (six_bit[:, :, 2] >> 2)
    result_6bit[:, 2::3] = ((six_bit[:, :, 2] & 0b11) << 6) | six_bit[:, :, 3]
    
    return result_2bit, result_6bit

@torch.no_grad()
def recover_tensor(result_2bit, result_6bit):
    assert result_2bit.shape[1] == 16 and result_6bit.shape[1] == 48, "Input tensors must have correct shapes"
    
    recovered_tensor = torch.zeros((result_2bit.shape[0], 64), dtype=torch.uint8, device=result_2bit.device)
    
    # 恢复前2位
    for i in range(4):
        recovered_tensor[:, i::4] |= ((result_2bit >> (6 - i*2)) & 0b11) << 6
    
    # 恢复后6位
    recovered_tensor[:, 0::4] |= (result_6bit[:, 0::3] >> 2) & 0b00111111
    recovered_tensor[:, 1::4] |= ((result_6bit[:, 0::3] & 0b11) << 4) | (result_6bit[:, 1::3] >> 4) & 0b00001111
    recovered_tensor[:, 2::4] |= ((result_6bit[:, 1::3] & 0b1111) << 2) | (result_6bit[:, 2::3] >> 6) & 0b00000011
    recovered_tensor[:, 3::4] |= result_6bit[:, 2::3] & 0b00111111
    
    return recovered_tensor

@torch.no_grad()
def compress(quantized_tensor):
    # 保存原始形状
    original_shape = quantized_tensor.shape
    
    # 步骤1：分组和填充
    grouped_tensor, pad_size = group_and_pad(quantized_tensor)
    
    # 步骤2：处理张量
    result_2bit, result_6bit = process_tensor(grouped_tensor)
    
    print('result_2bit: ', result_2bit.shape)
    print('result_2bit: ', result_2bit.dtype)
    # 在这里选择不同的压缩、还原、错误注入
    global compresser_type, bytes_per_group, bits_per_unit, ERR_TYPE, ERR_RATE, code_rate
    if compresser_type == 'Tree':
        Tree(result_2bit, bytes_per_group=bytes_per_group, bits_per_unit=bits_per_unit)
    elif compresser_type == 'Sparse':
        Sparse(result_2bit, bytes_per_group=bytes_per_group, bits_per_unit=bits_per_unit)
    # 注入错误（后6bit）
    result_6bit = insert_err_without_ECC(result_6bit,ERR_TYPE, ERR_RATE)
    
    # 步骤3：恢复张量
    recovered_tensor = recover_tensor(result_2bit, result_6bit)
    
    # 移除填充并恢复原始形状
    final_recovered_tensor = recovered_tensor.view(-1)[:-pad_size if pad_size > 0 else None].view(original_shape)
    
    # 断言检查恢复是否正确
    # assert torch.all(quantized_tensor == final_recovered_tensor), "Recovery failed"
    
    return final_recovered_tensor

@torch.no_grad()
def int16_to_uint8(tensor):
    """
    将 int16 张量转换为 uint8 张量。
    注意：这个函数假设输入张量的值在 0 到 255 之间。
    """
    return tensor.to(torch.uint8)

@torch.no_grad()
def uint8_to_int16(tensor):
    """
    将 uint8 张量转换为 int16 张量。
    """
    return tensor.to(torch.int16)

@torch.no_grad()
def exp(quantized_tensor):
    # 确保张量在GPU上并转换为int16
    # int16为了防止循环偏移溢出导致不循环
    int16_tensor = quantized_tensor.to(torch.int16)
    
    # 应用固定偏移（-96，相当于将128偏移到32）
    shifted_tensor = (int16_tensor - 96) % 256
    
    # 在这里加入后续实验
    uint8_shifted_tensor = int16_to_uint8(shifted_tensor)
    
    uint8_shifted_tensor_exp = compress(uint8_shifted_tensor)
    
    int16_shifted_tensor = uint8_to_int16(uint8_shifted_tensor_exp)
     
    # assert torch.all(shifted_tensor == int16_shifted_tensor), "压缩、解压错误"
        
    # 将数值偏移回原来的位置
    unshifted_tensor = (shifted_tensor + 96) % 256
    
    # 将偏移回原位置的tensor转回fp16格式并返回
    return unshifted_tensor.to(torch.float16)



# ===========================基础量化===========================

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    device = tensor.device
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = torch.zeros_like(scales, device=device)

    # 量化过程开始
    if inplace:
        quantized_tensor = tensor.div_(scales).round_().add_(zeros).clamp_(min_int, max_int)
    else:
        quantized_tensor = torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int)

    # 注释：这里是量化和反量化的分界处
    quantized_tensor = exp(quantized_tensor)

    # 反量化过程开始
    if inplace:
        dequantized_tensor = quantized_tensor.sub_(zeros).mul_(scales)
    else:
        dequantized_tensor = (quantized_tensor - zeros) * scales

    assert torch.isnan(dequantized_tensor).sum() == 0

    dequantized_tensor = dequantized_tensor.reshape(org_tensor_shape)

    return dequantized_tensor

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.reshape(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.reshape(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)



# ===========================实验主体===========================

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='cuda:0'
)
model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def process_cache(cache):
    processed_cache = quantize_activation_per_token_absmax(cache)
    return processed_cache

class CacheHook:
    def __init__(self):
        self.prev_key_cache_len = 0
        self.prev_value_cache_len = 0

    def k_cache_hook(self, module, input, output):
        _, _, cache = output
        if cache is not None and cache.key_cache is not None:
            key_cache_len = len(cache.key_cache)
            if key_cache_len > self.prev_key_cache_len:
                cache.key_cache[-1] = process_cache(cache.key_cache[-1])
            self.prev_key_cache_len = key_cache_len
        return output

    def v_cache_hook(self, module, input, output):
        _, _, cache = output
        if cache is not None and cache.value_cache is not None:
            value_cache_len = len(cache.value_cache)
            if value_cache_len > self.prev_value_cache_len:
                cache.value_cache[-1] = process_cache(cache.value_cache[-1])
            self.prev_value_cache_len = value_cache_len
        return output

# 注册钩子
cache_hook = CacheHook()
for layer in model.model.layers:
    layer.self_attn.register_forward_hook(cache_hook.k_cache_hook)
    layer.self_attn.register_forward_hook(cache_hook.v_cache_hook)
    
    
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    output = model.generate(input_ids, max_length=100)


decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated output:")
print(decoded_output)
print('compression_ratio: ', compression_ratio)
