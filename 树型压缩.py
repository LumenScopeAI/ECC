import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===========================全局变量===========================
compresser_type = 'Tree'
bytes_per_group = 64
bits_per_unit = 4
# 全局变量来记录压缩比
compression_ratio = 0

# 全局变量来记录kvcache历史长度
global_kv_cache_length = 0

model_path = "/media/tangshi/AI001/models/Llama-2-7b-hf"

# ===========================树型压缩===========================

@torch.no_grad()
def Tree(org_tensor, bytes_per_group=8, bits_per_unit=4):
    global compression_ratio
    def tree_compress(org_tensor, bytes_per_group=64, bits_per_unit=4):
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

    def tree_decompress(compressed_tensor, original_shape, bytes_per_group=64, bits_per_unit=4, pad_size=0):
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
    
    # 在这里选择不同的压缩、还原、错误注入
    global compresser_type, bytes_per_group, bits_per_unit
    if compresser_type == 'Tree':
        Tree(result_2bit, bytes_per_group=8, bits_per_unit=4)
    
    # 步骤3：恢复张量
    recovered_tensor = recover_tensor(result_2bit, result_6bit)
    
    # 移除填充并恢复原始形状
    final_recovered_tensor = recovered_tensor.view(-1)[:-pad_size if pad_size > 0 else None].view(original_shape)
    
    # 断言检查恢复是否正确
    assert torch.all(quantized_tensor == final_recovered_tensor), "Recovery failed"
    
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
     
    assert torch.all(shifted_tensor == int16_shifted_tensor), "压缩、解压错误"
        
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

def process_new_kv_cache(new_key, new_value):
    processed_new_key = quantize_activation_per_token_absmax(new_key)
    processed_new_value = quantize_activation_per_token_absmax(new_value)
    return processed_new_key, processed_new_value

def kv_cache_hook(module, input, output):
    global global_kv_cache_length
    _, _, cache = output
    key_cache = cache.key_cache
    value_cache = cache.value_cache
    
    # 检查是否为列表结构
    if isinstance(key_cache, list):
        # 假设所有层的缓存长度相同，我们使用第一层的长度
        current_length = key_cache[0].size(1)
    else:
        current_length = key_cache.size(1)
    
    new_tokens = current_length - global_kv_cache_length
    
    if new_tokens > 0:
        if isinstance(key_cache, list):
            for i in range(len(key_cache)):
                # 获取新增的key和value
                new_key = key_cache[i][:, global_kv_cache_length:, :]
                new_value = value_cache[i][:, global_kv_cache_length:, :]
                
                processed_new_key, processed_new_value = process_new_kv_cache(new_key, new_value)
                
                # 更新缓存，只替换新增的部分
                key_cache[i][:, global_kv_cache_length:, :] = processed_new_key
                value_cache[i][:, global_kv_cache_length:, :] = processed_new_value
        else:
            # 获取新增的key和value
            new_key = key_cache[:, global_kv_cache_length:, :]
            new_value = value_cache[:, global_kv_cache_length:, :]
            
            processed_new_key, processed_new_value = process_new_kv_cache(new_key, new_value)
            
            # 更新缓存，只替换新增的部分
            key_cache[:, global_kv_cache_length:, :] = processed_new_key
            value_cache[:, global_kv_cache_length:, :] = processed_new_value
        
        # 更新全局历史长度
        global_kv_cache_length = current_length
    
    return output

def reset_kv_cache_length():
    global global_kv_cache_length
    global_kv_cache_length = 0

# 注册hook
for layer in model.model.layers:
    layer.self_attn.register_forward_hook(kv_cache_hook)

input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    output = model.generate(input_ids, max_length=10)
    # 在开始新的推理任务时调用
    reset_kv_cache_length()

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated output:")
print(decoded_output)
print('compression_ratio: ', compression_ratio)
