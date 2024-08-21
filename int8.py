import torch
import random
import gc
import copy
import sys
import os
import traceback
import math
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 创建结果目录和文件
result_dir = '/root/autodl-tmp/ECC_LLM/result'
os.makedirs(result_dir, exist_ok=True)
csv_file = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_ECC_results.csv')
txt_file = os.path.join(result_dir, 'Qwen2-0.5B-Instruct.txt')

# 清空CSV文件和TXT文件
open(csv_file, 'w').close()
open(txt_file, 'w').close()

# 更新 CSV header
csv_header = ['ERR_Rate', 'ERR_Type', 'Changed_Param', 'Original_PPL', 'Modified_PPL', 'Total_Errors', 'Corrected_Errors']

# 打开CSV文件并写入header
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

def hamming_encode_bytes(data):
    device = data.device
    encoded = torch.zeros((*data.shape[:-1], 9), dtype=torch.uint8, device=device)
    encoded[..., :8] = data
    
    # 计算校验位
    encoded[..., 8] = (
        encoded[..., 0] ^ encoded[..., 1] ^ encoded[..., 3] ^ encoded[..., 4] ^ encoded[..., 6] ^
        encoded[..., 2] ^ encoded[..., 3] ^ encoded[..., 5] ^ encoded[..., 6] ^
        encoded[..., 4] ^ encoded[..., 5] ^ encoded[..., 6] ^ encoded[..., 7]
    )
    return encoded

def hamming_decode_and_correct_bytes(encoded):
    device = encoded.device
    syndrome = (
        (encoded[..., 0] ^ encoded[..., 1] ^ encoded[..., 3] ^ encoded[..., 4] ^ encoded[..., 6]) |
        ((encoded[..., 2] ^ encoded[..., 3] ^ encoded[..., 5] ^ encoded[..., 6]) << 1) |
        ((encoded[..., 4] ^ encoded[..., 5] ^ encoded[..., 6] ^ encoded[..., 7]) << 2)
    )
    error_pos = syndrome.unsqueeze(-1) == torch.arange(8, device=device)
    corrected = encoded[..., :8].clone()
    corrected[error_pos] ^= 1
    return corrected, error_pos.sum().item()

def sparse_hamming_encode_bytes(data):
    device = data.device
    # 确保输入是32字节（256位）
    assert data.shape[-1] == 32, "Input must be 32 bytes (256 bits)"
    
    encoded = torch.zeros((*data.shape[:-1], 33), dtype=torch.uint8, device=device)
    encoded[..., :32] = data
    
    # 只取每字节的前两位参与校验
    check_bits = (data & 0b11000000) >> 6
    
    # 计算校验位（1字节）
    encoded[..., 32] = (
        (check_bits[..., 0] ^ check_bits[..., 1] ^ check_bits[..., 3] ^ check_bits[..., 4] ^ check_bits[..., 6]) |
        ((check_bits[..., 2] ^ check_bits[..., 3] ^ check_bits[..., 5] ^ check_bits[..., 6]) << 2) |
        ((check_bits[..., 4] ^ check_bits[..., 5] ^ check_bits[..., 6] ^ check_bits[..., 7]) << 4) |
        ((check_bits[..., :].sum(dim=-1) & 1) << 7)
    )
    
    return encoded

def sparse_hamming_decode_and_correct_bytes(encoded):
    device = encoded.device
    data = encoded[..., :32].clone()
    check_bits = (data & 0b11000000) >> 6
    
    # 计算校验位
    computed_check = (
        (check_bits[..., 0] ^ check_bits[..., 1] ^ check_bits[..., 3] ^ check_bits[..., 4] ^ check_bits[..., 6]) |
        ((check_bits[..., 2] ^ check_bits[..., 3] ^ check_bits[..., 5] ^ check_bits[..., 6]) << 2) |
        ((check_bits[..., 4] ^ check_bits[..., 5] ^ check_bits[..., 6] ^ check_bits[..., 7]) << 4) |
        ((check_bits.sum(dim=-1) & 1) << 7)
    )
    
    syndrome = computed_check ^ encoded[..., 32]
    
    if syndrome.any():
        # 错误检测和纠正
        error_pos = torch.zeros((*syndrome.shape, 32), dtype=torch.bool, device=device)
        error_pos[..., 0] = syndrome & 1
        error_pos[..., 1] = syndrome & 2
        error_pos[..., 2] = syndrome & 4
        error_pos[..., 3] = (syndrome & 3) == 3
        error_pos[..., 4] = syndrome & 16
        error_pos[..., 5] = (syndrome & 20) == 20
        error_pos[..., 6] = (syndrome & 21) == 21
        error_pos[..., 7] = syndrome & 128
        
        # 只纠正前两位
        data[error_pos] ^= 0b11000000
    
    return data, syndrome.bool().sum().item()

def special_encode(x):
    # 将每个参数的数值+96
    encoded = (x + 96).byte()
    
    # 修改前两位
    mask_00 = (encoded & 0b11000000) == 0
    mask_01 = (encoded & 0b11000000) == 0b01000000
    
    encoded[mask_00] |= 0b01000000
    encoded[mask_01] &= 0b10111111
    
    return encoded

def special_decode(x):
    # 还原前两位
    mask_00 = (x & 0b11000000) == 0b01000000
    mask_01 = (x & 0b11000000) == 0
    
    decoded = x.clone()
    decoded[mask_00] &= 0b10111111
    decoded[mask_01] |= 0b01000000
    
    # 将每个参数的数值-96
    return (decoded - 96).byte()

def bit_flip(x, err_rate, err_type):
    device = x.device
    mask = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
    
    original = x.clone()
    total_errors = 0
    errors_corrected = 0
    
    if err_type in range(8):  # 对应于8个比特位中的一个
        err_rate = err_rate * 8
        flip_mask = (torch.rand(x.shape, device=device) < err_rate).bool()
        flip_bit = mask[7 - err_type]
        x = torch.where(flip_mask, x ^ flip_bit, x)
        total_errors = flip_mask.sum().item()
    elif err_type == 8:  # 所有位都可能翻转
        # 创建一个表示每个字节的8个位的张量
        x_bits = torch.stack([(x & bit) != 0 for bit in mask], dim=-1).bool()
        
        # 为每个位创建翻转掩码
        flip_mask = (torch.rand(x.shape + (8,), device=device) < err_rate).bool()
        
        # 应用翻转
        flipped_bits = x_bits ^ flip_mask
        
        # 将位重新组合成字节
        x = torch.zeros_like(x)
        for i, bit in enumerate(mask):
            x += flipped_bits[..., i] * bit
        
        total_errors = flip_mask.sum().item()
    elif err_type == 9:  # ECC(64, 72) 编码后的错误注入
        x = x.contiguous()
        original_shape = x.shape
        original_numel = x.numel()
        
        # 计算需要的填充
        padding = (8 - (original_numel % 8)) % 8
        if padding:
            x = torch.cat([x.view(-1), torch.zeros(padding, dtype=torch.uint8, device=device)])
        
        encoded = hamming_encode_bytes(x.view(-1, 8))
        
        # 对编码后的数据注入错误（包括校验位）
        flip_mask = (torch.rand(encoded.shape, device=device) < err_rate).bool()
        encoded_flipped = encoded ^ flip_mask.byte()
        
        # 解码并纠错
        corrected, errors_corrected = hamming_decode_and_correct_bytes(encoded_flipped)
        
        # 移除填充并恢复原始形状
        corrected = corrected.view(-1)[:original_numel].view(original_shape)
        
        total_errors = flip_mask.sum().item()
        x = corrected
    elif 10 <= err_type <= 16:
        protected_bits = err_type - 9  # 不出错的位数
        unprotected_bits = 8 - protected_bits  # 可能出错的位数
        
        # 为每个未受保护的位生成翻转掩码
        flip_masks = [
            (torch.rand(x.shape, device=device) < err_rate).bool()
            for _ in range(unprotected_bits)
        ]
        
        # 创建一个副本以保存结果
        result = x.clone()
        
        # 对每个未受保护的位应用错误注入
        for i, flip_mask in enumerate(flip_masks):
            flip_bit = mask[7 - protected_bits - i]  # 从最低的未受保护位开始
            result = torch.where(flip_mask, result ^ flip_bit, result)
            total_errors += flip_mask.sum().item()
        
        x = result
    elif err_type == 17:  # 新增的特殊编码方法
        # 特殊编码
        encoded = special_encode(x)
        
        # 使用 ECC(64, 72) 编码
        encoded = encoded.contiguous()
        original_shape = encoded.shape
        original_numel = encoded.numel()
        
        # 计算需要的填充
        padding = (8 - (original_numel % 8)) % 8
        if padding:
            encoded = torch.cat([encoded.view(-1), torch.zeros(padding, dtype=torch.uint8, device=device)])
        
        encoded = hamming_encode_bytes(encoded.view(-1, 8))
        
        # 对编码后的数据注入错误（包括校验位）
        flip_mask = (torch.rand(encoded.shape, device=device) < err_rate).bool()
        encoded_flipped = encoded ^ flip_mask.byte()
        
        # 解码并纠错
        corrected, errors_corrected = hamming_decode_and_correct_bytes(encoded_flipped)
        
        # 移除填充并恢复原始形状
        corrected = corrected.view(-1)[:original_numel].view(original_shape)
        
        # 特殊解码
        decoded = special_decode(corrected)
        
        total_errors = flip_mask.sum().item()
        x = decoded
    elif err_type == 18:  # 新增的稀疏汉明码方法
        # 特殊编码
        encoded = special_encode(x)
        
        # 使用稀疏汉明码编码
        encoded = encoded.contiguous()
        original_shape = encoded.shape
        original_numel = encoded.numel()
        
        # 计算需要的填充
        padding = (32 - (original_numel % 32)) % 32
        if padding:
            encoded = torch.cat([encoded.view(-1), torch.zeros(padding, dtype=torch.uint8, device=device)])
        
        encoded = sparse_hamming_encode_bytes(encoded.view(-1, 32))
        
        # 对编码后的数据注入错误
        flip_mask = (torch.rand(encoded.shape, device=device) < err_rate).bool()
        encoded_flipped = encoded ^ flip_mask.byte()
        
        # 解码并纠错
        corrected, errors_corrected = sparse_hamming_decode_and_correct_bytes(encoded_flipped)
        
        # 移除填充并恢复原始形状
        corrected = corrected.view(-1)[:original_numel].view(original_shape)
        
        # 特殊解码
        decoded = special_decode(corrected)
        
        total_errors = flip_mask.sum().item()
        x = decoded
    elif err_type == 19:  # 模拟ECC(64, 72)行为，但不进行实际编码解码
        x = x.contiguous()
        original_shape = x.shape
        original_numel = x.numel()
        
        # 计算需要的填充
        padding = (64 - (original_numel % 64)) % 64
        if padding:
            x = torch.cat([x.view(-1), torch.zeros(padding, dtype=torch.uint8, device=device)])
        
        # 将数据重塑为(N, 64)的形状，其中N是64位块的数量
        reshaped = x.view(-1, 64)
        
        # 为每一位创建翻转掩码
        flip_mask = (torch.rand(reshaped.shape, device=device) < err_rate).bool()
        
        # 计算每个64位块中翻转的位数
        flips_per_block = flip_mask.sum(dim=1)
        
        # 创建一个掩码，标识需要纠正的块（1或2位翻转）
        correctable_mask = (flips_per_block == 1) | (flips_per_block == 2)
        
        # 计算总错误数和被纠正的错误数量
        total_errors = flip_mask.sum().item()
        errors_corrected = flips_per_block[correctable_mask].sum().item()
        
        # 应用翻转，但只对不可纠正的块（3位或更多翻转）进行翻转
        result = reshaped.clone()
        result[~correctable_mask] ^= flip_mask[~correctable_mask]
        
        # 重塑回原始形状并移除填充
        x = result.view(-1)[:original_numel].view(original_shape)
        # 模拟Value-aware Parity Insertion ECC的行为
        threshold = 128  # 对应于0.5，因为我们使用的是uint8类型
        mask = x >= threshold
        x[mask] = x[mask] & 0xFC  # 将最低两位置为0（0xFC = 11111100
    else:
        raise ValueError("Invalid err_type. Must be 0-19.")
    
    return x, original, total_errors, errors_corrected

def process_int8_params_batch(model, err_rate, err_type, batch_size=1000000, layers_to_process=None):
    total_errors = 0
    total_errors_corrected = 0
    sample_changes = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if layers_to_process and not any(layer in name for layer in layers_to_process):
                continue
            if param.dtype == torch.int8:
                uint8_param = param.byte()
                original_shape = uint8_param.shape
                flattened = uint8_param.view(-1)
                
                for i in range(0, flattened.numel(), batch_size):
                    batch = flattened[i:i+batch_size]
                    flipped_batch, original_batch, errors, errors_corrected = bit_flip(batch, err_rate, err_type)
                    total_errors += errors
                    total_errors_corrected += errors_corrected
                    flattened[i:i+batch_size] = flipped_batch
                    
                    # 收集样本变化
                    if len(sample_changes) < 10:
                        for j in range(min(10 - len(sample_changes), len(original_batch))):
                            orig = original_batch[j]
                            flipped = flipped_batch[j]
                            sample_changes.append((orig.item(), flipped.item()))
                
                new_param = flattened.view(original_shape).to(torch.int8)
                param.data.copy_(new_param)
    
    return total_errors, total_errors_corrected, sample_changes

def calculate_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())

def evaluate_model(model, tokenizer):
    # Calculate PPL
    test_text = "人工智能是一个革命性的技术，它将改变我们的生活方式和工作方式。"
    ppl = calculate_ppl(model, tokenizer, test_text)
    return ppl

def check_param_changes(model_before, model_after):
    total_params = 0
    changed_params = 0

    for (name1, param1), (name2, param2) in zip(model_before.named_parameters(), model_after.named_parameters()):
        if name1 != name2:
            raise ValueError(f"Parameter names do not match: {name1} vs {name2}")

        if param1.dtype == torch.int8:
            total_params += param1.numel()
            diff = (param1 != param2).sum().item()
            changed_params += diff

    overall_change_percentage = (changed_params / total_params) * 100 if total_params > 0 else 0

    return overall_change_percentage, changed_params

def run_experiment(err_rate, err_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/ECC_LLM/models/Qwen2-0.5B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/ECC_LLM/models/Qwen2-0.5B-Instruct", quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

    # 评估原始模型性能
    original_ppl = evaluate_model(model, tokenizer)

    # 创建模型的深拷贝
    model_before = copy.deepcopy(model)

    try:
        # 注入错误
        total_errors, total_errors_corrected, sample_changes = process_int8_params_batch(model, err_rate, err_type, batch_size=8000000)

        # 检查参数变化
        overall_change_percentage, changed_params = check_param_changes(model_before, model)

        # 评估修改后的模型性能
        modified_ppl = evaluate_model(model, tokenizer)

        # 写入CSV文件
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([err_rate, err_type, changed_params, original_ppl, modified_ppl, total_errors, total_errors_corrected])

        # 写入TXT文件
        with open(txt_file, 'a') as f:
            f.write(f"ERR_Rate: {err_rate}, ERR_Type: {err_type}\n")
            f.write(f"Total Errors: {total_errors}, Corrected Errors: {total_errors_corrected}\n")
            for i, (orig, flipped) in enumerate(sample_changes, 1):
                f.write(f"Sample {i}:\n")
                f.write(f"  Original:  {bin(orig)[2:].zfill(8)}\n")
                f.write(f"  Flipped:   {bin(flipped)[2:].zfill(8)}\n")
            f.write("="*50 + "\n\n")

    except Exception as e:
        print(f"Error occurred during experiment:")
        print(f"ERR_Rate: {err_rate}, ERR_Type: {err_type}")
        print(traceback.format_exc())
    finally:
        # 清理内存
        del model
        del model_before
        torch.cuda.empty_cache()
        gc.collect()

# 主程序
if __name__ == "__main__":
    err_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8, 1e-9]
    err_types = range(20)  # 0-19，包括新的错误类型 

    total_experiments = len(err_rates) * len(err_types)

    for i, err_rate in enumerate(err_rates):
        for j, err_type in enumerate(err_types):
            current_experiment = i * len(err_types) + j + 1
            print(f"Starting experiment {current_experiment} of {total_experiments}")
            print(f"ERR_Rate: {err_rate}, ERR_Type: {err_type}")
            run_experiment(err_rate, err_type)
            print(f"Completed experiment {current_experiment} of {total_experiments}")
            print("="*50)

print("All experiments completed. Results saved to:", csv_file)
print("Detailed sample changes saved to:", txt_file)
