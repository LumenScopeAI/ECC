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

# 创建结果目录和CSV文件
result_dir = '/root/autodl-tmp/ECC_LLM/result'
os.makedirs(result_dir, exist_ok=True)
csv_file = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_ECC_results.csv')

# 更新 CSV header
csv_header = ['ERR_Rate', 'ERR_Type', 'Overall_Change_Percentage', 'Original_PPL', 'Modified_PPL', 'Corrected_Errors']

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


def bit_flip(x, err_rate, err_type):
    device = x.device
    mask = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
    
    if err_type in range(8):  # 对应于8个比特位中的一个
        err_rate = err_rate*8
        flip_mask = (torch.rand(x.shape, device=device) < err_rate).bool()
        flip_bit = mask[7 - err_type]
        return torch.where(flip_mask, x ^ flip_bit, x)
    elif err_type == 8:  # 所有位都可能翻转
        flip_mask = (torch.rand(x.shape + (8,), device=device) < err_rate).bool()
        flipped = x.unsqueeze(-1) ^ mask
        return torch.where(flip_mask, flipped, x.unsqueeze(-1)).sum(dim=-1).byte()
    elif err_type == 9:  # ECC(64, 72) 编码后的错误注入
        x = x.contiguous()
        if x.numel() % 8 != 0:
            padding = 8 - (x.numel() % 8)
            x = torch.cat([x, torch.zeros(padding, dtype=torch.uint8, device=device)])
        
        encoded = hamming_encode_bytes(x.view(-1, 8))
        
        # 对编码后的数据注入错误（包括校验位）
        flip_mask = (torch.rand(encoded.shape, device=device) < err_rate).bool()
        encoded = encoded ^ flip_mask.byte()
        
        # 解码并纠错
        corrected, errors_corrected = hamming_decode_and_correct_bytes(encoded)
        
        if x.numel() != corrected.numel():
            corrected = corrected[:x.numel()]
        
        return corrected.view(x.shape), errors_corrected
    else:
        raise ValueError("Invalid err_type. Must be 0-9.")

def process_int8_params_batch(model, err_rate, err_type, batch_size=1000000, layers_to_process=None):
    total_errors_corrected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if layers_to_process and not any(layer in name for layer in layers_to_process):
                continue
            if param.dtype == torch.int8:
                uint8_param = param.byte()
                original_shape = uint8_param.shape
                flattened = uint8_param.view(-1)
                
                # 确保 flattened 的长度是 8 的倍数
                padding = (8 - (flattened.numel() % 8)) % 8
                if padding:
                    flattened = torch.cat([flattened, torch.zeros(padding, dtype=torch.uint8, device=flattened.device)])
                
                for i in range(0, flattened.numel(), batch_size):
                    batch = flattened[i:i+batch_size]
                    if err_type == 9:
                        flipped_batch, errors_corrected = bit_flip(batch, err_rate, err_type)
                        total_errors_corrected += errors_corrected
                    else:
                        flipped_batch = bit_flip(batch, err_rate, err_type)
                    flattened[i:i+batch_size] = flipped_batch
                
                # 移除填充（如果有的话）
                if padding:
                    flattened = flattened[:-padding]
                
                new_param = flattened.view(original_shape).to(torch.int8)
                param.data.copy_(new_param)
    
    return total_errors_corrected

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

    return overall_change_percentage

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
        total_errors_corrected = process_int8_params_batch(model, err_rate, err_type, batch_size=8000000)

        # 检查参数变化
        overall_change_percentage = check_param_changes(model_before, model)

        # 评估修改后的模型性能
        modified_ppl = evaluate_model(model, tokenizer)

        # 写入CSV文件
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([err_rate, err_type, overall_change_percentage, original_ppl, modified_ppl, total_errors_corrected])

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
    err_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    err_types = range(10)  # 0-9，包括新的 ECC 类型

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
