import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# 全局变量用于记录统计信息
stats = defaultdict(list)

@torch.no_grad()
def exp(quantized_tensor, bytes_per_group=8, bits_per_unit=4):
    # 将张量转换为浮点类型，并确保在正确的设备上
    float_tensor = quantized_tensor.float()
    device = float_tensor.device

    # 计算基本统计信息
    mean = torch.mean(float_tensor).item()
    std = torch.std(float_tensor).item()
    min_val = torch.min(float_tensor).item()
    max_val = torch.max(float_tensor).item()
    
    # 计算分位数，确保 quantiles 在正确的设备上
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)
    q1, median, q3 = torch.quantile(float_tensor, quantiles).tolist()
    
    # 计算直方图
    hist = torch.histogram(float_tensor.cpu(), bins=10)  # 将张量移到 CPU 上计算直方图
    
    # 记录统计信息
    stats['mean'].append(mean)
    stats['std'].append(std)
    stats['min'].append(min_val)
    stats['max'].append(max_val)
    stats['median'].append(median)
    stats['q1'].append(q1)
    stats['q3'].append(q3)
    stats['hist_bins'].append(hist.bin_edges.tolist())
    stats['hist_counts'].append(hist.hist.tolist())
    
    return quantized_tensor  # 返回原始的量化张量


# 在程序末尾添加以下代码来绘制图表
def plot_distribution():
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # 绘制均值和标准差的变化
    axs[0, 0].plot(stats['mean'], label='Mean')
    axs[0, 0].plot(stats['std'], label='Std Dev')
    axs[0, 0].set_title('Mean and Standard Deviation')
    axs[0, 0].legend()
    
    # 绘制最小值、最大值和中位数的变化
    axs[0, 1].plot(stats['min'], label='Min')
    axs[0, 1].plot(stats['max'], label='Max')
    axs[0, 1].plot(stats['median'], label='Median')
    axs[0, 1].set_title('Min, Max, and Median')
    axs[0, 1].legend()
    
    # 绘制四分位数的变化
    axs[1, 0].plot(stats['q1'], label='Q1')
    axs[1, 0].plot(stats['median'], label='Median')
    axs[1, 0].plot(stats['q3'], label='Q3')
    axs[1, 0].set_title('Quartiles')
    axs[1, 0].legend()
    
    # 绘制最后一次的直方图
    if stats['hist_bins'] and stats['hist_counts']:
        axs[1, 1].bar(stats['hist_bins'][-1][:-1], stats['hist_counts'][-1], 
                      width=np.diff(stats['hist_bins'][-1]), align='edge')
        axs[1, 1].set_title('Last Histogram')
    
    plt.tight_layout()
    plt.savefig('quantized_tensor_distribution.png')
    print("Distribution plot saved as 'quantized_tensor_distribution.png'")


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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/media/tangshi/AI001/models/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='cuda:0'
)
model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def process_kv_cache(key_cache, value_cache):
    # 对 key_cache 进行量化
    processed_key_cache = [quantize_activation_per_token_absmax(k) for k in key_cache]
    
    # 对 value_cache 进行量化
    processed_value_cache = [quantize_activation_per_token_absmax(v) for v in value_cache]
    return processed_key_cache, processed_value_cache

def kv_cache_hook(module, input, output):
    _, _, cache = output
    key_cache = cache.key_cache
    value_cache = cache.value_cache
    
    processed_key_cache, processed_value_cache = process_kv_cache(key_cache, value_cache)
    
    cache.key_cache = processed_key_cache
    cache.value_cache = processed_value_cache
    
    return output

for layer in model.model.layers:
    layer.self_attn.register_forward_hook(kv_cache_hook)

input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated output:")
print(decoded_output)
# 在程序结束时绘制图表
plot_distribution()
