import torch
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 创建结果目录
result_dir = '/root/autodl-tmp/ECC_LLM/result'
os.makedirs(result_dir, exist_ok=True)
csv_file_front_two = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_front_two_bits_distribution.csv')
csv_file_composed_byte = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_composed_byte_distribution.csv')
png_file_front_two = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_front_two_bits_distribution.png')
png_file_composed_byte = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_composed_byte_distribution.png')
png_file_consecutive_zeros = os.path.join(result_dir, 'Qwen2-0.5B-Instruct_consecutive_zeros_distribution.png')

# 清空CSV文件
open(csv_file_front_two, 'w').close()
open(csv_file_composed_byte, 'w').close()

# 更新 CSV header
csv_header = ['Value', 'Percentage']

# 打开CSV文件并写入header
for file in [csv_file_front_two, csv_file_composed_byte]:
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

def flip_front_two_bits(byte):
    front_two = byte >> 6
    if front_two == 0b01:
        return (byte & 0b00111111) | 0b00000000
    elif front_two == 0b00:
        return (byte & 0b00111111) | 0b01000000
    return byte

def analyze_param_distribution(model):
    front_two_distribution = np.zeros(4, dtype=np.int64)
    composed_byte_distribution = np.zeros(256, dtype=np.int64)
    consecutive_zeros_count = np.zeros(7, dtype=np.int64)
    total_bytes = 0
    composed_bytes = []
    
    with torch.no_grad():
        for name, param in tqdm(model.named_parameters(), desc="Analyzing parameters"):
            if param.dtype == torch.int8:
                # 将参数转换为numpy数组
                uint8_param = param.byte().cpu().numpy().flatten()
                
                # +96取模
                uint8_param_plus96 = (uint8_param + 96) % 256
                
                # 交换特定前两位
                uint8_param_flipped = np.vectorize(flip_front_two_bits)(uint8_param_plus96)
                
                # 确保结果在0-255范围内
                uint8_param_final = uint8_param_flipped.astype(np.uint8)
                
                # 统计前两位的分布
                front_two_bits = uint8_param_final >> 6
                unique, counts = np.unique(front_two_bits, return_counts=True)
                for value, count in zip(unique, counts):
                    front_two_distribution[value] += count
                
                # 每8个字节组成一个新的字节
                for i in range(0, len(uint8_param_final), 8):
                    chunk = uint8_param_final[i:i+8]
                    if len(chunk) == 8:
                        composed_byte = sum((b >> 6) << (2 * (7-j)) for j, b in enumerate(chunk))
                        composed_bytes.append(composed_byte)
                
                total_bytes += uint8_param_final.size
    
    # 统计组合后字节的分布
    composed_bytes = np.array(composed_bytes, dtype=np.uint8)
    unique, counts = np.unique(composed_bytes, return_counts=True)
    for value, count in zip(unique, counts):
        composed_byte_distribution[value] += count
    
    # 统计连续零的概率
    for i in range(7):
        mask = 0b11 << (2*i)
        consecutive_zeros_count[i] = np.sum((composed_bytes & mask) == 0)
    
    # 计算百分比
    front_two_percentage = (front_two_distribution / total_bytes) * 100
    composed_byte_percentage = (composed_byte_distribution / len(composed_bytes)) * 100
    consecutive_zeros_percentage = (consecutive_zeros_count / len(composed_bytes)) * 100
    
    return front_two_percentage, composed_byte_percentage, consecutive_zeros_percentage

def plot_distribution(distribution, title, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(distribution)), distribution)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Percentage')
    plt.savefig(filename)
    plt.close()

def plot_consecutive_zeros(consecutive_zeros_percentage, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 8), consecutive_zeros_percentage)
    plt.title('Probability of Consecutive Zeros in Composed Bytes')
    plt.xlabel('Position of Consecutive Zeros')
    plt.ylabel('Percentage')
    plt.xticks(range(1, 8), ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8'])
    plt.savefig(filename)
    plt.close()

def save_distribution_to_csv(distribution, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for value, percentage in enumerate(distribution):
            if percentage > 0:  # 只保存非零值
                writer.writerow([value, f"{percentage:.6f}"])

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/ECC_LLM/models/Qwen2-0.5B-Instruct", 
                                                 quantization_config=bnb_config, 
                                                 device_map="auto", 
                                                 trust_remote_code=True)

    print("Analyzing parameter distribution...")
    front_two_percentage, composed_byte_percentage, consecutive_zeros_percentage = analyze_param_distribution(model)

    print("Saving results...")
    save_distribution_to_csv(front_two_percentage, csv_file_front_two)
    save_distribution_to_csv(composed_byte_percentage, csv_file_composed_byte)
    plot_distribution(front_two_percentage, 'Distribution of Front Two Bits', png_file_front_two)
    plot_distribution(composed_byte_percentage, 'Distribution of Composed Bytes', png_file_composed_byte)
    plot_consecutive_zeros(consecutive_zeros_percentage, png_file_consecutive_zeros)

    print("\nFront Two Bits Distribution:")
    for i, percentage in enumerate(front_two_percentage):
        print(f"Bits {i:02b}: {percentage:.2f}%")

    print("\nConsecutive Zeros Probabilities in Composed Bytes:")
    for i, percentage in enumerate(consecutive_zeros_percentage):
        print(f"Bits {i+1}-{i+2}: {percentage:.2f}%")

    print("\nExperiment completed. Results saved to:")
    print(f"CSV files: {csv_file_front_two}, {csv_file_composed_byte}")
    print(f"PNG files: {png_file_front_two}, {png_file_composed_byte}, {png_file_consecutive_zeros}")

    # 清理内存
    del model
    torch.cuda.empty_cache()
