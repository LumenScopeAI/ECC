import csv
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载ImageNet验证数据集
val_dir = '/root/autodl-tmp/data/val'
test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

def ERR(q_tensor, ERR_TYPE, ERR_RATE):
    # 确保张量在 GPU 上
    if not q_tensor.is_cuda:
        q_tensor = q_tensor.cuda()

    # 将张量转换为字节表示
    np_arr = q_tensor.cpu().numpy()
    byte_arr = np.frombuffer(np_arr.data, dtype=np.uint8)
    byte_tensor = torch.from_numpy(byte_arr).cuda()

    # 创建错误掩码
    error_mask = torch.rand(byte_tensor.shape, device=byte_tensor.device) < ERR_RATE

    if ERR_TYPE == "0":
        # 在第一位（符号位）注入错误
        flip_mask = error_mask & (byte_tensor & 0b10000000 != 0)
        byte_tensor = byte_tensor ^ (flip_mask.to(torch.uint8) << 7)
    elif ERR_TYPE == "1":
        # 在 2、3、4 位注入错误
        for i in range(5, 8):
            flip_mask = error_mask & ((byte_tensor & (1 << i)) != 0)
            byte_tensor = byte_tensor ^ (flip_mask.to(torch.uint8) << i)
    elif ERR_TYPE == "2":
        # 在 5、6、7、8 位注入错误
        for i in range(4):
            flip_mask = error_mask & ((byte_tensor & (1 << i)) != 0)
            byte_tensor = byte_tensor ^ (flip_mask.to(torch.uint8) << i)
    elif ERR_TYPE == "3":
        # 保存原始的第1位（符号位）作为校验位
        original_1 = byte_tensor & 0b10000000

        original_234 = (byte_tensor >> 5) & 0b111
        original_5678 = byte_tensor & 0b1111

        # 注入错误（包括第1位）
        for i in range(8):
            flip_mask = error_mask & ((byte_tensor & (1 << i)) != 0)
            byte_tensor = byte_tensor ^ (flip_mask.to(torch.uint8) << i)

        # 尝试恢复第1位
        new_1 = byte_tensor & 0b10000000
        first_bit_changed = original_1 != new_1
        byte_tensor = torch.where(first_bit_changed,
                                  byte_tensor ^ 0b10000000,  # 翻转第1位
                                  byte_tensor)

        new_234 = (byte_tensor >> 5) & 0b111
        new_5678 = byte_tensor & 0b1111

        # 检查 2、3、4 位的奇偶性
        parity_234_changed = (original_234.sum(-1) % 2) != (new_234.sum(-1) % 2)
        byte_tensor = torch.where(parity_234_changed.unsqueeze(-1), 
                                  byte_tensor & 0b10011111, 
                                  byte_tensor)

        # 检查 5、6、7、8 位的奇偶性
        parity_5678_changed = (original_5678.sum(-1) % 2) != (new_5678.sum(-1) % 2)
        byte_tensor = torch.where(parity_5678_changed.unsqueeze(-1), 
                                  byte_tensor & 0b11110000, 
                                  byte_tensor)

    elif ERR_TYPE == "4":
        # 在所有位置注入错误
        for i in range(8):
            flip_mask = error_mask & ((byte_tensor & (1 << i)) != 0)
            byte_tensor = byte_tensor ^ (flip_mask.to(torch.uint8) << i)
    else:
        raise ValueError("Invalid ERR_TYPE. Must be '0', '1', '2', '3', or '4'.")

    # 将字节张量转回原始数据类型
    byte_arr_with_error = byte_tensor.cpu().numpy()
    np_arr_with_error = np.frombuffer(byte_arr_with_error.data, dtype=np_arr.dtype)
    q_tensor_with_error = torch.from_numpy(np_arr_with_error).cuda().view(q_tensor.shape)

    return q_tensor_with_error

# 修改 quantize 函数以使用新的 ERR 函数
def quantize(tensor, ERR_TYPE, ERR_RATE, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - tensor.min() / scale
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor.clamp_(qmin, qmax)
    q_tensor = ERR(q_tensor.to(torch.uint8), ERR_TYPE, ERR_RATE)  # 确保输入是 uint8
    return q_tensor.to(tensor.dtype), scale, zero_point  # 转回原始数据类型

# 反量化函数
def dequantize(q_tensor, scale, zero_point):
    return scale * (q_tensor - zero_point)

# 测试函数
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 加载原始模型
original_model = resnet18(pretrained=False)
original_model.fc = nn.Linear(512, 1000)  # ImageNet 有 1000 个类别
original_model.load_state_dict(torch.load('original_model_imagenet.pth'))
original_model = original_model.to(device)

# 创建一个列表来存储结果
results = []

print("Original Model Performance:")
original_accuracy = test(original_model, test_loader, device)
print(f"Original Accuracy: {original_accuracy:.2f}%")
print()

# 将原始模型的准确率添加到结果列表中
results.append({
    'ERR_TYPE': 'Original',
    'ERR_RATE': 0,
    'Original_Accuracy': original_accuracy,
    'Quantized_Accuracy': original_accuracy,
    'Accuracy_Difference': 0
})

for ERR_TYPE in ["0", "1", "2", "3", "4"]:
    for ERR_RATE in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        print(f"ERR_TYPE: {ERR_TYPE}, ERR_RATE: {ERR_RATE}")
        quantized_model = resnet18(pretrained=False)
        quantized_model.fc = nn.Linear(512, 1000)  # ImageNet 有 1000 个类别
        quantized_model.load_state_dict(torch.load('original_model_imagenet.pth'))
        quantized_model = quantized_model.to(device)
        
        for name, param in quantized_model.named_parameters():
            q_param, scale, zero_point = quantize(param.data, ERR_TYPE, ERR_RATE)
            param.data = dequantize(q_param, scale, zero_point)

        print("Quantized Model Performance:")
        quantized_accuracy = test(quantized_model, test_loader, device)

        accuracy_difference = abs(original_accuracy - quantized_accuracy)
        print(f"Accuracy difference: {accuracy_difference:.2f}%")
        print()

        # 将结果添加到列表中
        results.append({
            'ERR_TYPE': ERR_TYPE,
            'ERR_RATE': ERR_RATE,
            'Original_Accuracy': original_accuracy,
            'Quantized_Accuracy': quantized_accuracy,
            'Accuracy_Difference': accuracy_difference
        })

# 将结果保存到 CSV 文件
csv_file_path = '/root/autodl-tmp/result_INT8_ImageNet.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['ERR_TYPE', 'ERR_RATE', 'Original_Accuracy', 'Quantized_Accuracy', 'Accuracy_Difference']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Results have been saved to {csv_file_path}")