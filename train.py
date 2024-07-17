import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载ImageNet数据集
train_dir = '/root/autodl-tmp/data/train'
val_dir = '/root/autodl-tmp/data/val'

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# 定义模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 1000)  # ImageNet 有 1000 个类别
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'Loss': train_loss/(batch_idx+1), 'Acc': 100.*correct/total})

# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Test')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Loss': test_loss/len(test_loader), 'Acc': 100.*correct/total})

    accuracy = 100. * correct / total
    print(f'Epoch: {epoch}, Test Loss: {test_loss/len(test_loader):.3f}, Test Acc: {accuracy:.2f}%')
    return accuracy

# 训练模型
best_acc = 0
for epoch in range(90):  # 通常 ImageNet 训练 90 个 epoch
    train(epoch)
    acc = test(epoch)
    scheduler.step()

    if acc > best_acc:
        print(f'Saving best model with accuracy: {acc:.2f}%')
        torch.save(model.state_dict(), 'original_model_imagenet.pth')
        best_acc = acc

print(f'Best accuracy: {best_acc:.2f}%')