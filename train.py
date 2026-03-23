import numpy as np
import matplotlib.pyplot as plt
import os
import json  # 新增：用于保存/加载标签映射
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

from utils import read_csv_with_encoding, save_model_info, load_image_safe
from model import get_model

# 数据增强配置
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class FolderImageDataset(Dataset):
    """适配文件夹分类结构的数据集类（替代原CSV版）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 从文件夹结构获取类别信息
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        
        # 遍历所有图片，生成图片路径和标签的列表
        self.img_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.img_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        
        print(f"数据集加载完成: {len(self.img_paths)} 个样本, {self.num_classes} 个类别")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        
        # 安全加载图片
        img = load_image_safe(img_path)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# 新增：保存标签映射的函数
def save_label_mapping(class_to_idx, idx_to_class, save_path='label_mapping.json'):
    """
    保存类别映射关系到JSON文件
    :param class_to_idx: 类别名到索引的字典（如{"玫瑰":0, "百合":1}）
    :param idx_to_class: 索引到类别名的字典（如{0:"玫瑰", 1:"百合"}）
    :param save_path: 保存路径，默认当前目录下label_mapping.json
    """
    # 转换int键为str（JSON不支持int作为键）
    idx_to_class_str = {str(k): v for k, v in idx_to_class.items()}
    label_mapping = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class_str,
        'num_classes': len(class_to_idx)
    }
    
    # 保存到JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    print(f"标签映射已保存到: {os.path.abspath(save_path)}")

# 新增：加载标签映射的函数（方便预测时调用）
def load_label_mapping(load_path='label_mapping.json'):
    """
    从JSON文件加载类别映射关系
    :param load_path: 加载路径
    :return: class_to_idx, idx_to_class
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"标签映射文件不存在: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    # 转换str键回int
    idx_to_class = {int(k): v for k, v in label_mapping['idx_to_class'].items()}
    return label_mapping['class_to_idx'], idx_to_class

def train_model():
    # 配置参数（修改为文件夹路径）
    train_img_dir = r"C:\Users\smtur\Desktop\submission\data\test"
    val_img_dir = r"C:\Users\smtur\Desktop\submission\data\val"
    model_name = 'resnet50'
    batch_size = 16
    epochs = 5
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查文件
    if not os.path.exists(train_img_dir) or not os.path.exists(val_img_dir):
        print("错误: 训练/验证数据路径不存在")
        print(f"训练集路径: {train_img_dir} (存在: {os.path.exists(train_img_dir)})")
        print(f"验证集路径: {val_img_dir} (存在: {os.path.exists(val_img_dir)})")
        return

    # 加载数据
    train_dataset = FolderImageDataset(train_img_dir, data_transforms['train'])
    val_dataset = FolderImageDataset(val_img_dir, data_transforms['valid'])
    
    # ========== 关键修改1：训练开始前保存标签映射 ==========
    save_label_mapping(
        class_to_idx=train_dataset.class_to_idx,
        idx_to_class=train_dataset.idx_to_class,
        save_path='label_mapping.json'  # 保存到当前目录，可修改路径如'./model/label_mapping.json'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # 创建模型
    model = get_model(model_name, train_dataset.num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)

    # 训练记录
    best_acc = 0.0
    train_losses = []
    val_accs = []

    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="训练")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(train_correct/train_total):.4f}"
            })

        # 计算训练指标
        epoch_loss = train_loss / train_total
        epoch_acc = train_correct.double() / train_total
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                pbar.set_postfix({'acc': f"{(val_correct/val_total):.4f}"})

        val_acc = val_correct.double() / val_total
        val_accs.append(val_acc.item())

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_model_info(model, train_dataset)
            print(f"保存最佳模型 (验证准确率: {val_acc:.4f})")

        # 打印 epoch 总结
        print(f"训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}")
        print(f"验证准确率: {val_acc:.4f}, 最佳准确率: {best_acc:.4f}")
        
        scheduler.step()

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, 'r-', label='验证准确率')
    plt.title('验证准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

    print("训练完成!")

if __name__ == '__main__':
    # 解决pandas警告
    pd.options.mode.chained_assignment = None
    train_model()