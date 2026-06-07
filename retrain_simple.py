#!/usr/bin/env python3
"""
简单重新训练闭眼检测模型

使用原始模型的输出作为特征，训练一个简单的分类器。

用法：
    python retrain_simple.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import json
import importlib.machinery

# 禁用 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')

# 修复 torchvision 兼容性问题
import types
import torchvision.transforms.functional as _tvF
_functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
_functional_tensor.rgb_to_grayscale = _tvF.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor


def load_original_api():
    """动态加载 eye_detect_api.1py 原始代码"""
    loader = importlib.machinery.SourceFileLoader(
        "eye_detect_api_original",
        "/yanglin/eye_detect/eye_detect_api.1py"
    )
    spec = importlib.util.spec_from_loader("eye_detect_api_original", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class HfFrameDataset(Dataset):
    """
    帧数据集

    从 JSON 标注文件加载数据
    """

    def __init__(self, json_path: str, transform=None, num_frames: int = 8):
        self.transform = transform
        self.num_frames = num_frames

        # 加载 JSON 标注
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        print(f'加载数据集: {len(self.annotations)} 个视频')
        print(f'  闭眼: {sum(1 for a in self.annotations if a["train_label"] == 1)}')
        print(f'  睁眼: {sum(1 for a in self.annotations if a["train_label"] == 0)}')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = ann['video_path']
        label = ann['train_label']
        frame_count = ann['video_frame_count']

        # 采样帧索引
        if frame_count >= self.num_frames:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            indices = np.random.choice(frame_count, self.num_frames, replace=True)

        # 加载帧
        frame_tensors = []
        for i in indices:
            frame_path = video_path % i
            try:
                frame = Image.open(frame_path).convert('RGB')
                if self.transform:
                    frame = self.transform(frame)
                frame_tensors.append(frame)
            except Exception as e:
                if self.transform:
                    dummy = Image.new('RGB', (224, 224))
                    frame_tensors.append(self.transform(dummy))
                else:
                    frame_tensors.append(torch.zeros(3, 224, 224))

        # 堆叠帧: (T, C, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)

        return video_tensor, label


class SimpleClassifier(nn.Module):
    """
    简单分类器

    使用原始模型的输出作为特征
    """

    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def extract_features(detector, video_tensor):
    """
    使用原始模型提取特征

    Args:
        detector: 检测器实例
        video_tensor: 视频张量 (T, C, H, W)

    Returns:
        特征向量 (2,)
    """
    # 转换为模型输入格式
    # 模型期望 (B, C, T, H, W) 格式
    video_tensor = video_tensor.unsqueeze(0)  # 添加 batch 维度: (1, T, C, H, W)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

    # 使用模型进行预测
    with torch.no_grad():
        # 包装在列表中，因为模型期望列表输入
        output = detector.model([video_tensor])
        # 应用 softmax 获取概率
        probs = torch.softmax(output, dim=-1)

    return probs.squeeze(0).numpy()


def create_data_transforms():
    """创建数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def main():
    print('=' * 80)
    print('简单重新训练闭眼检测模型')
    print('=' * 80)
    print()

    # 设置设备
    device = torch.device('cpu')
    print(f'使用设备: {device}')
    print()

    # Step 1: 加载预训练模型
    print('Step 1: 加载预训练模型...')
    api_module = load_original_api()

    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cpu',
        enable_face_extraction=False,
        face_pad_ratio=0.5,
    )
    print('  预训练模型加载成功')
    print()

    # Step 2: 准备数据
    print('Step 2: 准备训练数据...')
    train_transform, val_transform = create_data_transforms()

    train_dataset = HfFrameDataset(
        json_path='/yanglin/eye_detect/training_json/train.json',
        transform=train_transform,
        num_frames=8,
    )

    val_dataset = HfFrameDataset(
        json_path='/yanglin/eye_detect/training_json/val.json',
        transform=val_transform,
        num_frames=8,
    )

    print(f'  训练集: {len(train_dataset)} 个视频')
    print(f'  验证集: {len(val_dataset)} 个视频')
    print()

    # Step 3: 提取特征
    print('Step 3: 提取特征...')
    print('-' * 80)

    # 提取训练集特征
    train_features = []
    train_labels = []

    print('  提取训练集特征...')
    for i in range(len(train_dataset)):
        video_tensor, label = train_dataset[i]

        # 提取特征
        feature = extract_features(detector, video_tensor)

        train_features.append(feature)
        train_labels.append(label)

        if (i + 1) % 10 == 0:
            print(f'    进度: {i + 1}/{len(train_dataset)}')

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print(f'  训练特征形状: {train_features.shape}')
    print()

    # 提取验证集特征
    val_features = []
    val_labels = []

    print('  提取验证集特征...')
    for i in range(len(val_dataset)):
        video_tensor, label = val_dataset[i]

        # 提取特征
        feature = extract_features(detector, video_tensor)

        val_features.append(feature)
        val_labels.append(label)

        if (i + 1) % 5 == 0:
            print(f'    进度: {i + 1}/{len(val_dataset)}')

    val_features = np.array(val_features)
    val_labels = np.array(val_labels)

    print(f'  验证特征形状: {val_features.shape}')
    print()

    # Step 4: 训练分类器
    print('Step 4: 训练分类器...')

    # 转换为 PyTorch 张量
    train_features_tensor = torch.FloatTensor(train_features)
    train_labels_tensor = torch.LongTensor(train_labels)
    val_features_tensor = torch.FloatTensor(val_features)
    val_labels_tensor = torch.LongTensor(val_labels)

    # 创建分类器
    classifier = SimpleClassifier(input_dim=train_features.shape[1], hidden_dim=16, num_classes=2)

    # 使用类别权重平衡数据
    close_count = np.sum(train_labels == 1)
    open_count = np.sum(train_labels == 0)
    total_count = len(train_labels)

    weight_close = total_count / (2 * close_count) if close_count > 0 else 1.0
    weight_open = total_count / (2 * open_count) if open_count > 0 else 1.0

    class_weights = torch.FloatTensor([weight_open, weight_close])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练循环
    num_epochs = 100
    print(f'  Epochs: {num_epochs}')
    print(f'  学习率: 0.01')
    print(f'  类别权重: close={weight_close:.2f}, open={weight_open:.2f}')
    print()

    best_accuracy = 0.0
    best_close_recall = 0.0
    output_dir = Path('/yanglin/eye_detect/retrain_output')
    output_dir.mkdir(exist_ok=True)

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_close_recall': [],
    }

    for epoch in range(num_epochs):
        classifier.train()

        # 前向传播
        outputs = classifier(train_features_tensor)
        loss = criterion(outputs, train_labels_tensor)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        train_total = train_labels_tensor.size(0)
        train_correct = (predicted == train_labels_tensor).sum().item()
        train_acc = 100 * train_correct / train_total

        # 验证
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(val_features_tensor)
            val_loss = criterion(val_outputs, val_labels_tensor)

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total = val_labels_tensor.size(0)
            val_correct = (val_predicted == val_labels_tensor).sum().item()
            val_acc = 100 * val_correct / val_total

            # 计算各类别准确率
            val_close_mask = val_labels_tensor == 1
            val_open_mask = val_labels_tensor == 0

            val_close_correct = ((val_predicted == val_labels_tensor) & val_close_mask).sum().item()
            val_close_total = val_close_mask.sum().item()
            val_close_recall = 100 * val_close_correct / val_close_total if val_close_total > 0 else 0

            val_open_correct = ((val_predicted == val_labels_tensor) & val_open_mask).sum().item()
            val_open_total = val_open_mask.sum().item()
            val_open_recall = 100 * val_open_correct / val_open_total if val_open_total > 0 else 0

        # 记录历史
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        history['val_close_recall'].append(val_close_recall)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:03d}/{num_epochs}] '
                  f'Train Loss: {loss.item():.4f} Acc: {train_acc:.1f}% | '
                  f'Val Loss: {val_loss.item():.4f} Acc: {val_acc:.1f}% | '
                  f'Close: {val_close_recall:.1f}% Open: {val_open_recall:.1f}%')

        # 保存最佳模型（基于闭眼召回率）
        if val_close_recall > best_close_recall:
            best_close_recall = val_close_recall
            best_accuracy = val_acc

            save_path = output_dir / 'best_classifier.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss.item(),
                'close_recall': val_close_recall,
                'open_recall': val_open_recall,
            }, save_path)

            if (epoch + 1) % 10 == 0:
                print(f'  -> 保存最佳模型 (闭眼召回率: {val_close_recall:.1f}%, 准确率: {val_acc:.1f}%)')

    print()

    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # 总结
    print('=' * 80)
    print('训练完成！')
    print('=' * 80)
    print()
    print(f'【最佳模型性能】')
    print(f'  准确率: {best_accuracy:.1f}%')
    print(f'  闭眼召回率: {best_close_recall:.1f}%')
    print()
    print(f'【输出文件】')
    print(f'  分类器: {output_dir / "best_classifier.pth"}')
    print(f'  历史: {history_path}')
    print()
    print('下一步：')
    print('  1. 使用新分类器重新运行批量检测')
    print('  2. 对比微调前后的性能')


if __name__ == '__main__':
    main()
