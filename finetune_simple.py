#!/usr/bin/env python3
"""
简单微调闭眼检测模型

使用现有模型提取特征，然后训练一个简单的分类器。
避免直接修改模型架构的复杂性。

用法：
    python finetune_simple.py
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


class EyeFrameDataset(Dataset):
    """闭眼帧数据集"""

    def __init__(self, data_dir: str, transform=None, frames_per_video: int = 8):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.frames_per_video = frames_per_video

        # 收集所有样本
        self.samples = []

        # 闭眼样本 (label=1)
        close_dir = self.data_dir / 'close'
        if close_dir.exists():
            for video_dir in close_dir.iterdir():
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob('*.jpg'))
                    if len(frames) > 0:
                        self.samples.append({
                            'frames': frames,
                            'label': 1,  # 闭眼
                            'video_id': video_dir.name,
                        })

        # 睁眼样本 (label=0)
        open_dir = self.data_dir / 'open'
        if open_dir.exists():
            for video_dir in open_dir.iterdir():
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob('*.jpg'))
                    if len(frames) > 0:
                        self.samples.append({
                            'frames': frames,
                            'label': 0,  # 睁眼
                            'video_id': video_dir.name,
                        })

        print(f'加载数据集: {len(self.samples)} 个视频')
        print(f'  闭眼: {sum(1 for s in self.samples if s["label"] == 1)}')
        print(f'  睁眼: {sum(1 for s in self.samples if s["label"] == 0)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        label = sample['label']

        # 采样固定数量的帧
        if len(frames) >= self.frames_per_video:
            indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        else:
            indices = np.random.choice(len(frames), self.frames_per_video, replace=True)

        # 加载帧
        frame_tensors = []
        for i in indices:
            frame_path = frames[i]
            frame = Image.open(frame_path).convert('RGB')

            if self.transform:
                frame = self.transform(frame)

            frame_tensors.append(frame)

        # 堆叠帧: (T, C, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)

        return video_tensor, label


class FeatureExtractor:
    """使用现有模型提取特征"""

    def __init__(self, detector):
        self.detector = detector
        self.model = detector.model

    def extract_features(self, video_tensor):
        """
        从视频张量中提取特征

        Args:
            video_tensor: (T, C, H, W) 形状的视频帧

        Returns:
            特征向量
        """
        # 转换为模型输入格式
        # 需要将 (T, C, H, W) 转换为模型期望的格式
        frames = video_tensor.unsqueeze(0)  # 添加 batch 维度: (1, T, C, H, W)

        # 使用模型的内部处理
        with torch.no_grad():
            # 尝试直接获取模型输出
            try:
                # 方法1: 直接调用模型
                output = self.model(frames)
                return output.squeeze(0)
            except Exception as e:
                print(f'特征提取错误: {e}')
                return None


class SimpleClassifier(nn.Module):
    """简单分类器"""

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


def extract_features_from_dataset(detector, dataset, device):
    """从数据集中提取所有特征"""
    features = []
    labels = []

    for i in range(len(dataset)):
        video_tensor, label = dataset[i]

        # 提取特征
        feature = detector.model(video_tensor.unsqueeze(0))

        if feature is not None:
            features.append(feature.numpy())
            labels.append(label)

    return np.array(features), np.array(labels)


def main():
    print('=' * 80)
    print('简单微调闭眼检测模型')
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

    dataset = EyeFrameDataset(
        data_dir='/yanglin/eye_detect/training_data',
        transform=train_transform,
        frames_per_video=8,
    )

    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f'  训练集: {train_size} 个视频')
    print(f'  验证集: {val_size} 个视频')
    print()

    # Step 3: 提取特征
    print('Step 3: 提取特征...')
    print('  注意: 由于模型架构复杂，我们将使用简化方法')
    print()

    # 简化方法: 使用模型的 softmax 输出作为特征
    # 这些输出已经是 [睁眼概率, 闭眼概率]

    train_features = []
    train_labels = []

    print('  提取训练集特征...')
    for i in range(len(train_dataset)):
        video_tensor, label = train_dataset[i]

        # 使用模型进行预测
        with torch.no_grad():
            try:
                # 转换格式
                frames = video_tensor.unsqueeze(0)  # (1, T, C, H, W)

                # 使用检测器的 detect 方法
                # 需要创建一个临时的输入
                result = detector.model(frames)
                probs = torch.softmax(result, dim=-1)

                train_features.append(probs.squeeze(0).numpy())
                train_labels.append(label)

                if (i + 1) % 5 == 0:
                    print(f'    进度: {i + 1}/{len(train_dataset)}')

            except Exception as e:
                print(f'    跳过样本 {i}: {e}')
                continue

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print(f'  训练特征形状: {train_features.shape}')
    print()

    # Step 4: 训练简单分类器
    print('Step 4: 训练简单分类器...')

    # 转换为 PyTorch 张量
    train_features_tensor = torch.FloatTensor(train_features)
    train_labels_tensor = torch.LongTensor(train_labels)

    # 创建分类器
    classifier = SimpleClassifier(input_dim=train_features.shape[1], hidden_dim=16, num_classes=2)

    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练循环
    num_epochs = 50
    print(f'  Epochs: {num_epochs}')
    print(f'  学习率: 0.01')
    print()

    best_accuracy = 0.0
    output_dir = Path('/yanglin/eye_detect/finetuned_model')
    output_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        classifier.train()

        # 前向传播
        outputs = classifier(train_features_tensor)
        loss = criterion(outputs, train_labels_tensor)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total = train_labels_tensor.size(0)
        correct = (predicted == train_labels_tensor).sum().item()
        accuracy = 100 * correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:02d}/{num_epochs}] Loss: {loss.item():.4f} Acc: {accuracy:.1f}%')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = output_dir / 'best_classifier.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': loss.item(),
            }, save_path)

    print()
    print('=' * 80)
    print('微调完成！')
    print('=' * 80)
    print(f'最佳训练准确率: {best_accuracy:.1f}%')
    print(f'分类器保存路径: {output_dir / "best_classifier.pth"}')
    print()
    print('下一步：')
    print('  1. 使用微调后的分类器重新评估模型')
    print('  2. 创建集成模型（原始模型 + 分类器）')


if __name__ == '__main__':
    main()
