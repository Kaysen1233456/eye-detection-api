#!/usr/bin/env python3
"""
微调 UniFormerV2 闭眼检测模型

使用新提取的训练数据（11 个闭眼案例 + 11 个睁眼案例）进行微调，
提高模型对闭眼的检测能力。

用法：
    python finetune_model.py
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

import importlib.machinery


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
    """
    闭眼帧数据集

    从 training_data/close/ 和 training_data/open/ 加载帧图像
    """

    def __init__(self, data_dir: str, transform=None, frames_per_video: int = 8):
        """
        Args:
            data_dir: 数据目录 (包含 close/ 和 open/ 子目录)
            transform: 图像变换
            frames_per_video: 每个视频使用的帧数
        """
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
            # 均匀采样
            indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        else:
            # 重复采样
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


class SimpleEyeDetector(nn.Module):
    """
    简单的闭眼检测模型

    使用预训练的 UniFormerV2 作为特征提取器，
    添加一个简单的分类头进行微调。
    """

    def __init__(self, pretrained_model, num_classes=2):
        super().__init__()
        self.backbone = pretrained_model

        # 冻结 backbone 的大部分参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 获取 backbone 的输出维度
        # UniFormerV2 的输出维度通常是 768
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # 只训练分类头
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: (B, T, C, H, W) -> 需要转换为 (B, C, T, H, W)
        if x.dim() == 5:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        elif x.dim() == 4:
            # 如果是 (B, C, H, W)，添加时间维度
            x = x.unsqueeze(2)

        # 提取特征
        with torch.no_grad():
            features = self.backbone(x)

        # 分类
        output = self.classifier(features)
        return output


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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(videos)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy


def main():
    print('=' * 80)
    print('微调 UniFormerV2 闭眼检测模型')
    print('=' * 80)
    print()

    # 设置设备 - 使用 CPU 避免 CUDA 兼容性问题
    device = torch.device('cpu')
    print(f'使用设备: {device} (避免 CUDA 兼容性问题)')
    print()

    # Step 1: 加载预训练模型
    print('Step 1: 加载预训练模型...')
    api_module = load_original_api()

    # 创建检测器实例（只用于获取模型）
    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cpu',
        enable_face_extraction=False,  # 不需要人脸提取
        face_pad_ratio=0.5,
    )

    pretrained_model = detector.model
    print('  预训练模型加载成功')
    print()

    # Step 2: 创建微调模型
    print('Step 2: 创建微调模型...')
    model = SimpleEyeDetector(pretrained_model, num_classes=2)
    model = model.to(device)
    print('  微调模型创建成功')
    print(f'  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print()

    # Step 3: 准备数据
    print('Step 3: 准备训练数据...')
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

    # 为验证集设置不同的 transform
    val_dataset.dataset = EyeFrameDataset(
        data_dir='/yanglin/eye_detect/training_data',
        transform=val_transform,
        frames_per_video=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False,  # 禁用 pin_memory 避免 CUDA 问题
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=False,  # 禁用 pin_memory 避免 CUDA 问题
    )

    print(f'  训练集: {train_size} 个视频')
    print(f'  验证集: {val_size} 个视频')
    print()

    # Step 4: 设置训练参数
    print('Step 4: 设置训练参数...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 20
    print(f'  Epochs: {num_epochs}')
    print(f'  学习率: 1e-4')
    print(f'  Batch size: 4')
    print()

    # Step 5: 开始训练
    print('Step 5: 开始微调...')
    print('-' * 80)

    best_accuracy = 0.0
    output_dir = Path('/yanglin/eye_detect/finetuned_model')
    output_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        elapsed = time.time() - start_time

        print(f'Epoch [{epoch+1:02d}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | '
              f'Time: {elapsed:.1f}s')

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_path = output_dir / 'best_finetuned.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f'  -> 保存最佳模型 (准确率: {val_acc:.1f}%)')

    print('-' * 80)
    print()
    print('=' * 80)
    print('微调完成！')
    print('=' * 80)
    print(f'最佳验证准确率: {best_accuracy:.1f}%')
    print(f'模型保存路径: {output_dir / "best_finetuned.pth"}')
    print()
    print('下一步：')
    print('  1. 使用微调后的模型重新运行批量检测')
    print('  2. 对比微调前后的性能')


if __name__ == '__main__':
    main()
