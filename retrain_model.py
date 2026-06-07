#!/usr/bin/env python3
"""
重新训练 UniFormerV2 闭眼检测模型

使用增强后的数据重新训练模型，提高闭眼检测能力。

用法：
    python retrain_model.py
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
    帧数据集 - 兼容 SlowFast 框架

    从 JSON 标注文件加载数据
    """

    def __init__(self, json_path: str, transform=None, num_frames: int = 8):
        """
        Args:
            json_path: JSON 标注文件路径
            transform: 图像变换
            num_frames: 每个视频采样的帧数
        """
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
            # 均匀采样
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            # 重复采样
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
                # 如果帧加载失败，使用黑色帧
                if self.transform:
                    dummy = Image.new('RGB', (224, 224))
                    frame_tensors.append(self.transform(dummy))
                else:
                    frame_tensors.append(torch.zeros(3, 224, 224))

        # 堆叠帧: (T, C, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)

        return video_tensor, label


class SimpleEyeModel(nn.Module):
    """
    简单的闭眼检测模型

    使用预训练的 UniFormerV2 作为特征提取器，
    添加一个简单的分类头。
    """

    def __init__(self, pretrained_model, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone = pretrained_model

        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        if x.dim() == 5:
            x = x.permute(0, 2, 1, 3, 4)

        # 提取特征
        with torch.no_grad():
            # 使用模型的内部处理
            # 需要将输入转换为模型期望的格式
            features = self._extract_features(x)

        # features 形状: (N, C) - 2D 张量
        # 分类头期望 5D 张量，需要 reshape
        # 将 (N, C) 转换为 (N, C, 1, 1, 1) 以适应 AdaptiveAvgPool3d
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 分类
        output = self.classifier(features)
        return output

    def _extract_features(self, x):
        """提取特征"""
        # 获取 backbone (Uniformerv2 包含一个 backbone 属性)
        backbone = self.backbone.backbone if hasattr(self.backbone, 'backbone') else self.backbone

        # 尝试直接获取模型输出
        try:
            # 方法: 使用模型的 conv1 和 transformer
            # 输入形状: (B, C, T, H, W)
            B, C, T, H, W = x.shape

            # 通过 conv1
            # 输出形状: (B, 768, T_out, H_out, W_out)
            x = backbone.conv1(x)
            N, C, T_out, H_out, W_out = x.shape

            # reshape: (B, 768, T_out, H_out, W_out) -> (B * T_out, H_out * W_out, 768)
            x = x.permute(0, 2, 3, 4, 1).reshape(N * T_out, H_out * W_out, C)

            # 添加 cls token
            x = torch.cat([
                backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x
            ], dim=1)

            # 添加位置编码
            x = x + backbone.positional_embedding.to(x.dtype)
            x = backbone.ln_pre(x)

            # 通过 transformer 的 resblocks（不包括最后的 proj 层）
            # 我们需要手动遍历 resblocks
            for resblock in backbone.transformer.resblocks:
                x = resblock(x, T_out)

            # x 形状: (L, N*T_out, C)
            # 取 cls token (第一个位置)
            cls_features = x[0]  # (N*T_out, C)

            # reshape 并取平均
            cls_features = cls_features.reshape(N, T_out, -1).mean(dim=1)  # (N, C)

            return cls_features

        except Exception as e:
            print(f'特征提取错误: {e}')
            import traceback
            traceback.print_exc()
            # 返回零特征
            return torch.zeros(x.shape[0], 768)


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

    # 用于计算各类别准确率
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

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

            # 统计各类别
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)

    # 计算各类别准确率
    class_accuracy = {}
    for cls in [0, 1]:
        if class_total[cls] > 0:
            class_accuracy[cls] = 100. * class_correct[cls] / class_total[cls]
        else:
            class_accuracy[cls] = 0.0

    return avg_loss, accuracy, class_accuracy


def main():
    print('=' * 80)
    print('重新训练 UniFormerV2 闭眼检测模型')
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

    pretrained_model = detector.model
    print('  预训练模型加载成功')
    print()

    # Step 2: 创建训练模型
    print('Step 2: 创建训练模型...')
    model = SimpleEyeModel(pretrained_model, num_classes=2, freeze_backbone=True)
    model = model.to(device)
    print('  模型创建成功')
    print(f'  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print()

    # Step 3: 准备数据
    print('Step 3: 准备训练数据...')
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    print(f'  训练集: {len(train_dataset)} 个视频')
    print(f'  验证集: {len(val_dataset)} 个视频')
    print()

    # Step 4: 设置训练参数
    print('Step 4: 设置训练参数...')

    # 使用类别权重平衡数据
    # 闭眼样本较少，给予更高权重
    close_count = sum(1 for a in train_dataset.annotations if a['train_label'] == 1)
    open_count = sum(1 for a in train_dataset.annotations if a['train_label'] == 0)
    total_count = close_count + open_count

    weight_close = total_count / (2 * close_count) if close_count > 0 else 1.0
    weight_open = total_count / (2 * open_count) if open_count > 0 else 1.0

    class_weights = torch.FloatTensor([weight_open, weight_close])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6,
    )

    num_epochs = 50
    print(f'  Epochs: {num_epochs}')
    print(f'  学习率: 1e-4')
    print(f'  Batch size: 4')
    print(f'  类别权重: close={weight_close:.2f}, open={weight_open:.2f}')
    print()

    # Step 5: 开始训练
    print('Step 5: 开始训练...')
    print('-' * 80)

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
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        elapsed = time.time() - start_time

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_close_recall'].append(class_acc.get(1, 0))

        # 打印进度
        close_recall = class_acc.get(1, 0)
        open_recall = class_acc.get(0, 0)

        print(f'Epoch [{epoch+1:02d}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | '
              f'Close: {close_recall:.1f}% Open: {open_recall:.1f}% | '
              f'Time: {elapsed:.1f}s')

        # 保存最佳模型（基于闭眼召回率）
        if close_recall > best_close_recall:
            best_close_recall = close_recall
            best_accuracy = val_acc

            save_path = output_dir / 'best_retrained.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'close_recall': close_recall,
                'open_recall': open_recall,
            }, save_path)
            print(f'  -> 保存最佳模型 (闭眼召回率: {close_recall:.1f}%, 准确率: {val_acc:.1f}%)')

    print('-' * 80)
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
    print(f'  模型: {output_dir / "best_retrained.pth"}')
    print(f'  历史: {history_path}')
    print()
    print('下一步：')
    print('  1. 使用新模型重新运行批量检测')
    print('  2. 对比微调前后的性能')


if __name__ == '__main__':
    main()
