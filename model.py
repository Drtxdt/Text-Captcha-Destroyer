import torch
import torch.nn as nn
from config import config


class CaptchaModel(nn.Module):
    def __init__(self, num_classes):
        super(CaptchaModel, self).__init__()
        self.num_classes = num_classes

        # 特征提取器 - 处理3通道输入，逐步提取特征
        self.features = nn.Sequential(
            # 输入尺寸: [batch, 3, 64, 180]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: [batch, 32, 32, 90]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: [batch, 64, 16, 45]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),  # 输出尺寸: [batch, 128, 8, 45]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),  # 输出尺寸: [batch, 256, 4, 45]

            # 使用自适应池化层确保输出固定尺寸
            nn.AdaptiveAvgPool2d((4, 5))  # 输出尺寸: [batch, 256, 4, 5]
        )

        # 序列处理模块 - 处理展平后的特征
        self.sequence = nn.Sequential(
            nn.Flatten(start_dim=1),  # 展平为 [batch, 256 * 4 * 5 = 5120]
            nn.Linear(256 * 4 * 5, 512),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # 多输出分类器 - 为验证码中的每个字符位置创建一个独立的分类器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),  # 第一层全连接
                nn.ReLU(),
                nn.Linear(256, num_classes)  # 第二层全连接输出到num_classes
            ) for _ in range(config.MAX_CAPTCHA_LEN)  # 为每个验证码位置创建相同的分类器
        ])

    def forward(self, x):
        # 特征提取
        x = self.features(x)  # 输出形状: [batch, 256, 4, 5]

        # 序列处理
        x = self.sequence(x)  # 输出形状: [batch, 512]

        # 每个位置独立的分类器
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))  # 每个分类器输出形状: [batch, num_classes]

        # 组合所有分类器的输出
        # 输出形状: [batch, MAX_CAPTCHA_LEN, num_classes]
        return torch.stack(outputs, dim=1)