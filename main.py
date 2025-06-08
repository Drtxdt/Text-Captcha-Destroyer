import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from config import config
from model import CaptchaModel
from utils import (get_transform, CaptchaDataset,
                   save_model, load_model, encode_captcha,
                   decode_predictions, plot_training_history,
                   visualize_predictions)


def setup_environment():
    """设置运行环境，打印重要信息"""
    print("\n" + "=" * 50)
    print(f"验证码识别训练系统")
    print(f"运行设备: {config.DEVICE}")
    print(f"验证码字符集: {config.CHAR_SET} (共 {len(config.CHAR_SET)} 个字符)")
    print(f"最大验证码长度: {config.MAX_CAPTCHA_LEN}")
    print(f"图像尺寸: {config.IMAGE_HEIGHT}x{config.IMAGE_WIDTH}")
    print("=" * 50 + "\n")

    # 创建必要的目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.DATASET_DIR, exist_ok=True)


def check_dataset():
    """检查数据集是否存在"""
    if not os.path.exists(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print(f"\n{'!' * 20} 错误 {'!' * 20}")
        print(f"数据集目录 '{config.DATASET_DIR}' 为空或不存在!")
        print("请先运行 get_dataset.py 生成数据集")
        print(f"使用命令: python get_dataset.py")
        print(f"{'!' * 48}\n")
        return False
    return True


def calculate_metrics(outputs, labels, criterion=None):
    """计算模型性能指标（损失、字符准确率、整体准确率）"""
    # 初始化指标
    metrics = {
        'loss': 0.0,
        'char_acc': 0.0,
        'whole_acc': 0.0,
        'correct_chars': 0,
        'total_chars': 0,
        'correct_whole': 0,
        'total_samples': outputs.size(0)
    }

    # 如果有损失函数，计算损失
    if criterion:
        for i in range(config.MAX_CAPTCHA_LEN):
            metrics['loss'] += criterion(outputs[:, i, :], labels[:, i]).item()

    # 获取预测结果
    _, preds = torch.max(outputs, dim=2)  # [batch, MAX_CAPTCHA_LEN]

    # 字符准确率（忽略空白位置）
    mask = labels != len(config.CHAR_SET)  # 只计算有效字符位置
    metrics['correct_chars'] = (preds[mask] == labels[mask]).sum().item()
    metrics['total_chars'] = mask.sum().item()
    metrics['char_acc'] = metrics['correct_chars'] / metrics['total_chars'] if metrics['total_chars'] > 0 else 0

    # 整体验证码准确率
    for i in range(outputs.size(0)):
        # 计算实际验证码长度 - 解决类型警告问题
        # 使用显式类型转换解决IDE对布尔张量的警告
        mask_i = labels[i] < len(config.CHAR_SET)  # 布尔张量
        real_length = torch.sum(mask_i).item()  # 显式使用torch.sum()计算长度

        # 只比较有效部分
        if torch.equal(preds[i, :real_length], labels[i, :real_length]):
            metrics['correct_whole'] += 1

    metrics['whole_acc'] = metrics['correct_whole'] / metrics['total_samples'] if metrics['total_samples'] > 0 else 0

    return metrics, preds


def train_epoch(model, loader, optimizer, criterion, epoch):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_whole = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc=f"训练轮次 {epoch + 1}", ncols=100)
    for images, labels in progress_bar:
        # 转移数据到设备
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)  # [batch, MAX_CAPTCHA_LEN, num_classes]

        # 计算损失
        metrics, _ = calculate_metrics(outputs, labels, criterion)
        loss = criterion(outputs[:, 0, :], labels[:, 0])  # 使用任意一个位置的损失作为基准
        for i in range(1, config.MAX_CAPTCHA_LEN):
            loss += criterion(outputs[:, i, :], labels[:, i])

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累加指标
        total_loss += metrics['loss']
        correct_chars += metrics['correct_chars']
        total_chars += metrics['total_chars']
        correct_whole += metrics['correct_whole']
        total_samples += metrics['total_samples']

        # 更新进度条
        avg_loss = total_loss / len(progress_bar)
        char_acc = correct_chars / total_chars if total_chars > 0 else 0
        whole_acc = correct_whole / total_samples if total_samples > 0 else 0

        progress_bar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            '字符准确率': f"{char_acc:.4f}",
            '整体准确率': f"{whole_acc:.4f}"
        })

    # 计算平均指标
    epoch_loss = total_loss / (len(loader) * config.MAX_CAPTCHA_LEN)
    epoch_char_acc = correct_chars / total_chars if total_chars > 0 else 0
    epoch_whole_acc = correct_whole / total_samples if total_samples > 0 else 0

    return epoch_loss, epoch_char_acc, epoch_whole_acc


def validate_model(model, loader, criterion):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_whole = 0
    total_samples = 0

    # 收集预测结果用于可视化
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="验证中", ncols=100):
            # 转移数据到设备
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            # 前向传播
            outputs = model(images)

            # 计算指标
            metrics, preds = calculate_metrics(outputs, labels, criterion)

            # 累加指标
            total_loss += metrics['loss']
            correct_chars += metrics['correct_chars']
            total_chars += metrics['total_chars']
            correct_whole += metrics['correct_whole']
            total_samples += metrics['total_samples']

            # 保存样本用于可视化
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # 计算平均指标
    epoch_loss = total_loss / (len(loader) * config.MAX_CAPTCHA_LEN)
    epoch_char_acc = correct_chars / total_chars if total_chars > 0 else 0
    epoch_whole_acc = correct_whole / total_samples if total_samples > 0 else 0

    # 准备可视化数据
    sample_images = torch.cat(all_images)[:20]  # 取前20个样本
    sample_labels = torch.cat(all_labels)[:20]
    sample_preds = torch.cat(all_preds)[:20]

    return epoch_loss, epoch_char_acc, epoch_whole_acc, sample_images, sample_labels, sample_preds


def main():
    """主训练函数"""
    # 设置环境
    setup_environment()

    # 检查数据集
    if not check_dataset():
        return

    # 初始化模型
    model = CaptchaModel(len(config.CHAR_SET) + 1)  # +1 用于空白字符
    model.to(config.DEVICE)
    print(f"模型架构已创建, 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 尝试加载现有模型
    try:
        model, _, start_epoch = load_model(model, path=config.MODEL_PATH)
        print(f"从第 {start_epoch + 1} 轮继续训练...")
    except Exception as e:
        print(f"无法加载现有模型: {str(e)}")
        print("将从头开始训练新模型")
        start_epoch = 0

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss(ignore_index=len(config.CHAR_SET))  # 忽略空白字符
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.5, patience=3
    )

    # 加载数据集
    print(f"\n加载数据集: {config.DATASET_DIR}")
    transform = get_transform(is_train=True)
    dataset = CaptchaDataset(config.DATASET_DIR, transform=transform)

    # 划分训练集和测试集
    train_size = int(config.TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # 初始化记录变量
    train_losses, val_losses = [], []
    train_char_accs, val_char_accs = [], []
    train_whole_accs, val_whole_accs = [], []
    best_whole_acc = 0.0

    # 训练和评估循环
    for epoch in range(start_epoch, config.EPOCHS):
        # 训练一个epoch
        train_loss, train_char_acc, train_whole_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        # 记录训练指标
        train_losses.append(train_loss)
        train_char_accs.append(train_char_acc)
        train_whole_accs.append(train_whole_acc)

        # 验证
        val_loss, val_char_acc, val_whole_acc, _, _, _ = validate_model(
            model, test_loader, criterion
        )

        # 记录验证指标
        val_losses.append(val_loss)
        val_char_accs.append(val_char_acc)
        val_whole_accs.append(val_whole_acc)

        # 更新学习率
        scheduler.step(val_whole_acc)

        # 打印epoch结果
        print(f"\n轮次 {epoch + 1}/{config.EPOCHS} 结果:")
        print(f"  训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"  训练字符准确率: {train_char_acc:.4f} | 验证字符准确率: {val_char_acc:.4f}")
        print(f"  训练整体准确率: {train_whole_acc:.4f} | 验证整体准确率: {val_whole_acc:.4f}")

        # 保存模型和最佳模型
        if val_whole_acc > best_whole_acc:
            best_whole_acc = val_whole_acc
            save_model(model, optimizer, epoch, train_loss, val_loss, val_whole_acc)
            print(f"  保存新模型: 验证整体准确率 {val_whole_acc:.4f}")

        # 定期保存训练历史和可视化
        if (epoch + 1) % 5 == 0 or epoch == config.EPOCHS - 1:
            # 保存训练历史
            plot_training_history(
                train_losses, val_losses,
                train_whole_accs, val_whole_accs,
                save_dir=config.SAVE_DIR
            )

            # 可视化一些预测样本
            visualize_predictions(
                model, test_loader,
                save_dir=config.SAVE_DIR,
                num_samples=5
            )

    print("\n训练完成!")
    print(f"最佳验证整体准确率: {best_whole_acc:.4f}")


if __name__ == "__main__":
    main()