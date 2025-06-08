import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import config
import string
import re
import torch


# ----------------------------
# 编码解码函数
# ----------------------------

def encode_captcha(captcha_text):
    """
    将验证码文本转换为多标签张量

    参数:
        captcha_text (str): 验证码文本

    返回:
        torch.LongTensor: 形状为[MAX_CAPTCHA_LEN]的张量，包含每个字符的索引
    """
    # 创建填充为空白字符的标签张量
    labels = torch.full((config.MAX_CAPTCHA_LEN,),
                        len(config.CHAR_SET),
                        dtype=torch.long)

    # 对验证码中的每个字符进行编码
    for i, char in enumerate(captcha_text):
        # 如果已经达到最大长度，则停止
        if i >= config.MAX_CAPTCHA_LEN:
            break

        # 如果字符在字符集中，设置对应的标签值
        if char in config.CHAR_SET:
            labels[i] = config.CHAR_SET.index(char)

    return labels


def decode_predictions(predictions):
    """
    将模型输出转换回文本

    参数:
        predictions (torch.Tensor): 模型输出的张量，形状为[MAX_CAPTCHA_LEN, num_classes]
                                    或[batch_size, MAX_CAPTCHA_LEN, num_classes]

    返回:
        str 或 list[str]: 解码后的验证码文本
    """
    # 确保输入是3D张量 [batch_size, MAX_CAPTCHA_LEN, num_classes]
    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(0)

    decoded_texts = []

    # 对批次中的每个预测进行处理
    for batch_pred in predictions:
        # 获取每个位置概率最高的字符索引
        char_indices = batch_pred.argmax(dim=-1)

        # 转换索引为字符
        captcha_text = ""
        for idx in char_indices:
            # 如果索引在有效范围内，添加到字符串
            if idx < len(config.CHAR_SET):
                captcha_text += config.CHAR_SET[idx]
            # 如果遇到空白字符，停止解码
            else:
                break

        decoded_texts.append(captcha_text)

    # 如果只有一个结果，直接返回字符串
    return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts


# 新增的函数：专门解码标签数据
def decode_labels(labels):
    """
    将标签张量转换回文本

    参数:
        labels (torch.Tensor): 标签张量，形状为[MAX_CAPTCHA_LEN]或[batch_size, MAX_CAPTCHA_LEN]

    返回:
        str 或 list[str]: 解码后的验证码文本
    """
    # 确保输入是2D张量 [batch_size, MAX_CAPTCHA_LEN]
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)

    decoded_texts = []

    # 对批次中的每个标签进行处理
    for label_sequence in labels:
        # 转换索引为字符
        captcha_text = ""
        for idx in label_sequence:
            # 如果索引在有效范围内，添加到字符串
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx_val < len(config.CHAR_SET):
                captcha_text += config.CHAR_SET[idx_val]
            # 如果遇到空白字符，停止解码
            else:
                break

        decoded_texts.append(captcha_text)

    # 如果只有一个结果，直接返回字符串
    return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts


# ----------------------------
# 数据预处理函数
# ----------------------------

def get_transform(is_train=True):
    """
    获取数据预处理和增强流程

    参数:
        is_train (bool): 是否为训练集模式，训练集使用更多增强

    返回:
        transforms.Compose: 数据预处理流程
    """
    # 训练集使用增强
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # 测试集使用基本预处理
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ----------------------------
# 数据集类
# ----------------------------

class CaptchaDataset(torch.utils.data.Dataset):
    """
    自定义验证码数据集类

    参数:
        root_dir (str): 数据集根目录
        transform (callable, optional): 数据转换/增强函数
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # 遍历目录收集图像和标签
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 从文件名提取验证码文本
                # 支持格式: captcha_abc123.png, abc123.jpg, 等等
                match = re.search(r'([a-zA-Z0-9]{4,5})$', os.path.splitext(filename)[0])
                if match:
                    captcha_text = match.group(1).lower()
                    self.image_files.append(os.path.join(root_dir, filename))
                    self.labels.append(encode_captcha(captcha_text))
                else:
                    print(f"警告: 忽略不标准文件名的图像: {filename}")

        self.num_samples = len(self.image_files)
        print(f"加载 {self.num_samples} 个样本")

        # 如果没有找到样本，发出警告
        if self.num_samples == 0:
            print(f"警告: 目录 {root_dir} 中没有找到有效的验证码图像!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        # 打开图像并确保是RGB格式
        image = Image.open(img_path).convert('RGB')

        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# 模型工具函数
# ----------------------------

def save_model(model, optimizer, epoch, train_loss, val_loss, val_acc):
    """
    保存模型状态到文件

    参数:
        model (nn.Module): 要保存的模型
        optimizer (optim.Optimizer): 优化器状态
        epoch (int): 当前训练轮次
        train_loss (float): 当前训练损失
        val_loss (float): 当前验证损失
        val_acc (float): 当前验证准确率
    """
    # 确保保存目录存在
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 创建文件路径
    save_path = os.path.join(config.SAVE_DIR, f"captcha_model_epoch_{epoch}.pth")

    # 保存模型状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'char_set': config.CHAR_SET,
        'max_len': config.MAX_CAPTCHA_LEN
    }, save_path)

    print(f"保存模型到: {save_path}")

    # 额外保存一个最佳模型副本
    best_path = os.path.join(config.SAVE_DIR, "captcha_model_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_set': config.CHAR_SET,
        'max_len': config.MAX_CAPTCHA_LEN
    }, best_path)
    print(f"保存最佳模型到: {best_path}")


def load_model(model, optimizer=None, path=None):
    """
    加载模型状态

    参数:
        model (nn.Module): 要加载状态的模型
        optimizer (optim.Optimizer, optional): 要加载状态的优化器
        path (str, optional): 模型文件路径，默认自动查找最新模型

    返回:
        tuple: (加载后的模型, 加载后的优化器, 训练轮次)
    """
    # 如果未提供路径，自动查找最新模型
    if path is None:
        # 获取所有模型文件
        checkpoint_files = [f for f in os.listdir(config.SAVE_DIR) if f.endswith('.pth')]

        # 如果没有模型文件，抛出错误
        if not checkpoint_files:
            raise FileNotFoundError(f"未找到模型文件，请检查目录: {config.SAVE_DIR}")

        # 按轮次排序找到最新模型
        checkpoint_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
        path = os.path.join(config.SAVE_DIR, checkpoint_files[-1])

    # 加载模型
    print(f"从 {path} 加载模型...")
    checkpoint = torch.load(path, map_location=config.DEVICE)

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)

    # 如果有优化器，加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 获取训练轮次
    epoch = checkpoint.get('epoch', 0)

    # 获取验证准确率（如果有）
    val_acc = checkpoint.get('val_acc', 0.0)

    print(f"模型加载完成 (轮次 {epoch}, 验证准确率 {val_acc:.4f})")
    return model, optimizer, epoch


# ----------------------------
# 数据可视化函数
# ----------------------------

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir=None):
    """
    可视化训练历史

    参数:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        train_accs (list): 训练准确率列表
        val_accs (list): 验证准确率列表
        save_dir (str, optional): 保存图像的目录
    """
    # 创建图表
    plt.figure(figsize=(15, 6))

    # 损失图表
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss', marker='o')
    plt.plot(val_losses, label='Verify loss', marker='o')
    plt.title('Training loss and Verify loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 准确率图表
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training accuracy', marker='o')
    plt.plot(val_accs, label='Verify accuracy', marker='o')
    plt.title('Train accuracy and Validate accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"训练历史图已保存到: {save_path}")
    else:
        plt.show()


def visualize_predictions(model, loader, save_dir=None, num_samples=5):
    """
    可视化模型在测试集上的预测结果

    参数:
        model (nn.Module): 训练好的模型
        loader (DataLoader): 数据加载器
        save_dir (str, optional): 保存图像的目录
        num_samples (int): 可视化的样本数量
    """
    # 切换到评估模式
    model.eval()

    # 获取一批数据
    images, labels = next(iter(loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 预测
    with torch.no_grad():
        outputs = model(images.to(config.DEVICE))

    # 修复：使用新的decode_labels函数处理标签
    true_texts = [decode_labels(label) for label in labels]

    # 修复：为每个样本单独处理模型输出
    pred_texts = []
    for output in outputs:
        pred_texts.append(decode_predictions(output))

    # 可视化结果
    plt.figure(figsize=(15, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)

        # 显示图像（需要反归一化）
        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.title(f"The Truth: {true_texts[i]}  The prediction: {pred_texts[i]}",
                  color='green' if pred_texts[i] == true_texts[i] else 'red')
        plt.axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测样本图已保存到: {save_path}")
    else:
        plt.show()