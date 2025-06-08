from pathlib import Path

import torch
import string


class Config:
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据参数
    IMAGE_HEIGHT = 64
    IMAGE_WIDTH = 180
    CHAR_SET = string.digits + string.ascii_lowercase + string.ascii_uppercase  # 36 + 26个字符: 0-9 + a-z
    MAX_CAPTCHA_LEN = 5  # 支持4-5位验证码

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.8
    RANDOM_SEED = 42
    NUM_WORKERS = 4

    # 路径配置
    SAVE_DIR = "captcha_models"
    DATASET_DIR = "captcha_dataset"
    MODEL_PATH = Path("captcha_models")


config = Config()