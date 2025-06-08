from captcha.image import ImageCaptcha
import random
import os
from config import config
from tqdm import tqdm


def generate_captcha_dataset(num_samples):
    """
    生成指定数量的验证码图像

    参数:
        num_samples (int): 要生成的验证码数量
    """
    # 确保输出目录存在
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    print(f"将在 '{config.DATASET_DIR}' 目录中生成 {num_samples} 个验证码图像...")

    # 创建ImageCaptcha对象，设置图像大小
    image = ImageCaptcha(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT)

    # 生成验证码
    for i in tqdm(range(num_samples)):
        # 随机生成长度为4-5的验证码
        length = random.randint(4, 5)
        captcha_text = ''.join(random.choices(config.CHAR_SET, k=length))

        # 生成图像
        captcha_image = image.generate_image(captcha_text)

        # 保存图像 (文件名为: captcha_<验证码文本>.png)
        filename = f"captcha_{captcha_text}.png"
        captcha_image.save(os.path.join(config.DATASET_DIR, filename))

    print(f"成功生成 {num_samples} 个验证码图像到 '{config.DATASET_DIR}' 目录")


if __name__ == "__main__":
    # 获取用户输入
    num_samples = int(input("请输入要生成的验证码数量: "))

    # 生成数据集
    generate_captcha_dataset(num_samples)