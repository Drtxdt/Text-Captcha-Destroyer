import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from config import config
from model import CaptchaModel
from utils import get_transform, decode_predictions, load_model


def predict_captcha(model, image_path, save_dir=None):
    """
    预测验证码图像

    参数:
        model: 训练好的验证码识别模型
        image_path: 验证码图像文件路径
        save_dir: 保存预测结果的目录（可选）

    返回:
        str: 预测的验证码文本
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在!")
        return None

    try:
        # 打开图像并确保是RGB格式
        image = Image.open(image_path).convert('RGB')

        # 获取测试转换
        transform = get_transform(is_train=False)

        # 应用变换（添加批次维度）
        image_tensor = transform(image).unsqueeze(0)

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return None

    # 预测
    model.eval()
    with torch.no_grad():
        try:
            # 将图像移动到设备并预测
            image_tensor = image_tensor.to(config.DEVICE)
            output = model(image_tensor)  # [1, MAX_CAPTCHA_LEN, num_classes]

            # 解码预测结果
            prediction = decode_predictions(output[0])
        except Exception as e:
            print(f"预测验证码时出错: {str(e)}")
            return None

    # 创建可视化图像
    plt.figure(figsize=(8, 6))

    # 显示原始图像
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title(f"原始验证码图像: {os.path.basename(image_path)}")
    plt.axis('off')

    # 添加预测结果文本
    plt.subplot(2, 1, 2)
    plt.text(0.5, 0.5, f"预测结果: {prediction}",
             fontsize=16, ha='center', va='center')
    plt.axis('off')

    plt.suptitle(f"验证码识别结果: {prediction}", fontsize=18)
    plt.tight_layout()

    # 保存或显示结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"prediction_{os.path.basename(image_path)}"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"预测结果已保存到: {save_path}")
    else:
        plt.show()

    return prediction


def main():
    """主预测函数"""
    print("\n" + "=" * 50)
    print("验证码识别预测系统")
    print(f"使用设备: {config.DEVICE}")
    print("=" * 50 + "\n")

    # 初始化模型
    model = CaptchaModel(len(config.CHAR_SET) + 1).to(config.DEVICE)
    print(f"已加载模型结构")

    # 尝试加载预训练模型
    try:
        model, _, _ = load_model(model, path=os.path.join(config.SAVE_DIR, "captcha_model_best.pth"))
        print(f"成功加载预训练模型")
    except Exception as e:
        print(f"无法加载模型: {str(e)}")
        print("请确保您已训练模型 (运行python main.py)")
        return

    while True:
        # 获取用户输入
        print("\n" + "-" * 50)
        print("请输入要预测的验证码图片路径 (或输入 'q' 退出)")
        image_path = input("图片路径: ").strip()

        # 退出条件
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("\n感谢使用验证码识别系统!")
            break

        # 验证输入
        if not image_path:
            print("错误: 请输入有效的文件路径")
            continue

        # 进行预测
        prediction = predict_captcha(model, image_path, save_dir="captcha_predictions")

        if prediction is not None:
            print(f"\n预测结果: {prediction}")
        else:
            print("预测失败，请检查输入图像")

        # 间隔符
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()