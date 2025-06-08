# Text Captcha Destroyer

## 这是什么

本项目是基于一种基于PyTorch深度学习的文字验证码识别模型，可以识别数字（1~9）和所有大小写英文字母(A~Z, a~z)。有效识别的验证码位数为4~5位，覆盖了市面上大多数的文字验证码

## 项目中的这些程序都是什么

- 本项目设有`config.py`程序，可以自定义配置设置（例如是否使用GPU）、数据参数、训练参数、路径配置
- `main.py`是本项目的主程序，也是训练模型的程序。初始训练轮数（在`config.py`中定义）为50轮，每次执行时都会尝试加载已有模型。每轮训练结束后都会保留当前最佳模型，每5轮训练结束后会使用`matplotbib`生成可视化的训练情况折线图以及当前的预测分析结果。模型和分析图片将被保存在`captcha_models`文件夹中

- 本项目设有`get_dataset.py`程序，调用了`captcha`模块用来自动生成指定数量的示例验证码作为训练集，并将训练集保存在`captcha_dataset`文件夹中

- 本项目设有`guess.py`程序，可以指定待测验证码的路径并使用已经训练好的模型进行检验
- 有关模型的基本训练原理定义在`model.py`中
- 程序`utils.py`是函数集，包含了编码解码函数等多个要用到的函数

## 如何使用本项目

首先，确保你已经安装了Python且版本最好为3.10及以上，并已经有了git

### 配置环境

运行以下命令来将本项目克隆到本地：

```bash
git clone https://github.com/Drtxdt/Text-Captcha-Destroyer.git
```

然后，切换到项目目录：

```bash
cd Text-Captcha-Destroyer
```

运行以下命令来安装依赖：

```bash
pip install -r requirements.txt
```

### 开始

首先，运行`get_dataset.py`来生成训练集。程序将自动询问想要的训练集规模，这里推荐至少10000

```bash
python get_dataset.py
```

然后，运行主程序进行训练

```bash
python main.py
```

训练时会生成图片报告，可以在文件夹`captcha_models`中找到

训练完成之后，可以运行验证程序进行评估

```bash
python guess.py
```

## 上文提到的那两个文件夹在哪里/我也没看到数据集呀

别急，为了使大家克隆仓库时更加顺利，我并没有上传已经生成好的数据集和模型。按照上文的运行方法就可以自动生成`captcha_dataset`和`captcha_models`这两个文件夹了

如果受某些原因影响，您就是想要开箱即用的数据集或者模型，请移步Release部分来下载压缩包，在和项目相同的目录下解压即可

