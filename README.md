# 分布式攻防实验
## 实验数据集

- 本实验使用的数据集为 $\text{CIFAR-10}$，是常用于识别普适物体的经典数据集。共包含 $10$ 个类别的 $3$ 通道彩色 $\text{RGB}$ 图像：飞机、汽车、鸟类、猫、鹿、狗、蛙类、马、船和卡车。图片的尺寸为 $32×32$ ，数据集中一共有 $50000$ 张训练图片和 $10000$ 张测试图片。图片样例如下图所示：

<div align=center>
<img src="\pic\image-cifar.png" alt="img" style="zoom: 80%;" />
</div>

## 评估指标

- 本实验使用准确率 ($\text{Accuracy}$) 作为评估标准，其公式如下：

$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FN+FP}
$$

<img src="\pic\image-20240518125358539.png" alt="image-20240518125358539" style="zoom: 67%;" />

## 基础模型

- 本实验采用 $\text{CIFAR 10 model}$ 卷积神经网络模型，具体架构如下所示：

<img src="\pic\image-20240518114827642.png" alt="image-20240518114827642" style="zoom:50%;" />

## 实验参数

- 本实验使用基于动量的梯度下降算法，设定学习率 $α=0.01$ 、动量 $μ=0.5$，训练批次大小 $\text{Batch Size}=128$、训练轮数 $\text{Epoch} = 60$，同时设定分布式学习模型的工作节点数为 $13$ 。

## 文件结构

- 各文件的功能为：
  - $\text{simpleCNN.py}$：封装有基于 $\text{PyTorch}$ 的 $\text{CIFAR 10 model}$ 类
  - $\text{krum.py}$：封装有实现 $\text{Krum}$ 和 $\text{Multi Krum}$ 算法的函数
  - $\text{normal tarin.py}$：非分布式学习模型的代码
  - $\text{distributed main.py}$：分布式学习模型的启动代码
  - $\text{distributed train.py}$：分布式学习模型的主体代码 (含分布式攻防参数与代码)