import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxPool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxPool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxPool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.maxPool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

