import time
import torch
import simpleCNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 常量
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.5
NUM_EPOCHS = 50

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_load():
    # CIFAR-10 数据预处理, 归一化分别为RGB三通道的均值和标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 数据加载
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


# 训练函数
def train(model, optimizer, criterion, train_loader, test_loader, train_accs, test_accs):
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)                # 获得输出
            loss = criterion(output, target)    # 计算损失
            train_loss += loss.item()
            optimizer.zero_grad()               # 梯度清零
            loss.backward()                     # 反向传播
            optimizer.step()                    # 参数更新

            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

        train_acc = correct_train / total_train
        train_accs.append(train_acc)

        # 在测试集上评估模型
        test_acc = test(model, criterion, test_loader, test_accs)
        print(f'Epoch {epoch} Train Accuracy: {100 * train_acc:.2f}%, Test Accuracy: {100 * test_acc:.2f}%')
        # print(f'Epoch {epoch} Loss: {train_loss / len(train_loader):.6f} Accuracy: {100 * train_acc:.2f}%')

    return


# 测试函数
def test(model, criterion, test_loader, test_accs):
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

    test_acc = correct_test / total_test
    test_accs.append(test_acc)

    return test_acc


if __name__ == '__main__':
    start_time = time.time()

    model = simpleCNN.SimpleCNN().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # 训练数据
    train_loader, test_loader = data_load()

    # 训练 Accuracy
    train_accs = []
    test_accs = []

    train(model, optimizer, criterion, train_loader, test_loader, train_accs, test_accs)

    end_time = time.time()

    print("Total time %s s" % (end_time - start_time))

    # 绘制学习曲线
    plt.plot(range(1, NUM_EPOCHS + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, NUM_EPOCHS + 1), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
